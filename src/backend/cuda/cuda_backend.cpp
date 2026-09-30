// CUDA / OptiX backend. Builds one GAS per mesh and one IAS over the scene instances, uploads the
// flat SceneView arrays to the device and launches the ray-generation program that runs the
// shared dense-band integrator (include/spectral/kernel/integrator.h).
//
// Requirements: CUDA >= 12.0, OptiX >= 8.0 (developed against OptiX 9 / CUDA 12.8).
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "launch_params.h"
#include "spectral/backend/backend.h"

// Preview launchers (preview_kernels.cu).
cudaError_t spectral_cuda_preview(const float* film, const float* depth, const uint32_t* seg, int w, int h, int n,
                                  const spectral::PreviewParams& p, uint8_t* out, cudaStream_t stream);
cudaError_t spectral_cuda_luminance(const float* film, int w, int h, int n, const float* cmf, float inv_spp, int stride,
                                    float* out, int* out_w, int* out_h, cudaStream_t stream);

// Generated at build time from programs/device_programs.cu (PTX), see cmake/CudaBackend.cmake.
extern "C" const unsigned char spectral_optix_ptx[];
extern "C" const unsigned long long spectral_optix_ptx_size;

namespace spectral {

namespace {

#define CUDA_CHECK(call)                                                                              \
  do {                                                                                               \
    cudaError_t e_ = (call);                                                                         \
    if (e_ != cudaSuccess)                                                                           \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " #call); \
  } while (0)

#define OPTIX_CHECK(call)                                                                                  \
  do {                                                                                                    \
    OptixResult r_ = (call);                                                                              \
    if (r_ != OPTIX_SUCCESS)                                                                              \
      throw std::runtime_error(std::string("OptiX error: ") + optixGetErrorName(r_) + " (" +                \
                               optixGetErrorString(r_) + ") at " #call);                                  \
  } while (0)

// RAII device buffer.
struct DeviceBuffer {
  void* ptr = nullptr;
  size_t size = 0;
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), size(o.size) { o.ptr = nullptr; o.size = 0; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      release();
      ptr = o.ptr;
      size = o.size;
      o.ptr = nullptr;
      o.size = 0;
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }
  void release() {
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
    size = 0;
  }
  void alloc(size_t n) {
    if (n == size && ptr) return;
    release();
    if (n) CUDA_CHECK(cudaMalloc(&ptr, n));
    size = n;
  }
  template <class T>
  void upload(const std::vector<T>& v) {
    alloc(v.size() * sizeof(T));
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(ptr, v.data(), size, cudaMemcpyHostToDevice));
  }
  void upload(const void* data, size_t n) {
    alloc(n);
    if (n) CUDA_CHECK(cudaMemcpy(ptr, data, n, cudaMemcpyHostToDevice));
  }
  CUdeviceptr dptr() const { return reinterpret_cast<CUdeviceptr>(ptr); }
  template <class T>
  T* as() const { return static_cast<T*>(ptr); }
};

template <class T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};
struct EmptyData {};

void optix_log(unsigned level, const char* tag, const char* msg, void*) {
  if (level <= 2) std::fprintf(stderr, "[optix][%u][%s] %s\n", level, tag, msg);
}

class CudaBackend final : public Backend {
 public:
  CudaBackend() {
    CUDA_CHECK(cudaFree(nullptr));  // initialize the primary context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions opts = {};
    opts.logCallbackFunction = &optix_log;
    opts.logCallbackLevel = 2;
    OPTIX_CHECK(optixDeviceContextCreate(/*cuCtx=*/0, &opts, &context_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    create_pipeline();
  }

  ~CudaBackend() override {
    for (cudaTextureObject_t t : textures_) cudaDestroyTextureObject(t);
    for (cudaArray_t a : arrays_) cudaFreeArray(a);
    if (pipeline_) optixPipelineDestroy(pipeline_);
    for (OptixProgramGroup g : {raygen_pg_, miss_pg_, miss_shadow_pg_, hit_pg_})
      if (g) optixProgramGroupDestroy(g);
    if (module_) optixModuleDestroy(module_);
    if (stream_) cudaStreamDestroy(stream_);
    if (context_) optixDeviceContextDestroy(context_);
  }

  std::string name() const override { return "cuda"; }

  void prepare(const Scene& scene) override {
    gas_.clear();
    gas_handles_.clear();
    for (cudaTextureObject_t t : textures_) cudaDestroyTextureObject(t);
    for (cudaArray_t a : arrays_) cudaFreeArray(a);
    textures_.clear();
    arrays_.clear();
    tex_views_host_.clear();
    tex_views_.release();
    lut_data_.release();
    uploaded_positions_ = size_t(-1);
    uploaded_env_version_ = uint64_t(-1);
    upload_scene(scene);
    build_accels(scene);
  }

  // Incremental: geometry arrays only when they grew, GAS only for new meshes, textures only new
  // ones, environment only when its version changed; instance/material/light buffers and the IAS
  // are refreshed every time (cheap).
  void update(const Scene& scene) override {
    upload_scene(scene);
    build_accels(scene);
  }

  void preview(const FilmBuffers& film, const PreviewParams& params, std::vector<uint8_t>& rgb) override {
    if (!film_radiance_.ptr) return Backend::preview(film, params, rgb);
    upload_cmf(params.cmf, film.n_bands);
    PreviewParams p = params;
    p.cmf = cmf_.as<float>();
    const size_t n = size_t(film.width) * film.height * 3;
    preview_buf_.alloc(n);
    CUDA_CHECK(spectral_cuda_preview(film_radiance_.as<float>(), film_depth_.as<float>(), film_seg_.as<uint32_t>(),
                                     film.width, film.height, film.n_bands, p, preview_buf_.as<uint8_t>(), stream_));
    rgb.resize(n);
    CUDA_CHECK(cudaMemcpyAsync(rgb.data(), preview_buf_.ptr, n, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  void luminance(const FilmBuffers& film, const float* cmf, float inv_spp, int stride,
                 std::vector<float>& out) override {
    if (!film_radiance_.ptr) return Backend::luminance(film, cmf, inv_spp, stride, out);
    upload_cmf(cmf, film.n_bands);
    lum_buf_.alloc(size_t(film.width / stride + 1) * (film.height / stride + 1) * sizeof(float));
    int ow = 0, oh = 0;
    CUDA_CHECK(spectral_cuda_luminance(film_radiance_.as<float>(), film.width, film.height, film.n_bands,
                                       cmf_.as<float>(), inv_spp, stride, lum_buf_.as<float>(), &ow, &oh, stream_));
    out.resize(size_t(ow) * oh);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), lum_buf_.ptr, out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  void render_pass(const Scene& scene, FilmBuffers& film, int first_sample, int count) override {
    const size_t npix = size_t(film.width) * film.height;
    if (film_radiance_.size != npix * film.n_bands * sizeof(float) || first_sample == 0) {
      film_radiance_.alloc(npix * film.n_bands * sizeof(float));
      film_depth_.alloc(npix * sizeof(float));
      film_seg_.alloc(npix * sizeof(uint32_t));
      CUDA_CHECK(cudaMemset(film_radiance_.ptr, 0, film_radiance_.size));
    }
    cuda::LaunchParams lp;
    lp.scene = scene_view_;
    lp.scene.camera = scene.camera;
    lp.scene.params = scene.params;
    lp.film.width = film.width;
    lp.film.height = film.height;
    lp.film.n_bands = film.n_bands;
    lp.film.radiance = film_radiance_.as<float>();
    lp.film.depth = film_depth_.as<float>();
    lp.film.seg_id = film_seg_.as<uint32_t>();
    lp.handle = ias_handle_;
    lp.first_sample = first_sample;
    lp.sample_count = count;
    params_buf_.upload(&lp, sizeof(lp));
    OPTIX_CHECK(optixLaunch(pipeline_, stream_, params_buf_.dptr(), sizeof(lp), &sbt_, unsigned(film.width),
                            unsigned(film.height), 1));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    film.spp_done += count;
  }

  void finish(FilmBuffers& film) override {
    if (!film_radiance_.ptr) return;
    CUDA_CHECK(cudaMemcpy(film.radiance_sum.data(), film_radiance_.ptr, film.radiance_sum.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(film.depth.data(), film_depth_.ptr, film.depth.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(film.seg_id.data(), film_seg_.ptr, film.seg_id.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
  }

 private:
  void create_pipeline() {
    OptixModuleCompileOptions mco = {};
    mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    pco_ = {};
    pco_.usesMotionBlur = 0;
    pco_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pco_.numPayloadValues = cuda::kNumPayloadValues;
    pco_.numAttributeValues = 2;
    pco_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco_.pipelineLaunchParamsVariableName = "params";
    pco_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[4096];
    size_t log_size = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(context_, &mco, &pco_, reinterpret_cast<const char*>(spectral_optix_ptx),
                                  size_t(spectral_optix_ptx_size), log, &log_size, &module_));

    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc d[4] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    d[0].raygen.module = module_;
    d[0].raygen.entryFunctionName = "__raygen__render";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    d[1].miss.module = module_;
    d[1].miss.entryFunctionName = "__miss__radiance";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    d[2].miss.module = module_;
    d[2].miss.entryFunctionName = "__miss__shadow";
    d[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    d[3].hitgroup.moduleCH = module_;
    d[3].hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    d[3].hitgroup.moduleAH = module_;
    d[3].hitgroup.entryFunctionNameAH = "__anyhit__alpha";
    OptixProgramGroup groups[4];
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_, d, 4, &pgo, log, &log_size, groups));
    raygen_pg_ = groups[0];
    miss_pg_ = groups[1];
    miss_shadow_pg_ = groups[2];
    hit_pg_ = groups[3];

    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;  // no recursive tracing: the integrator loop lives in raygen
    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(context_, &pco_, &plo, groups, 4, log, &log_size, &pipeline_));

    OptixStackSizes ss = {};
    for (OptixProgramGroup g : groups) OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &ss, pipeline_));
    unsigned dc_trav = 0, dc_state = 0, cont = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&ss, plo.maxTraceDepth, 0, 0, &dc_trav, &dc_state, &cont));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, dc_trav, dc_state, cont, /*maxTraversableGraphDepth=*/2));

    // Shader binding table (no per-record data: everything is reached through LaunchParams).
    SbtRecord<EmptyData> rg, ms[2], hg;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg_, &rg));
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_, &ms[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_shadow_pg_, &ms[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(hit_pg_, &hg));
    sbt_raygen_.upload(&rg, sizeof(rg));
    sbt_miss_.upload(ms, sizeof(ms));
    sbt_hit_.upload(&hg, sizeof(hg));
    sbt_ = {};
    sbt_.raygenRecord = sbt_raygen_.dptr();
    sbt_.missRecordBase = sbt_miss_.dptr();
    sbt_.missRecordStrideInBytes = sizeof(SbtRecord<EmptyData>);
    sbt_.missRecordCount = 2;
    sbt_.hitgroupRecordBase = sbt_hit_.dptr();
    sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord<EmptyData>);
    sbt_.hitgroupRecordCount = 1;
  }

  void upload_cmf(const float* cmf, int n_bands) {
    std::vector<float> host(cmf, cmf + 3 * n_bands);
    if (host != cmf_host_) {
      cmf_.upload(host);
      cmf_host_ = std::move(host);
    }
  }

  void upload_scene(const Scene& scene) {
    SceneView v = scene.view();
    if (uploaded_positions_ != scene.positions().size() || uploaded_meshes_ != scene.meshes().size()) {
      positions_.upload(scene.positions());
      normals_.upload(scene.normals());
      uvs_.upload(scene.uvs());
      indices_.upload(scene.indices());
      meshes_.upload(scene.meshes());
      uploaded_positions_ = scene.positions().size();
      uploaded_meshes_ = scene.meshes().size();
    }
    instances_.upload(scene.instances());
    materials_.upload(scene.materials());
    spectra_.upload(scene.spectra());
    d65n_.upload(scene.d65n_bands());
    lights_.upload(scene.lights());
    alias_.upload(scene.light_alias());
    if (uploaded_env_version_ != scene.env_version()) {
      env_data_.upload(scene.env_data());
      env_func_.upload(scene.env_dist().func);
      env_cond_.upload(scene.env_dist().cond_cdf);
      env_row_.upload(scene.env_dist().row_int);
      env_marg_.upload(scene.env_dist().marg_cdf);
      uploaded_env_version_ = scene.env_version();
    }
    if (scene.uplift && !lut_data_.ptr) {
      lut_scale_.upload(scene.uplift->scale);
      lut_data_.upload(scene.uplift->data);
    }
    upload_textures(scene);

    v.positions = positions_.as<Vec3>();
    v.normals = normals_.as<Vec3>();
    v.uvs = uvs_.as<Vec2>();
    v.indices = indices_.as<uint32_t>();
    v.meshes = meshes_.as<MeshRecord>();
    v.instances = instances_.as<InstanceRecord>();
    v.materials = materials_.as<MaterialRecord>();
    v.textures = tex_views_.as<TextureView>();
    v.spectra = spectra_.as<float>();
    v.d65n_bands = d65n_.as<float>();
    v.lut.scale = lut_scale_.as<float>();
    v.lut.data = lut_data_.as<float>();
    v.lights = lights_.as<LightRecord>();
    v.light_alias = alias_.as<AliasEntry>();
    v.env.data = env_data_.as<float>();
    v.env.dist.func = env_func_.as<float>();
    v.env.dist.cond_cdf = env_cond_.as<float>();
    v.env.dist.row_int = env_row_.as<float>();
    v.env.dist.marg_cdf = env_marg_.as<float>();
    scene_view_ = v;
  }

  void upload_textures(const Scene& scene) {
    // Existing texture objects are kept; only textures added since the last upload are created.
    if (textures_.size() == scene.textures().size() && tex_views_.ptr) return;
    std::vector<TextureView>& views = tex_views_host_;
    for (size_t ti = textures_.size(); ti < scene.textures().size(); ++ti) {
      const TextureData& td = scene.textures()[ti];
      cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
      cudaArray_t arr;
      CUDA_CHECK(cudaMallocArray(&arr, &ch, size_t(td.width), size_t(td.height)));
      CUDA_CHECK(cudaMemcpy2DToArray(arr, 0, 0, td.rgba.data(), size_t(td.width) * 4, size_t(td.width) * 4,
                                     size_t(td.height), cudaMemcpyHostToDevice));
      cudaResourceDesc res = {};
      res.resType = cudaResourceTypeArray;
      res.res.array.array = arr;
      cudaTextureDesc tex = {};
      cudaTextureAddressMode am = td.wrap == kWrapClamp    ? cudaAddressModeClamp
                                  : td.wrap == kWrapMirror ? cudaAddressModeMirror
                                                           : cudaAddressModeWrap;
      tex.addressMode[0] = tex.addressMode[1] = am;
      tex.filterMode = cudaFilterModeLinear;
      tex.readMode = cudaReadModeNormalizedFloat;
      tex.normalizedCoords = 1;
      tex.sRGB = td.srgb ? 1 : 0;  // hardware sRGB -> linear for RGB (alpha stays linear)
      cudaTextureObject_t obj;
      CUDA_CHECK(cudaCreateTextureObject(&obj, &res, &tex, nullptr));
      arrays_.push_back(arr);
      textures_.push_back(obj);
      TextureView view;
      view.width = td.width;
      view.height = td.height;
      view.srgb = td.srgb;
      view.wrap = td.wrap;
      view.data = nullptr;
      view.gpu_texture = static_cast<unsigned long long>(obj);
      views.push_back(view);
    }
    tex_views_.upload(views);
  }

  OptixTraversableHandle build(const OptixBuildInput& input, DeviceBuffer& out, bool compact) {
    OptixAccelBuildOptions opts = {};
    opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | (compact ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0);
    opts.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &opts, &input, 1, &sizes));
    DeviceBuffer temp, full, compacted_size;
    temp.alloc(sizes.tempSizeInBytes);
    full.alloc(sizes.outputSizeInBytes);
    compacted_size.alloc(sizeof(uint64_t));
    OptixAccelEmitDesc emit = {};
    emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit.result = compacted_size.dptr();
    OptixTraversableHandle handle = 0;
    OPTIX_CHECK(optixAccelBuild(context_, stream_, &opts, &input, 1, temp.dptr(), temp.size, full.dptr(), full.size,
                                &handle, compact ? &emit : nullptr, compact ? 1 : 0));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (compact) {
      uint64_t csize = 0;
      CUDA_CHECK(cudaMemcpy(&csize, compacted_size.ptr, sizeof(csize), cudaMemcpyDeviceToHost));
      if (csize < full.size) {
        out.alloc(csize);
        OPTIX_CHECK(optixAccelCompact(context_, stream_, handle, out.dptr(), csize, &handle));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return handle;
      }
    }
    out = std::move(full);
    return handle;
  }

  void build_accels(const Scene& scene) {
    const auto& meshes = scene.meshes();
    const auto& insts = scene.instances();
    // GAS only for meshes that have none yet (meshes are never modified once added).
    size_t first_new = gas_handles_.size();
    gas_.resize(meshes.size());
    gas_handles_.resize(meshes.size(), 0);
    std::vector<OptixTraversableHandle>& gas_handles = gas_handles_;
    const unsigned geom_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    for (size_t mi = first_new; mi < meshes.size(); ++mi) {
      const MeshRecord& m = meshes[mi];
      if (m.tri_count == 0) continue;
      CUdeviceptr verts = positions_.dptr() + CUdeviceptr(m.vtx_offset) * sizeof(Vec3);
      OptixBuildInput in = {};
      in.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      in.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      in.triangleArray.vertexStrideInBytes = sizeof(Vec3);
      in.triangleArray.numVertices = m.vtx_count;
      in.triangleArray.vertexBuffers = &verts;
      in.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      in.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
      in.triangleArray.numIndexTriplets = m.tri_count;
      in.triangleArray.indexBuffer = indices_.dptr() + CUdeviceptr(m.tri_offset) * 3 * sizeof(uint32_t);
      in.triangleArray.flags = &geom_flags;
      in.triangleArray.numSbtRecords = 1;
      gas_handles[mi] = build(in, gas_[mi], true);
    }
    std::vector<OptixInstance> oi(insts.size());
    for (size_t i = 0; i < insts.size(); ++i) {
      const InstanceRecord& r = insts[i];
      OptixInstance& o = oi[i];
      std::memset(&o, 0, sizeof(o));
      std::memcpy(o.transform, r.to_world.m, sizeof(o.transform));  // 3x4 row-major, same layout
      o.instanceId = unsigned(i);
      o.sbtOffset = 0;
      o.visibilityMask = (gas_handles[r.mesh] && !r.hidden) ? 255 : 0;
      bool masked = r.material >= 0 && scene.materials()[r.material].alpha_mode == kAlphaMask;
      o.flags = masked ? OPTIX_INSTANCE_FLAG_NONE : OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
      o.traversableHandle = gas_handles[r.mesh];
    }
    instances_buf_.upload(oi);
    OptixBuildInput in = {};
    in.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    in.instanceArray.instances = instances_buf_.dptr();
    in.instanceArray.numInstances = unsigned(oi.size());
    ias_handle_ = oi.empty() ? 0 : build(in, ias_, false);
  }

  OptixDeviceContext context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  OptixModule module_ = nullptr;
  OptixPipelineCompileOptions pco_ = {};
  OptixProgramGroup raygen_pg_ = nullptr, miss_pg_ = nullptr, miss_shadow_pg_ = nullptr, hit_pg_ = nullptr;
  OptixPipeline pipeline_ = nullptr;
  OptixShaderBindingTable sbt_ = {};
  DeviceBuffer sbt_raygen_, sbt_miss_, sbt_hit_, params_buf_;

  DeviceBuffer positions_, normals_, uvs_, indices_, meshes_, instances_, materials_, spectra_, d65n_, lights_, alias_;
  DeviceBuffer env_data_, env_func_, env_cond_, env_row_, env_marg_, lut_scale_, lut_data_, tex_views_;
  std::vector<cudaArray_t> arrays_;
  std::vector<cudaTextureObject_t> textures_;
  std::vector<DeviceBuffer> gas_;
  std::vector<OptixTraversableHandle> gas_handles_;
  std::vector<TextureView> tex_views_host_;
  size_t uploaded_positions_ = size_t(-1), uploaded_meshes_ = size_t(-1);
  uint64_t uploaded_env_version_ = uint64_t(-1);
  DeviceBuffer cmf_, preview_buf_, lum_buf_;
  std::vector<float> cmf_host_;
  DeviceBuffer ias_, instances_buf_;
  OptixTraversableHandle ias_handle_ = 0;
  SceneView scene_view_;
  DeviceBuffer film_radiance_, film_depth_, film_seg_;
};

}  // namespace

bool cuda_backend_available() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  return optixInit() == OPTIX_SUCCESS;
}

std::unique_ptr<Backend> make_cuda_backend() { return std::make_unique<CudaBackend>(); }

}  // namespace spectral
