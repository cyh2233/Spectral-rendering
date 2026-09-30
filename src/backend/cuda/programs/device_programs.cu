// OptiX device programs. All shading runs in the ray-generation program through the shared
// kernel headers (identical code to the CPU backend); hit/miss programs only report hits.
#include <optix.h>

#include "../launch_params.h"
#include "spectral/kernel/integrator.h"

using namespace spectral;
using spectral::cuda::LaunchParams;

// Launch parameters as raw constant storage: kernel structs carry default member initializers,
// which CUDA does not allow for __constant__ objects. The byte size equals sizeof(LaunchParams).
// OptiX binds it by name (pipelineLaunchParamsVariableName), so it must be a .visible PTX symbol: the PTX is
// compiled with -rdc=true (cmake/CudaBackend.cmake); without it nvcc emits a module-local .const.
extern "C" {
__constant__ __align__(16) unsigned char params[sizeof(LaunchParams)];
}

static __forceinline__ __device__ const LaunchParams& P() {
  return *reinterpret_cast<const LaunchParams*>(params);
}

struct OptixIntersector {
  __device__ bool closest(const Ray& r, float tmax, Hit* hit) const {
    unsigned p0 = __float_as_uint(-1.f), p1 = kInvalidId, p2 = kInvalidId, p3 = 0, p4 = 0;
    optixTrace(P().handle, make_float3(r.o.x, r.o.y, r.o.z), make_float3(r.d.x, r.d.y, r.d.z), 0.f, tmax, 0.f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, spectral::cuda::kRayTypeRadiance, p0, p1, p2, p3,
               p4);
    if (p1 == kInvalidId) return false;
    hit->t = __uint_as_float(p0);
    hit->instance = p1;
    hit->prim = p2;
    hit->b1 = __uint_as_float(p3);
    hit->b2 = __uint_as_float(p4);
    return true;
  }
  __device__ bool occluded(const Ray& r, float tmax) const {
    unsigned p0 = 1u, p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    optixTrace(P().handle, make_float3(r.o.x, r.o.y, r.o.z), make_float3(r.d.x, r.d.y, r.d.z), 0.f, tmax, 0.f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1,
               spectral::cuda::kRayTypeShadow, p0, p1, p2, p3, p4);
    return p0 != 0u;
  }
};

extern "C" __global__ void __raygen__render() {
  const LaunchParams& lp = P();
  const uint3 idx = optixGetLaunchIndex();
  const int x = int(idx.x), y = int(idx.y);
  if (x >= lp.film.width || y >= lp.film.height) return;
  OptixIntersector isect;
  FilmView film = lp.film;
  Spectrum L;
  for (int k = 0; k < lp.sample_count; ++k) {
    const int s = lp.first_sample + k;
    PathAov aov;
    render_sample(lp.scene, isect, x, y, uint32_t(s), L, s == 0 ? &aov : nullptr);
    film.accumulate(x, y, L);  // one thread per pixel: no atomics needed
    if (s == 0) {
      film.depth[size_t(y) * film.width + x] = aov.depth;
      film.seg_id[size_t(y) * film.width + x] = aov.seg_id;
    }
  }
}

extern "C" __global__ void __closesthit__radiance() {
  const float2 b = optixGetTriangleBarycentrics();
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetInstanceIndex());
  optixSetPayload_2(optixGetPrimitiveIndex());
  optixSetPayload_3(__float_as_uint(b.x));
  optixSetPayload_4(__float_as_uint(b.y));
}

// Alpha-mask test. Attached to all instances; opaque instances disable any-hit via instance flags.
extern "C" __global__ void __anyhit__alpha() {
  const float2 b = optixGetTriangleBarycentrics();
  if (!alpha_test(P().scene, optixGetInstanceIndex(), optixGetPrimitiveIndex(), b.x, b.y)) optixIgnoreIntersection();
}

extern "C" __global__ void __miss__radiance() { optixSetPayload_1(kInvalidId); }

extern "C" __global__ void __miss__shadow() { optixSetPayload_0(0u); }
