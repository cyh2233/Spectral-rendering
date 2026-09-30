// Flat, POD scene representation consumed by the integrator on CPU and GPU.
// All pointers refer to host memory (CPU backend) or device memory (CUDA backend).
#pragma once
#include <cstdint>

#include "spectral/kernel/math.h"
#include "spectral/kernel/sampling.h"
#include "spectral/spectrum/spectrum.h"

namespace spectral {

constexpr uint32_t kInvalidId = 0xFFFFFFFFu;
constexpr uint32_t kSkySegId = 0xFFFFFFFFu;

enum MeshFlags : uint32_t { kMeshHasNormals = 1u, kMeshHasUVs = 2u };

struct MeshRecord {
  uint32_t vtx_offset = 0;  // into positions/normals/uvs
  uint32_t vtx_count = 0;
  uint32_t tri_offset = 0;  // triangle index into indices (3 uint32 per triangle)
  uint32_t tri_count = 0;
  int32_t material = -1;
  uint32_t flags = 0;
};

struct InstanceRecord {
  Affine to_world;
  Affine to_local;
  uint32_t mesh = 0;
  int32_t material = -1;      // resolved material (per-instance overrides applied on the host)
  uint32_t seg_id = 0;        // semantic segmentation id
  int32_t light_offset = -1;  // index of the first triangle light of this instance, or -1
  uint32_t hidden = 0;        // 1: excluded from traversal and light sampling (removed/pooled)
};

enum TextureWrap : int32_t { kWrapRepeat = 0, kWrapClamp = 1, kWrapMirror = 2 };

struct TextureView {
  int32_t width = 0, height = 0;
  int32_t srgb = 0;  // 1: RGB channels are sRGB-encoded (alpha always linear)
  int32_t wrap = kWrapRepeat;
  const uint8_t* data = nullptr;  // RGBA8, row-major, top row first
  unsigned long long gpu_texture = 0;  // cudaTextureObject_t on the CUDA backend
};

// Jakob-Hanika RGB -> spectrum coefficient table (rgb2spec layout).
struct UpliftLutView {
  int32_t res = 0;
  const float* scale = nullptr;  // res entries
  const float* data = nullptr;   // 3 * res^3 * 3 coefficients (wavelength domain, nm)
};

enum MaterialType : int32_t { kMatPbr = 0, kMatDielectric = 1, kMatThinDielectric = 2 };
enum AlphaMode : int32_t { kAlphaOpaque = 0, kAlphaMask = 1, kAlphaBlend = 2 };

struct MaterialRecord {
  int32_t type = kMatPbr;
  float base_color[4] = {1, 1, 1, 1};  // linear RGBA factor (glTF)
  int32_t base_color_tex = -1;
  float metallic = 0.f, roughness = 0.5f;
  float specular = 1.f;  // KHR_materials_specular factor (0 = pure Lambert diffuse)
  int32_t metal_rough_tex = -1;  // glTF: G = roughness, B = metallic
  int32_t normal_tex = -1;
  float normal_scale = 1.f;
  float emissive[3] = {0, 0, 0};  // linear RGB factor
  float emissive_strength = 1.f;
  int32_t emissive_tex = -1;
  // Spectral data (offsets into SceneView::spectra, each n_bands floats; -1 = none)
  int32_t reflectance_spec = -1;    // replaces uplifted base color
  int32_t emission_spec = -1;       // replaces uplifted emissive (radiance, W/m^2/sr/nm)
  float emission_scale = 1.f;
  int32_t transmittance_spec = -1;  // thin dielectric: internal transmittance per pass
  int32_t absorption_spec = -1;     // dielectric: absorption coefficient [1/m]
  int32_t ior_spec = -1;            // per-band IOR for Fresnel; -1 = constant `ior`
  float ior = 1.5f;                 // reference IOR (refraction direction, d-line)
  int32_t alpha_mode = kAlphaOpaque;
  float alpha_cutoff = 0.5f;
  int32_t double_sided = 0;
};

enum LightType : uint32_t {
  kLightPoint = 0,
  kLightSpot = 1,
  kLightDirectional = 2,
  kLightSun = 3,       // finite-angle disk at infinity
  kLightTriangle = 4,  // emissive triangle of an instance
  kLightEnv = 5        // environment table (sky or HDR map)
};

struct LightRecord {
  uint32_t type = kLightPoint;
  int32_t spectrum = -1;  // intensity (point/spot, W/sr/nm), irradiance (directional, W/m^2/nm),
                          // or radiance (sun disk, W/m^2/sr/nm); unused for triangles and env
  float scale = 1.f;
  Vec3 position;
  Vec3 direction;  // spot axis / direction of light travel for directional; direction *to* the sun for sun
  float cos_inner = 1.f, cos_outer = 0.f;  // spot cone
  float cos_max = 1.f;                     // sun disk half-angle cosine
  uint32_t instance = 0, prim = 0;         // triangle lights
  float area = 0.f;                        // world-space triangle area
};

// Environment radiance table in lat-long parameterization:
// u = phi / 2pi (phi measured from +X towards +Z), v = theta / pi (theta from +Y).
struct EnvView {
  int32_t present = 0;
  int32_t width = 0, height = 0;
  const float* data = nullptr;  // height * width * n_bands radiance (W/m^2/sr/nm)
  Distribution2DView dist;
  float scale = 1.f;
  float rotation = 0.f;  // rotation about +Y in radians applied to lookups
};

enum CameraType : int32_t { kCameraPinhole = 0 };

struct CameraParams {
  int32_t type = kCameraPinhole;
  Affine cam_to_world = Affine::identity();  // camera looks down -Z, +Y up, +X right
  float tan_half_fov_x = 1.f, tan_half_fov_y = 1.f;
  int32_t width = 64, height = 64;
};

enum DepthMode : int32_t { kDepthDistance = 0, kDepthZ = 1 };

struct RenderParams {
  int32_t max_depth = 8;  // maximum number of non-glass scattering events
  int32_t max_glass_events = 16;  // glass (dielectric) events do not consume max_depth
  int32_t rr_depth = 3;   // start Russian roulette after this many bounces
  uint64_t seed = 0;
  int32_t transparent_shadows = 1;  // shadow rays pass through glass with Fresnel/absorption attenuation
  float uplift_hold_lambda = 780.f; // RGB-uplifted spectra are held constant beyond this wavelength
  int32_t depth_mode = kDepthDistance;
  float clamp_contribution = 0.f;   // >0: clamp per-sample max band radiance (biased, off by default)
  int32_t has_glass = 0;            // scene contains dielectric materials
};

struct SceneView {
  BandGrid grid;
  const Vec3* positions = nullptr;
  const Vec3* normals = nullptr;
  const Vec2* uvs = nullptr;
  const uint32_t* indices = nullptr;
  const MeshRecord* meshes = nullptr;
  const InstanceRecord* instances = nullptr;
  uint32_t n_instances = 0;
  const MaterialRecord* materials = nullptr;
  const TextureView* textures = nullptr;
  const float* spectra = nullptr;
  const float* d65n_bands = nullptr;  // normalized D65 on the band grid (RGB emission uplift)
  UpliftLutView lut;
  const LightRecord* lights = nullptr;
  uint32_t n_lights = 0;
  const AliasEntry* light_alias = nullptr;
  int32_t env_light = -1;  // index of the env light record or -1
  int32_t sun_light = -1;  // index of the sun light record or -1
  EnvView env;
  CameraParams camera;
  RenderParams params;
};

struct Ray {
  Vec3 o, d;
};

struct Hit {
  float t = kInf;
  uint32_t instance = kInvalidId, prim = kInvalidId;
  float b1 = 0.f, b2 = 0.f;
};

}  // namespace spectral
