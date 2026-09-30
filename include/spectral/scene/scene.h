// Host scene container: owns all geometry/material/light data in the flat layout
// used by the kernels and produces a SceneView for the CPU backend (or for upload).
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "spectral/kernel/scene_view.h"
#include "spectral/spectrum/dense_spectrum.h"

namespace spectral {

struct MeshData {
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;  // optional (empty or same size as positions)
  std::vector<Vec2> uvs;      // optional
  std::vector<uint32_t> indices;
  int32_t material = -1;
  std::string name;
};

struct TextureData {
  int width = 0, height = 0;
  bool srgb = false;
  int wrap = kWrapRepeat;
  std::vector<uint8_t> rgba;  // RGBA8
  std::string name;
};

struct UpliftLut {
  int res = 0;
  std::vector<float> scale, data;
  static UpliftLut load(const std::string& path);  // rgb2spec ".coeff" format
  UpliftLutView view() const;
};

struct RenderSettings {
  int spp = 64;
  int spp_per_pass = 4;
  int threads = 0;  // 0 = all hardware threads
  std::string backend = "auto";
  float light_reference_distance = 10.f;
};

struct OutputSettings {
  std::string exr, npz, preview_png;
  bool exr_half = false;
  std::string metadata_json = "{}";
};

class Scene {
 public:
  Scene();

  // ---- configuration
  BandGrid grid;
  CameraParams camera;
  RenderParams params;
  RenderSettings render;
  OutputSettings output;
  std::shared_ptr<const UpliftLut> uplift;

  void set_band_grid(float lambda_min, float lambda_max, float step);

  // ---- spectra (values on the band grid, n floats each); returns offset into the pool
  int32_t add_band_spectrum(const std::vector<float>& bands);
  int32_t add_spectrum(const DenseSpectrum& s) { return add_band_spectrum(s.to_bands(grid)); }
  std::vector<float> band_spectrum(int32_t offset) const;

  // ---- geometry / materials
  int32_t add_texture(TextureData tex);
  int32_t add_material(const MaterialRecord& m, const std::string& name);
  uint32_t add_mesh(const MeshData& mesh);
  uint32_t add_instance(uint32_t mesh, const Affine& to_world, int32_t material_override = -1, uint32_t seg_id = 0);
  // Dynamic updates (live sessions). Call finalize() afterwards.
  void set_instance_transform(uint32_t instance, const Affine& to_world);
  void set_instance_hidden(uint32_t instance, bool hidden);
  void set_instance_seg_id(uint32_t instance, uint32_t seg_id);

  // ---- lights
  int32_t add_light(const LightRecord& l);
  // Lights replaced wholesale every frame (e.g. vehicle headlights). Spectra must already be in the pool.
  void set_dynamic_lights(std::vector<LightRecord> lights);
  // Overwrites the band values of an existing pooled spectrum (same size).
  void overwrite_band_spectrum(int32_t offset, const std::vector<float>& bands);
  // Environment radiance table: height*width*n_bands values (lat-long, see EnvView).
  void set_environment(int width, int height, std::vector<float> radiance, float scale = 1.f, float rotation = 0.f);
  // Sun disk: direction towards the sun, half-angle in radians, disk radiance spectrum.
  // Replaces an existing sun (its spectrum slot is reused).
  void set_sun(Vec3 direction_to_sun, float half_angle_rad, const std::vector<float>& radiance_bands,
               float scale = 1.f);

  // Builds triangle lights for emissive instances and the light-selection table.
  void finalize();
  bool finalized() const { return finalized_; }

  SceneView view() const;

  // ---- accessors
  const std::vector<MeshRecord>& meshes() const { return meshes_; }
  const std::vector<InstanceRecord>& instances() const { return instances_; }
  std::vector<InstanceRecord>& instances_mut() { return instances_; }
  const std::vector<MaterialRecord>& materials() const { return materials_; }
  std::vector<MaterialRecord>& materials_mut() { return materials_; }
  const std::vector<std::string>& material_names() const { return material_names_; }
  const std::vector<LightRecord>& lights() const { return lights_; }
  const std::vector<TextureData>& textures() const { return textures_; }
  const std::vector<Vec3>& positions() const { return positions_; }
  const std::vector<Vec3>& normals() const { return normals_; }
  const std::vector<Vec2>& uvs() const { return uvs_; }
  const std::vector<uint32_t>& indices() const { return indices_; }
  const std::vector<float>& spectra() const { return spectra_; }
  const std::vector<float>& env_data() const { return env_data_; }
  const std::vector<AliasEntry>& light_alias() const { return light_alias_; }
  const std::vector<float>& d65n_bands() const { return d65n_bands_; }
  struct EnvDistData {
    std::vector<float> func, cond_cdf, row_int, marg_cdf;
    float marg_int = 0.f;
  };
  const EnvDistData& env_dist() const { return env_dist_; }
  const EnvView& env_params() const { return env_; }
  int32_t env_light_index() const { return env_light_; }
  uint64_t env_version() const { return env_version_; }  // bumped by set_environment
  int32_t sun_light_index() const { return sun_light_; }
  bool has_sun() const {
    for (const auto& l : user_lights_)
      if (l.type == kLightSun) return true;
    return false;
  }

 private:
  std::vector<Vec3> positions_, normals_;
  std::vector<Vec2> uvs_;
  std::vector<uint32_t> indices_;
  std::vector<MeshRecord> meshes_;
  std::vector<InstanceRecord> instances_;
  std::vector<MaterialRecord> materials_;
  std::vector<std::string> material_names_;
  std::vector<TextureData> textures_;
  std::vector<TextureView> texture_views_;
  std::vector<float> spectra_;
  std::vector<LightRecord> user_lights_;     // added via add_light / set_sun
  std::vector<LightRecord> dynamic_lights_;  // replaced by set_dynamic_lights
  std::vector<LightRecord> lights_;          // finalized: user + dynamic + triangle + env
  std::vector<AliasEntry> light_alias_;
  std::vector<float> env_data_;
  EnvDistData env_dist_;
  EnvView env_;
  std::vector<float> d65n_bands_;
  int32_t env_light_ = -1, sun_light_ = -1;
  uint64_t env_version_ = 0;
  bool finalized_ = false;
};

// Vose alias table from non-negative weights (all-zero -> uniform).
std::vector<AliasEntry> build_alias_table(const std::vector<double>& weights);

}  // namespace spectral
