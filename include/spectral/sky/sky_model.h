// Spectral sky models. A sky model is tabulated once per scene into the environment
// radiance table (lat-long x bands) plus an explicit sun-disk light, so the kernels
// never call the model directly (identical behaviour on CPU and GPU).
//
// World convention: +Y up. Sun elevation/azimuth in JSON: azimuth measured from -Z
// ("north") towards +X ("east"): dir = (sin(az) cos(el), sin(el), -cos(az) cos(el)).
#pragma once
#include <memory>
#include <string>

#include "spectral/kernel/math.h"
#include "spectral/spectrum/dense_spectrum.h"

namespace spectral {

class Scene;

struct SkyParams {
  Vec3 sun_direction = Vec3(0.f, 1.f, 0.f);  // unit vector towards the sun
  double visibility_km = 59.4;
  double ground_albedo = 0.33;
  double altitude_m = 0.0;
};

class SkyModel {
 public:
  virtual ~SkyModel() = default;
  virtual std::string name() const = 0;
  virtual void configure(const SkyParams& p) = 0;
  // Diffuse sky radiance (sun disk excluded), W / (m^2 sr nm). dir: unit, world space.
  virtual double sky_radiance(Vec3 dir, double lambda_nm) const = 0;
  // Radiance of the sun disk as seen from the ground (after atmospheric transmittance).
  virtual double sun_radiance(double lambda_nm) const = 0;
  virtual double sun_half_angle() const = 0;  // radians
  // Radiance below the horizon (ground not modelled by geometry). Default: 0.
  virtual bool below_horizon_black() const { return true; }
};

Vec3 sun_direction_from_angles(double elevation_deg, double azimuth_deg);

// Constant-radiance sky (spectrum) + sun disk with given radiance spectrum.
std::unique_ptr<SkyModel> make_simple_sky(const DenseSpectrum& sky_radiance, const DenseSpectrum& sun_radiance,
                                          double sun_half_angle_rad = 0.004654793);
// Prague Sky Model (Wilkie et al. 2021). Dataset: e.g. PragueSkyModelDatasetSWIR.dat (280-2480 nm).
bool prague_sky_available();
std::unique_ptr<SkyModel> make_prague_sky(const std::string& dataset_path, double visibility_km = 0.0);
// Dynamically loaded sky plugin implementing the C ABI in spectral/sky/sky_plugin.h.
std::unique_ptr<SkyModel> load_sky_plugin(const std::string& library_path, const std::string& options_json);

struct SkyTabulation {
  int width = 512, height = 256;
  float scale = 1.f;
  float rotation = 0.f;
  int supersample = 2;          // per-texel sub-samples per axis (averaged)
  std::string cache_path;       // optional binary cache for the tabulated table
};
// Evaluates the model on the scene's band grid; sets the environment table and sun light.
void add_sky_to_scene(const SkyModel& sky, const SkyParams& params, const SkyTabulation& tab, Scene& scene);

}  // namespace spectral
