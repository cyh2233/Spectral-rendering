#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <vector>

#include "spectral/scene/scene.h"
#include "spectral/sky/sky_model.h"
#include "spectral/sky/sky_plugin.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(SPECTRAL_HAS_PRAGUE)
#include "PragueSkyModel.h"
#endif

namespace spectral {

Vec3 sun_direction_from_angles(double el_deg, double az_deg) {
  double el = el_deg * M_PI / 180.0, az = az_deg * M_PI / 180.0;
  return normalize(Vec3(float(std::sin(az) * std::cos(el)), float(std::sin(el)), float(-std::cos(az) * std::cos(el))));
}

// ------------------------------------------------------------------ simple sky
namespace {
class SimpleSky final : public SkyModel {
 public:
  SimpleSky(DenseSpectrum sky, DenseSpectrum sun, double half) : sky_(std::move(sky)), sun_(std::move(sun)), half_(half) {}
  std::string name() const override { return "simple"; }
  void configure(const SkyParams&) override {}
  double sky_radiance(Vec3, double l) const override { return sky_.eval(float(l)); }
  double sun_radiance(double l) const override { return sun_.eval(float(l)); }
  double sun_half_angle() const override { return half_; }
  bool below_horizon_black() const override { return false; }

 private:
  DenseSpectrum sky_, sun_;
  double half_;
};
}  // namespace

std::unique_ptr<SkyModel> make_simple_sky(const DenseSpectrum& sky, const DenseSpectrum& sun, double half) {
  return std::make_unique<SimpleSky>(sky, sun, half);
}

// ------------------------------------------------------------------ Prague
#if defined(SPECTRAL_HAS_PRAGUE)
namespace {
class PragueSky final : public SkyModel {
 public:
  PragueSky(const std::string& path, double vis) { model_.initialize(path, vis); }
  std::string name() const override { return "prague"; }
  void configure(const SkyParams& p) override {
    p_ = p;
    Vec3 s = to_prague(p.sun_direction);
    elevation_ = std::asin(std::max(-1.0, std::min(1.0, double(s.z))));
    azimuth_ = std::atan2(double(s.y), double(s.x));
    if (azimuth_ < 0) azimuth_ += 2 * M_PI;
    auto sun_params = model_.computeParameters(PragueSkyModel::Vector3(0, 0, p.altitude_m),
                                               PragueSkyModel::Vector3(s.x, s.y, s.z), elevation_, azimuth_,
                                               p.visibility_km, p.ground_albedo);
    sun_params_ = sun_params;
  }
  double sky_radiance(Vec3 dir, double l) const override {
    Vec3 d = to_prague(dir);
    if (d.z < 0.f) return 0.0;
    auto params = model_.computeParameters(PragueSkyModel::Vector3(0, 0, p_.altitude_m),
                                           PragueSkyModel::Vector3(d.x, d.y, std::max(d.z, 1e-4f)), elevation_,
                                           azimuth_, p_.visibility_km, p_.ground_albedo);
    return std::max(0.0, model_.skyRadiance(params, l));
  }
  double sun_radiance(double l) const override {
    // Evaluate at the sun centre (gamma = 0) including atmospheric transmittance.
    return std::max(0.0, model_.sunRadiance(sun_params_, l));
  }
  double sun_half_angle() const override { return 0.004654793; }

 private:
  // Renderer world (+Y up, -Z north) -> Prague frame (+Z up), right-handed: (x, -z, y).
  static Vec3 to_prague(Vec3 w) { return Vec3(w.x, -w.z, w.y); }
  PragueSkyModel model_;
  PragueSkyModel::Parameters sun_params_{};
  SkyParams p_;
  double elevation_ = 0, azimuth_ = 0;
};
}  // namespace
bool prague_sky_available() { return true; }
std::unique_ptr<SkyModel> make_prague_sky(const std::string& path, double vis) {
  return std::make_unique<PragueSky>(path, vis);
}
#else
bool prague_sky_available() { return false; }
std::unique_ptr<SkyModel> make_prague_sky(const std::string&, double) {
  throw std::runtime_error("Prague sky model not compiled in (SPECTRAL_ENABLE_PRAGUE_SKY=OFF)");
}
#endif

// ------------------------------------------------------------------ plugin
namespace {
class PluginSky final : public SkyModel {
 public:
  PluginSky(const std::string& lib, const std::string& opts) {
#if defined(_WIN32)
    handle_ = LoadLibraryA(lib.c_str());
#else
    handle_ = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (!handle_) throw std::runtime_error("Cannot load sky plugin '" + lib + "'" + last_error());
    load(abi_, "spectral_sky_abi_version");
    load(create_, "spectral_sky_create");
    load(configure_, "spectral_sky_configure");
    load(radiance_, "spectral_sky_radiance");
    load(sun_, "spectral_sky_sun_radiance");
    load(half_, "spectral_sky_sun_half_angle");
    load(destroy_, "spectral_sky_destroy");
    if (abi_() != SPECTRAL_SKY_ABI_VERSION)
      throw std::runtime_error("Sky plugin ABI mismatch: " + std::to_string(abi_()));
    sky_ = create_(opts.c_str());
    if (!sky_) throw std::runtime_error("Sky plugin failed to create an instance");
  }
  ~PluginSky() override {
    if (sky_) destroy_(sky_);
#if defined(_WIN32)
    if (handle_) FreeLibrary(static_cast<HMODULE>(handle_));
#else
    if (handle_) dlclose(handle_);
#endif
  }
  std::string name() const override { return "plugin"; }
  void configure(const SkyParams& p) override {
    if (configure_(sky_, p.sun_direction.x, p.sun_direction.y, p.sun_direction.z, p.visibility_km, p.ground_albedo,
                   p.altitude_m) != 0)
      throw std::runtime_error("Sky plugin configure() failed");
  }
  double sky_radiance(Vec3 d, double l) const override { return radiance_(sky_, d.x, d.y, d.z, l); }
  double sun_radiance(double l) const override { return sun_(sky_, l); }
  double sun_half_angle() const override { return half_(sky_); }

 private:
  static std::string last_error() {
#if defined(_WIN32)
    return "";
#else
    const char* e = dlerror();
    return e ? std::string(": ") + e : "";
#endif
  }
  template <class F>
  void load(F& f, const char* sym) {
#if defined(_WIN32)
    f = reinterpret_cast<F>(GetProcAddress(static_cast<HMODULE>(handle_), sym));
#else
    f = reinterpret_cast<F>(dlsym(handle_, sym));
#endif
    if (!f) throw std::runtime_error(std::string("Sky plugin missing symbol ") + sym);
  }
  void* handle_ = nullptr;
  spectral_sky* sky_ = nullptr;
  int (*abi_)() = nullptr;
  spectral_sky* (*create_)(const char*) = nullptr;
  int (*configure_)(spectral_sky*, double, double, double, double, double, double) = nullptr;
  double (*radiance_)(const spectral_sky*, double, double, double, double) = nullptr;
  double (*sun_)(const spectral_sky*, double) = nullptr;
  double (*half_)(const spectral_sky*) = nullptr;
  void (*destroy_)(spectral_sky*) = nullptr;
};
}  // namespace

std::unique_ptr<SkyModel> load_sky_plugin(const std::string& lib, const std::string& opts) {
  return std::make_unique<PluginSky>(lib, opts);
}

// ------------------------------------------------------------------ tabulation
void add_sky_to_scene(const SkyModel& sky, const SkyParams& params, const SkyTabulation& tab, Scene& scene) {
  const BandGrid g = scene.grid;
  const int W = tab.width, H = tab.height, ss = std::max(1, tab.supersample);
  std::vector<float> table;
  bool loaded = false;
  if (!tab.cache_path.empty()) {
    std::ifstream in(tab.cache_path, std::ios::binary);
    if (in) {
      int32_t hdr[3];
      in.read(reinterpret_cast<char*>(hdr), sizeof hdr);
      if (in && hdr[0] == W && hdr[1] == H && hdr[2] == g.n) {
        table.resize(size_t(W) * H * g.n);
        in.read(reinterpret_cast<char*>(table.data()), std::streamsize(table.size() * 4));
        loaded = bool(in);
      }
    }
  }
  if (!loaded) {
    table.assign(size_t(W) * H * g.n, 0.f);
    // Evaluate each band at sub-wavelengths across the band and average (5 samples).
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        float* px = table.data() + (size_t(y) * W + x) * g.n;
        for (int sy = 0; sy < ss; ++sy)
          for (int sx = 0; sx < ss; ++sx) {
            float u = (float(x) + (float(sx) + 0.5f) / ss) / float(W);
            float v = (float(y) + (float(sy) + 0.5f) / ss) / float(H);
            float theta = v * kPi, phi = u * 2.f * kPi + tab.rotation;
            Vec3 d = spherical_direction(std::sin(theta), std::cos(theta), phi);
            if (d.y < 0.f && sky.below_horizon_black()) continue;
            for (int b = 0; b < g.n; ++b) {
              double acc = 0;
              for (int k = 0; k < 3; ++k) acc += sky.sky_radiance(d, g.center(b) + (float(k) - 1.f) * g.step / 3.f);
              px[b] += float(acc / 3.0) / float(ss * ss);
            }
          }
      }
    }
    if (!tab.cache_path.empty()) {
      std::ofstream out(tab.cache_path, std::ios::binary);
      int32_t hdr[3] = {W, H, g.n};
      out.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
      out.write(reinterpret_cast<const char*>(table.data()), std::streamsize(table.size() * 4));
    }
  }
  scene.set_environment(W, H, std::move(table), tab.scale, tab.rotation);
  if (params.sun_direction.y > -0.05f) {
    std::vector<float> sun(g.n);
    for (int b = 0; b < g.n; ++b) sun[b] = float(sky.sun_radiance(g.center(b)));
    double s = 0;
    for (float v : sun) s += v;
    if (s > 0) scene.set_sun(params.sun_direction, float(sky.sun_half_angle()), sun, tab.scale);
  }
}

}  // namespace spectral
