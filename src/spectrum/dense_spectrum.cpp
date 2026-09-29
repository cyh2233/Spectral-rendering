#include "spectral/spectrum/dense_spectrum.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "spectral/spectrum/colorimetry.h"

namespace spectral {

namespace cie {
float lookup(const float* table, float lambda) {
  if (lambda < kLambdaMin || lambda > kLambdaMax) return 0.f;
  float x = lambda - kLambdaMin;
  int i = std::min(int(x), kSamples - 2);
  float t = x - float(i);
  return table[i] * (1.f - t) + table[i + 1] * t;
}
}  // namespace cie

DenseSpectrum DenseSpectrum::constant(float c) {
  DenseSpectrum s;
  s.values_.fill(c);
  return s;
}

DenseSpectrum DenseSpectrum::from_samples(const std::vector<float>& wl, const std::vector<float>& v,
                                          Extrapolation extrap) {
  if (wl.size() != v.size() || wl.empty()) throw std::runtime_error("DenseSpectrum: empty or mismatched samples");
  for (size_t i = 1; i < wl.size(); ++i)
    if (!(wl[i] > wl[i - 1])) throw std::runtime_error("DenseSpectrum: wavelengths must be strictly ascending");
  DenseSpectrum s;
  for (int i = 0; i < kSamples; ++i) {
    float l = lambda_at(i);
    float val;
    if (l < wl.front()) {
      val = extrap == Extrapolation::Clamp ? v.front() : 0.f;
    } else if (l > wl.back()) {
      val = extrap == Extrapolation::Clamp ? v.back() : 0.f;
    } else {
      size_t k = std::upper_bound(wl.begin(), wl.end(), l) - wl.begin();
      if (k >= wl.size()) {
        val = v.back();
      } else if (k == 0) {
        val = v.front();
      } else {
        float t = (l - wl[k - 1]) / (wl[k] - wl[k - 1]);
        val = v[k - 1] * (1.f - t) + v[k] * t;
      }
    }
    s.values_[i] = val;
  }
  return s;
}

DenseSpectrum DenseSpectrum::blackbody(float T) {
  const double h = 6.62607015e-34, c = 299792458.0, kb = 1.380649e-23;
  DenseSpectrum s;
  for (int i = 0; i < kSamples; ++i) {
    double l = double(lambda_at(i)) * 1e-9;
    double le = 2.0 * h * c * c / (std::pow(l, 5) * (std::exp(h * c / (l * kb * double(T))) - 1.0));
    s.values_[i] = float(le * 1e-9);  // per m -> per nm
  }
  return s;
}

DenseSpectrum DenseSpectrum::cie_illuminant(const std::string& name) {
  DenseSpectrum s;
  if (name == "D65") {
    for (int i = 0; i < kSamples; ++i) {
      float l = std::clamp(lambda_at(i), cie::kLambdaMin, cie::kLambdaMax);
      s.values_[i] = cie::lookup(cie::d65, l);
    }
    // Below 360 nm the D65 table is not stored; hold the 360 nm value.
    return s;
  }
  if (name == "A") {
    const double c2 = 1.435e7;  // nm K (CIE definition of illuminant A)
    const double T = 2848.0;
    for (int i = 0; i < kSamples; ++i) {
      double l = lambda_at(i);
      s.values_[i] = float(100.0 * std::pow(560.0 / l, 5) * (std::exp(c2 / (T * 560.0)) - 1.0) /
                           (std::exp(c2 / (T * l)) - 1.0));
    }
    return s;
  }
  if (name == "E") return constant(1.f);
  throw std::runtime_error("Unknown CIE illuminant '" + name + "' (built-in: D65, A, E; others via CSV)");
}

DenseSpectrum DenseSpectrum::cmf(int c, cie::Observer obs) {
  const float* tabs31[3] = {cie::x31, cie::y31, cie::z31};
  const float* tabs64[3] = {cie::x64, cie::y64, cie::z64};
  const float* t = (obs == cie::Observer::CIE1931_2deg ? tabs31 : tabs64)[c];
  DenseSpectrum s;
  for (int i = 0; i < kSamples; ++i) s.values_[i] = cie::lookup(t, lambda_at(i));
  return s;
}

float DenseSpectrum::eval(float l) const {
  float x = std::clamp(l, kLambdaMin, kLambdaMax) - kLambdaMin;
  int i = std::min(int(x), kSamples - 2);
  float t = x - float(i);
  return values_[i] * (1.f - t) + values_[i + 1] * t;
}

DenseSpectrum DenseSpectrum::operator*(const DenseSpectrum& o) const {
  DenseSpectrum r;
  for (int i = 0; i < kSamples; ++i) r.values_[i] = values_[i] * o.values_[i];
  return r;
}
DenseSpectrum DenseSpectrum::operator*(float s) const {
  DenseSpectrum r;
  for (int i = 0; i < kSamples; ++i) r.values_[i] = values_[i] * s;
  return r;
}
DenseSpectrum DenseSpectrum::operator+(const DenseSpectrum& o) const {
  DenseSpectrum r;
  for (int i = 0; i < kSamples; ++i) r.values_[i] = values_[i] + o.values_[i];
  return r;
}
float DenseSpectrum::max_value() const { return *std::max_element(values_.begin(), values_.end()); }

std::vector<float> DenseSpectrum::to_bands(const BandGrid& g) const {
  std::vector<float> out(g.n);
  for (int b = 0; b < g.n; ++b) {
    float lo = g.center(b) - 0.5f * g.step, hi = g.center(b) + 0.5f * g.step;
    // Average of the piecewise-linear function over [lo, hi] using fine sub-sampling.
    const int sub = std::max(4, int(std::ceil((hi - lo) * 4.f)));
    double acc = 0.0;
    for (int k = 0; k < sub; ++k) acc += eval(lo + (hi - lo) * (float(k) + 0.5f) / float(sub));
    out[b] = float(acc / sub);
  }
  return out;
}

float DenseSpectrum::integral(float lo, float hi) const {
  double acc = 0.0;
  for (int i = 0; i + 1 < kSamples; ++i) {
    float a = lambda_at(i), b = lambda_at(i + 1);
    if (b <= lo || a >= hi) continue;
    float a2 = std::max(a, lo), b2 = std::min(b, hi);
    acc += 0.5 * double(eval(a2) + eval(b2)) * double(b2 - a2);
  }
  return float(acc);
}

std::array<float, 3> DenseSpectrum::xyz(cie::Observer obs) const {
  std::array<float, 3> r{};
  for (int c = 0; c < 3; ++c) r[c] = ((*this) * cmf(c, obs)).integral(cie::kLambdaMin, cie::kLambdaMax);
  return r;
}

DenseSpectrum from_bands(const std::vector<float>& bands, const BandGrid& g) {
  DenseSpectrum s;
  for (int i = 0; i < DenseSpectrum::kSamples; ++i) s.at_index(i) = bands[g.nearest_band(DenseSpectrum::lambda_at(i))];
  return s;
}

const DenseSpectrum& d65_normalized() {
  static const DenseSpectrum d = [] {
    DenseSpectrum d65 = DenseSpectrum::cie_illuminant("D65");
    float y_d65 = d65.xyz()[1];
    float y_one = DenseSpectrum::constant(1.f).xyz()[1];
    return d65 * (y_one / y_d65);
  }();
  return d;
}

// ---------------------------------------------------------------- colorimetry
std::vector<float> band_cmf_weights(const BandGrid& g, cie::Observer obs) {
  std::vector<float> w(3 * g.n);
  for (int c = 0; c < 3; ++c) {
    DenseSpectrum m = DenseSpectrum::cmf(c, obs);
    for (int b = 0; b < g.n; ++b)
      w[c * g.n + b] = m.integral(g.center(b) - 0.5f * g.step, g.center(b) + 0.5f * g.step);
  }
  return w;
}

Vec3d bands_to_xyz(const float* bands, const BandGrid& g, const std::vector<float>& w) {
  Vec3d r{0, 0, 0};
  for (int c = 0; c < 3; ++c)
    for (int b = 0; b < g.n; ++b) r[c] += double(bands[b]) * double(w[c * g.n + b]);
  return r;
}

Vec3d xyz_to_linear_srgb(const Vec3d& v) {
  return {3.2404542 * v[0] - 1.5371385 * v[1] - 0.4985314 * v[2],
          -0.9692660 * v[0] + 1.8760108 * v[1] + 0.0415560 * v[2],
          0.0556434 * v[0] - 0.2040259 * v[1] + 1.0572252 * v[2]};
}
Vec3d linear_srgb_to_xyz(const Vec3d& v) {
  return {0.4124564 * v[0] + 0.3575761 * v[1] + 0.1804375 * v[2],
          0.2126729 * v[0] + 0.7151522 * v[1] + 0.0721750 * v[2],
          0.0193339 * v[0] + 0.1191920 * v[1] + 0.9503041 * v[2]};
}
Vec3d xyz_to_lab(const Vec3d& v, const Vec3d& w) {
  auto f = [](double t) {
    const double d = 6.0 / 29.0;
    return t > d * d * d ? std::cbrt(t) : t / (3 * d * d) + 4.0 / 29.0;
  };
  double fx = f(v[0] / w[0]), fy = f(v[1] / w[1]), fz = f(v[2] / w[2]);
  return {116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)};
}
double delta_e76(const Vec3d& a, const Vec3d& b) {
  return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}
float srgb_encode(float x) {
  x = std::max(0.f, x);
  return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f;
}
float srgb_decode(float x) { return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f); }

}  // namespace spectral
