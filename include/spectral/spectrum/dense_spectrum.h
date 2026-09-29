// Host-side densely sampled spectrum (1 nm grid, 300..1100 nm) used for loading,
// resampling onto the output band grid, and colorimetry. Not used in kernels.
#pragma once
#include <array>
#include <string>
#include <vector>

#include "spectral/spectrum/cie.h"
#include "spectral/spectrum/spectrum.h"

namespace spectral {

enum class Extrapolation { Clamp, Zero };

class DenseSpectrum {
 public:
  static constexpr float kLambdaMin = 300.f;
  static constexpr float kLambdaMax = 1100.f;
  static constexpr int kSamples = 801;

  DenseSpectrum() { values_.fill(0.f); }

  static DenseSpectrum constant(float c);
  // Piecewise-linear from (wavelength, value) samples; wavelengths must be ascending.
  static DenseSpectrum from_samples(const std::vector<float>& wavelengths, const std::vector<float>& values,
                                    Extrapolation extrap = Extrapolation::Clamp);
  // Planck's law, spectral radiance in W / (m^2 sr nm).
  static DenseSpectrum blackbody(float temperature_k);
  // Named CIE illuminants: "D65" (100 at 560 nm, clamp-held beyond 830 nm),
  // "A" (analytic, defined for all wavelengths, 100 at 560 nm), "E" (constant 1).
  static DenseSpectrum cie_illuminant(const std::string& name);
  // Color matching function component (0=x, 1=y, 2=z); zero outside 360..830.
  static DenseSpectrum cmf(int component, cie::Observer obs = cie::Observer::CIE1931_2deg);

  float eval(float lambda_nm) const;
  float& at_index(int i) { return values_[i]; }
  float at_index(int i) const { return values_[i]; }
  static float lambda_at(int i) { return kLambdaMin + float(i); }

  DenseSpectrum operator*(const DenseSpectrum& o) const;
  DenseSpectrum operator*(float s) const;
  DenseSpectrum operator+(const DenseSpectrum& o) const;
  float max_value() const;

  // Box-average onto the output band grid (each band covers center +- step/2).
  std::vector<float> to_bands(const BandGrid& grid) const;
  // Integral of this spectrum over [lo, hi] nm (trapezoidal on the 1 nm grid).
  float integral(float lo = kLambdaMin, float hi = kLambdaMax) const;
  // Unnormalized tristimulus integral: X = sum S(l) xbar(l) dl over 360..830 nm.
  std::array<float, 3> xyz(cie::Observer obs = cie::Observer::CIE1931_2deg) const;

 private:
  std::array<float, kSamples> values_;
};

// Reconstruct a (piecewise-constant per band) dense spectrum from band values.
DenseSpectrum from_bands(const std::vector<float>& bands, const BandGrid& grid);

// Relative D65 normalized so that an RGB (1,1,1) illuminant has the same luminance as a
// constant spectrum of value 1: Y(D65n) = Y(1). Used for RGB -> emission uplifting.
const DenseSpectrum& d65_normalized();

}  // namespace spectral
