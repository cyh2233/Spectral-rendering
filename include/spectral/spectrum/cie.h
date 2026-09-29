// CIE colorimetric tables (host only). 1 nm spacing, 360..830 nm.
#pragma once

namespace spectral::cie {

constexpr int kSamples = 471;
constexpr float kLambdaMin = 360.f;
constexpr float kLambdaMax = 830.f;

extern const float x31[kSamples], y31[kSamples], z31[kSamples];  // CIE 1931 2 deg
extern const float x64[kSamples], y64[kSamples], z64[kSamples];  // CIE 1964 10 deg
extern const float d65[kSamples];                                 // D65 relative SPD, 100 at 560 nm

// Linear interpolation into a table; returns 0 outside [360, 830].
float lookup(const float* table, float lambda_nm);

enum class Observer { CIE1931_2deg, CIE1964_10deg };

}  // namespace spectral::cie
