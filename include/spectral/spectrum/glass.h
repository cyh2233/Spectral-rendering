// Refractive index models for dielectrics (glass catalog in data/glass/catalog.json).
#pragma once
#include <string>
#include <vector>

#include "spectral/spectrum/dense_spectrum.h"

namespace spectral {

struct GlassModel {
  std::string name, model;              // "sellmeier" | "power_series"
  std::vector<double> B, C;             // sellmeier
  std::vector<std::pair<double, double>> series;  // power_series (coefficient, power)
  double ior(double lambda_nm) const;
  DenseSpectrum spectrum() const;
};

// Loads a glass by name from the catalog (default: <data>/glass/catalog.json).
GlassModel find_glass(const std::string& name, const std::string& catalog_path = "");

}  // namespace spectral
