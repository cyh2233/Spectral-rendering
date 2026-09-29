#include "spectral/spectrum/glass.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "spectral/util/paths.h"

namespace spectral {

double GlassModel::ior(double lambda_nm) const {
  double l = lambda_nm * 1e-3, l2 = l * l;
  if (model == "sellmeier") {
    double n2 = 1.0;
    for (size_t i = 0; i < B.size(); ++i) n2 += B[i] * l2 / (l2 - C[i]);
    return std::sqrt(std::max(n2, 1.0));
  }
  double n = 0.0;
  for (auto& [c, p] : series) n += c * std::pow(l, p);
  return n;
}

DenseSpectrum GlassModel::spectrum() const {
  DenseSpectrum s;
  for (int i = 0; i < DenseSpectrum::kSamples; ++i) s.at_index(i) = float(ior(DenseSpectrum::lambda_at(i)));
  return s;
}

GlassModel find_glass(const std::string& name, const std::string& catalog_path) {
  std::string path = catalog_path.empty() ? (std::filesystem::path(data_dir()) / "glass" / "catalog.json").string()
                                          : catalog_path;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open glass catalog " + path);
  nlohmann::json j = nlohmann::json::parse(in);
  if (!j.contains(name)) throw std::runtime_error("Unknown glass '" + name + "' in " + path);
  const auto& g = j[name];
  GlassModel m;
  m.name = name;
  m.model = g.at("model").get<std::string>();
  if (m.model == "sellmeier") {
    m.B = g.at("B").get<std::vector<double>>();
    m.C = g.at("C").get<std::vector<double>>();
    if (m.B.size() != m.C.size()) throw std::runtime_error("Sellmeier B/C size mismatch for " + name);
  } else if (m.model == "power_series") {
    for (const auto& t : g.at("coefficients")) m.series.push_back({t.at(0).get<double>(), t.at(1).get<double>()});
  } else {
    throw std::runtime_error("Unknown glass model '" + m.model + "'");
  }
  return m;
}

}  // namespace spectral
