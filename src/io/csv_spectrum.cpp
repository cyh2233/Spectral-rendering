#include "spectral/io/csv_spectrum.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace spectral {

static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

static std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ',' || c == ';' || c == '\t') {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(trim(cur));
  return out;
}

static bool parse_float(const std::string& s, float* v) {
  if (s.empty()) return false;
  char* end = nullptr;
  *v = std::strtof(s.c_str(), &end);
  return end && *end == '\0';
}

int CsvSpectra::column_index(const std::string& name) const {
  for (size_t i = 0; i < column_names.size(); ++i)
    if (column_names[i] == name) return int(i);
  bool numeric = !name.empty() && std::all_of(name.begin(), name.end(), [](char c) { return std::isdigit(c); });
  if (numeric) {
    int i = std::stoi(name);
    if (i >= 0 && i < int(columns.size())) return i;
  }
  return -1;
}

DenseSpectrum CsvSpectra::spectrum(int column) const {
  if (column < 0 || column >= int(columns.size())) throw std::runtime_error("CSV column out of range in " + path);
  return DenseSpectrum::from_samples(wavelengths, columns[column], extrapolation);
}

CsvSpectra read_csv_spectra(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open spectrum CSV: " + path);
  CsvSpectra out;
  out.path = path;
  std::string line;
  int lineno = 0;
  std::vector<std::pair<float, std::vector<float>>> rows;
  while (std::getline(in, line)) {
    ++lineno;
    std::string t = trim(line);
    if (t.empty()) continue;
    if (t[0] == '#') {
      std::string lower = t;
      std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
      auto p = lower.find("extrapolation:");
      if (p != std::string::npos) {
        std::string v = trim(lower.substr(p + 14));
        out.extrapolation = (v.rfind("zero", 0) == 0) ? Extrapolation::Zero : Extrapolation::Clamp;
      }
      continue;
    }
    auto cells = split(t);
    float w;
    if (!parse_float(cells[0], &w)) {
      if (!rows.empty() || !out.column_names.empty())
        throw std::runtime_error(path + ":" + std::to_string(lineno) + ": unexpected non-numeric row");
      out.column_names.assign(cells.begin() + 1, cells.end());
      continue;
    }
    std::vector<float> vals;
    for (size_t i = 1; i < cells.size(); ++i) {
      float v;
      if (!parse_float(cells[i], &v))
        throw std::runtime_error(path + ":" + std::to_string(lineno) + ": bad number '" + cells[i] + "'");
      vals.push_back(v);
    }
    if (vals.empty()) throw std::runtime_error(path + ":" + std::to_string(lineno) + ": missing value column");
    if (!rows.empty() && vals.size() != rows.front().second.size())
      throw std::runtime_error(path + ":" + std::to_string(lineno) + ": inconsistent column count");
    rows.push_back({w, vals});
  }
  if (rows.empty()) throw std::runtime_error("No data rows in " + path);
  std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first < b.first; });
  size_t nc = rows.front().second.size();
  out.columns.assign(nc, {});
  for (auto& r : rows) {
    if (!out.wavelengths.empty() && r.first == out.wavelengths.back())
      throw std::runtime_error(path + ": duplicate wavelength " + std::to_string(r.first));
    out.wavelengths.push_back(r.first);
    for (size_t c = 0; c < nc; ++c) out.columns[c].push_back(r.second[c]);
  }
  if (out.column_names.size() != nc) {
    out.column_names.clear();
    for (size_t c = 0; c < nc; ++c) out.column_names.push_back(nc == 1 ? "value" : std::to_string(c));
  }
  return out;
}

}  // namespace spectral
