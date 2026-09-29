// CSV spectral data: first column wavelength (nm), one or more value columns.
// Lines starting with '#' are comments. A comment "# extrapolation: zero|clamp"
// selects the extrapolation policy (default clamp). An optional header row names columns.
#pragma once
#include <string>
#include <vector>

#include "spectral/spectrum/dense_spectrum.h"

namespace spectral {

struct CsvSpectra {
  std::string path;
  std::vector<std::string> column_names;  // value columns only
  std::vector<float> wavelengths;
  std::vector<std::vector<float>> columns;
  Extrapolation extrapolation = Extrapolation::Clamp;

  int column_index(const std::string& name) const;  // name or integer index; -1 if missing
  DenseSpectrum spectrum(int column) const;
};

CsvSpectra read_csv_spectra(const std::string& path);

}  // namespace spectral
