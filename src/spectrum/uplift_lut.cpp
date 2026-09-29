#include "spectral/spectrum/uplift_lut.h"

#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "spectral/util/paths.h"

// Provided by third_party/rgb2spec/rgb2spec_opt.cpp compiled with -Dmain=spectral_rgb2spec_main.
extern "C++" int spectral_rgb2spec_main(int argc, char** argv);

namespace fs = std::filesystem;

namespace spectral {

void generate_uplift_lut(const std::string& output_path, int res) {
  std::string r = std::to_string(res), out = output_path, gamut = "sRGB", prog = "rgb2spec_opt";
  std::vector<char*> argv = {prog.data(), r.data(), out.data(), gamut.data()};
  std::fprintf(stderr, "[spectral] generating RGB->spectrum table (res %d) -> %s\n", res, output_path.c_str());
  spectral_rgb2spec_main(int(argv.size()), argv.data());
}

std::shared_ptr<const UpliftLut> obtain_uplift_lut(const std::string& explicit_path, int res) {
  static std::mutex mu;
  static std::map<std::string, std::shared_ptr<const UpliftLut>> cache;
  std::lock_guard<std::mutex> lock(mu);
  std::vector<std::string> candidates;
  if (!explicit_path.empty()) candidates.push_back(explicit_path);
  if (const char* e = std::getenv("SPECTRAL_UPLIFT_LUT")) candidates.push_back(e);
  std::string name = "srgb_" + std::to_string(res) + ".coeff";
  candidates.push_back((fs::path(data_dir()) / "uplift" / name).string());
  std::string cache_file = (fs::path(cache_dir()) / name).string();
  candidates.push_back(cache_file);
  for (const auto& c : candidates) {
    auto it = cache.find(c);
    if (it != cache.end()) return it->second;
    if (fs::exists(c)) {
      auto lut = std::make_shared<const UpliftLut>(UpliftLut::load(c));
      cache[c] = lut;
      return lut;
    }
  }
  if (!explicit_path.empty() && !fs::exists(explicit_path)) {
    generate_uplift_lut(explicit_path, res);
    auto lut = std::make_shared<const UpliftLut>(UpliftLut::load(explicit_path));
    cache[explicit_path] = lut;
    return lut;
  }
  generate_uplift_lut(cache_file, res);
  auto lut = std::make_shared<const UpliftLut>(UpliftLut::load(cache_file));
  cache[cache_file] = lut;
  return lut;
}

}  // namespace spectral
