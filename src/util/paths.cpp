#include "spectral/util/paths.h"

#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace spectral {

std::string data_dir() {
  if (const char* e = std::getenv("SPECTRAL_DATA_DIR")) return e;
#ifdef SPECTRAL_SOURCE_DATA_DIR
  if (fs::exists(SPECTRAL_SOURCE_DATA_DIR)) return SPECTRAL_SOURCE_DATA_DIR;
#endif
#ifdef SPECTRAL_INSTALL_DATA_DIR
  return SPECTRAL_INSTALL_DATA_DIR;
#else
  return "data";
#endif
}

std::string cache_dir() {
  fs::path p;
  if (const char* e = std::getenv("SPECTRAL_CACHE_DIR")) p = e;
  else if (const char* x = std::getenv("XDG_CACHE_HOME")) p = fs::path(x) / "spectral";
  else if (const char* h = std::getenv("HOME")) p = fs::path(h) / ".cache" / "spectral";
  else if (const char* u = std::getenv("LOCALAPPDATA")) p = fs::path(u) / "spectral";
  else p = fs::temp_directory_path() / "spectral";
  std::error_code ec;
  fs::create_directories(p, ec);
  return p.string();
}

std::string expand_vars(const std::string& s, const std::string& scene_dir) {
  std::string out;
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '$' && i + 1 < s.size() && s[i + 1] == '{') {
      size_t end = s.find('}', i);
      if (end == std::string::npos) {
        out += s.substr(i);
        break;
      }
      std::string var = s.substr(i + 2, end - i - 2);
      if (var == "SPECTRAL_DATA") out += data_dir();
      else if (var == "SCENE_DIR") out += scene_dir;
      else if (const char* e = std::getenv(var.c_str())) out += e;
      i = end + 1;
    } else {
      out += s[i++];
    }
  }
  return out;
}

std::string resolve_path(const std::string& p, const std::string& base) {
  fs::path q = expand_vars(p, base);
  if (q.is_relative() && !base.empty()) q = fs::path(base) / q;
  return q.lexically_normal().string();
}

}  // namespace spectral
