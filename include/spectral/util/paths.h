#pragma once
#include <string>

namespace spectral {

// Data directory: $SPECTRAL_DATA_DIR, else the source-tree data dir, else the install data dir.
std::string data_dir();
// Cache directory: $SPECTRAL_CACHE_DIR, else $XDG_CACHE_HOME/spectral, else ~/.cache/spectral (created).
std::string cache_dir();
// Expands ${SPECTRAL_DATA}, ${SCENE_DIR} and ${ENV_VAR} references.
std::string expand_vars(const std::string& s, const std::string& scene_dir);
// Expands variables and resolves relative paths against base_dir.
std::string resolve_path(const std::string& p, const std::string& base_dir);

}  // namespace spectral
