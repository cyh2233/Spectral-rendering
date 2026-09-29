// Scene description (JSON) loading. Schema: docs/scene_json_schema.md.
#pragma once
#include <memory>
#include <string>

#include "nlohmann/json.hpp"
#include "spectral/scene/scene.h"

namespace spectral {

std::shared_ptr<Scene> load_scene_json_file(const std::string& path);
std::shared_ptr<Scene> load_scene_json(const nlohmann::json& j, const std::string& base_dir);

}  // namespace spectral
