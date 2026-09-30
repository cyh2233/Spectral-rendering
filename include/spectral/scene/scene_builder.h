// Scene construction from JSON that stays alive after the initial load, so live sessions can add
// assets, instances and lights, move the sun or change the camera incrementally.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "spectral/scene/scene.h"

namespace spectral {

struct InstancePart {
  uint32_t instance;       // Scene instance index
  Affine local;            // part transform relative to the instance transform (glTF node)
};

class SceneBuilder {
 public:
  SceneBuilder(nlohmann::json root, std::string base_dir);
  ~SceneBuilder();
  SceneBuilder(const SceneBuilder&) = delete;
  SceneBuilder& operator=(const SceneBuilder&) = delete;

  // Initial load of the whole JSON document (finalizes the scene).
  std::shared_ptr<Scene> build();

  // --- incremental API (call Scene::finalize() afterwards; Session does this)
  // Adds an instance described like an entry of "instances" (asset or primitive).
  std::vector<InstancePart> add_instance(const nlohmann::json& instance);
  void add_asset(const std::string& key, const std::string& path);
  std::vector<float> spectrum(const nlohmann::json& spec, bool illuminant);
  // Point / spot / directional light record; identical emission specs share one pooled spectrum.
  LightRecord make_light(const nlohmann::json& light);
  // Moves the sun of the scene's sky light (re-tabulates the sky; Prague tables are cached).
  void set_sun(Vec3 direction_to_sun);
  bool has_sky() const;
  void set_camera(const nlohmann::json& camera);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::unique_ptr<SceneBuilder> make_scene_builder_from_file(const std::string& path);

// Pose of an instance description: "transform", or translate / rotation | rotate_deg / scale.
Affine parse_instance_transform(const nlohmann::json& instance);

}  // namespace spectral
