// glTF 2.0 (.gltf/.glb) loading into a Scene.
#pragma once
#include <string>
#include <vector>

#include "spectral/scene/scene.h"

namespace spectral {

struct GltfPart {
  uint32_t mesh;         // Scene mesh index
  Affine node_to_asset;  // node transform relative to the asset root
  int32_t material;      // Scene material index (-1 = default)
  std::string material_name;
  std::string node_name;
};

struct GltfAsset {
  std::string path;
  std::vector<GltfPart> parts;
};

// Loads meshes, materials and textures into `scene`. glTF materials are mapped as:
//  pbrMetallicRoughness -> PBR; KHR_materials_transmission (factor >= 0.5) -> thin dielectric,
//  or solid dielectric if KHR_materials_volume.thicknessFactor > 0; KHR_materials_ior -> ior;
//  KHR_materials_emissive_strength -> emissive_strength.
GltfAsset load_gltf(const std::string& path, Scene& scene);

}  // namespace spectral
