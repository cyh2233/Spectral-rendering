#include "spectral/scene/gltf_loader.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>

#include "tiny_gltf.h"

namespace spectral {

namespace {

Affine node_local_transform(const tinygltf::Node& n) {
  Affine a = Affine::identity();
  if (n.matrix.size() == 16) {
    // glTF matrices are column-major.
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c) a.m[r][c] = float(n.matrix[c * 4 + r]);
    return a;
  }
  double t[3] = {0, 0, 0}, q[4] = {0, 0, 0, 1}, s[3] = {1, 1, 1};
  if (n.translation.size() == 3)
    for (int i = 0; i < 3; ++i) t[i] = n.translation[i];
  if (n.rotation.size() == 4)
    for (int i = 0; i < 4; ++i) q[i] = n.rotation[i];
  if (n.scale.size() == 3)
    for (int i = 0; i < 3; ++i) s[i] = n.scale[i];
  double x = q[0], y = q[1], z = q[2], w = q[3];
  double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                    {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                    {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) a.m[r][c] = float(R[r][c] * s[c]);
    a.m[r][3] = float(t[r]);
  }
  return a;
}

template <class T>
T read_component(const unsigned char* p, int ctype, bool normalized) {
  switch (ctype) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
      float f;
      std::memcpy(&f, p, 4);
      return T(f);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: return normalized ? T(*p / 255.0) : T(*p);
    case TINYGLTF_COMPONENT_TYPE_BYTE: {
      int8_t v;
      std::memcpy(&v, p, 1);
      return normalized ? T(std::max(v / 127.0, -1.0)) : T(v);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
      uint16_t v;
      std::memcpy(&v, p, 2);
      return normalized ? T(v / 65535.0) : T(v);
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT: {
      int16_t v;
      std::memcpy(&v, p, 2);
      return normalized ? T(std::max(v / 32767.0, -1.0)) : T(v);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
      uint32_t v;
      std::memcpy(&v, p, 4);
      return T(v);
    }
  }
  throw std::runtime_error("glTF: unsupported component type");
}

// Reads an accessor into a flat vector of `ncomp` components per element.
template <class T>
std::vector<T> read_accessor(const tinygltf::Model& m, int index, int ncomp) {
  const tinygltf::Accessor& acc = m.accessors.at(index);
  int comps = tinygltf::GetNumComponentsInType(acc.type);
  int csize = tinygltf::GetComponentSizeInBytes(acc.componentType);
  if (comps < ncomp) throw std::runtime_error("glTF: accessor has too few components");
  std::vector<T> out(acc.count * ncomp, T(0));
  if (acc.bufferView < 0) return out;  // all zeros (sparse not supported)
  const tinygltf::BufferView& bv = m.bufferViews.at(acc.bufferView);
  const tinygltf::Buffer& buf = m.buffers.at(bv.buffer);
  size_t stride = bv.byteStride ? bv.byteStride : size_t(comps * csize);
  const unsigned char* base = buf.data.data() + bv.byteOffset + acc.byteOffset;
  for (size_t i = 0; i < acc.count; ++i)
    for (int c = 0; c < ncomp; ++c)
      out[i * ncomp + c] = read_component<T>(base + i * stride + c * csize, acc.componentType, acc.normalized);
  return out;
}

double ext_number(const tinygltf::ExtensionMap& ext, const char* ext_name, const char* key, double def) {
  auto it = ext.find(ext_name);
  if (it == ext.end() || !it->second.Has(key)) return def;
  const tinygltf::Value& v = it->second.Get(key);
  return v.IsNumber() ? v.GetNumberAsDouble() : def;
}

}  // namespace

GltfAsset load_gltf(const std::string& path, Scene& scene) {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string err, warn;
  bool ok = (path.size() > 4 && path.substr(path.size() - 4) == ".glb")
                ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                : loader.LoadASCIIFromFile(&model, &err, &warn, path);
  if (!ok) throw std::runtime_error("Failed to load glTF '" + path + "': " + err);

  GltfAsset asset;
  asset.path = path;

  // Textures (cached per (texture, srgb)).
  std::map<std::pair<int, bool>, int32_t> tex_cache;
  auto get_texture = [&](int tex_index, bool srgb) -> int32_t {
    if (tex_index < 0 || tex_index >= int(model.textures.size())) return -1;
    auto key = std::make_pair(tex_index, srgb);
    auto it = tex_cache.find(key);
    if (it != tex_cache.end()) return it->second;
    const tinygltf::Texture& t = model.textures[tex_index];
    if (t.source < 0) return -1;
    const tinygltf::Image& img = model.images.at(t.source);
    if (img.image.empty() || img.width <= 0) return -1;
    TextureData td;
    td.width = img.width;
    td.height = img.height;
    td.srgb = srgb;
    td.name = img.name.empty() ? img.uri : img.name;
    td.rgba.resize(size_t(img.width) * img.height * 4);
    int comp = img.component;
    bool is16 = img.bits == 16;
    for (size_t p = 0; p < size_t(img.width) * img.height; ++p) {
      for (int c = 0; c < 4; ++c) {
        uint8_t v = c == 3 ? 255 : 0;
        if (c < comp) {
          v = is16 ? img.image[(p * comp + c) * 2 + 1] : img.image[p * comp + c];
        } else if (comp == 1 && c < 3) {
          v = img.image[is16 ? p * 2 + 1 : p];  // gray -> rgb
        }
        td.rgba[p * 4 + c] = v;
      }
    }
    if (t.sampler >= 0) {
      int ws = model.samplers[t.sampler].wrapS;
      td.wrap = ws == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE ? kWrapClamp
                : ws == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT ? kWrapMirror
                                                              : kWrapRepeat;
    }
    int32_t id = scene.add_texture(std::move(td));
    tex_cache[key] = id;
    return id;
  };

  // Materials.
  std::vector<int32_t> mat_ids(model.materials.size(), -1);
  for (size_t i = 0; i < model.materials.size(); ++i) {
    const tinygltf::Material& gm = model.materials[i];
    MaterialRecord m;
    const auto& pbr = gm.pbrMetallicRoughness;
    for (int c = 0; c < 4 && c < int(pbr.baseColorFactor.size()); ++c) m.base_color[c] = float(pbr.baseColorFactor[c]);
    m.base_color_tex = get_texture(pbr.baseColorTexture.index, true);
    m.metallic = float(pbr.metallicFactor);
    m.roughness = float(pbr.roughnessFactor);
    m.metal_rough_tex = get_texture(pbr.metallicRoughnessTexture.index, false);
    m.normal_tex = get_texture(gm.normalTexture.index, false);
    m.normal_scale = float(gm.normalTexture.scale);
    for (int c = 0; c < 3 && c < int(gm.emissiveFactor.size()); ++c) m.emissive[c] = float(gm.emissiveFactor[c]);
    m.emissive_tex = get_texture(gm.emissiveTexture.index, true);
    m.emissive_strength = float(ext_number(gm.extensions, "KHR_materials_emissive_strength", "emissiveStrength", 1.0));
    m.ior = float(ext_number(gm.extensions, "KHR_materials_ior", "ior", 1.5));
    m.alpha_mode = gm.alphaMode == "MASK" ? kAlphaMask : (gm.alphaMode == "BLEND" ? kAlphaBlend : kAlphaOpaque);
    m.alpha_cutoff = float(gm.alphaCutoff);
    m.double_sided = gm.doubleSided ? 1 : 0;
    double transmission = ext_number(gm.extensions, "KHR_materials_transmission", "transmissionFactor", 0.0);
    if (transmission >= 0.5) {
      double thickness = ext_number(gm.extensions, "KHR_materials_volume", "thicknessFactor", 0.0);
      m.type = thickness > 0.0 ? kMatDielectric : kMatThinDielectric;
    }
    mat_ids[i] = scene.add_material(m, gm.name.empty() ? "material_" + std::to_string(i) : gm.name);
  }

  // Meshes: one Scene mesh per primitive.
  std::vector<std::vector<std::pair<uint32_t, int>>> mesh_prims(model.meshes.size());  // (scene mesh, gltf material)
  for (size_t mi = 0; mi < model.meshes.size(); ++mi) {
    const tinygltf::Mesh& gmesh = model.meshes[mi];
    for (size_t pi = 0; pi < gmesh.primitives.size(); ++pi) {
      const tinygltf::Primitive& prim = gmesh.primitives[pi];
      int mode = prim.mode < 0 ? TINYGLTF_MODE_TRIANGLES : prim.mode;
      if (mode != TINYGLTF_MODE_TRIANGLES && mode != TINYGLTF_MODE_TRIANGLE_STRIP && mode != TINYGLTF_MODE_TRIANGLE_FAN)
        continue;
      auto pos_it = prim.attributes.find("POSITION");
      if (pos_it == prim.attributes.end()) continue;
      MeshData md;
      md.name = gmesh.name + "#" + std::to_string(pi);
      auto pos = read_accessor<float>(model, pos_it->second, 3);
      size_t nv = pos.size() / 3;
      md.positions.resize(nv);
      for (size_t v = 0; v < nv; ++v) md.positions[v] = Vec3(pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]);
      auto n_it = prim.attributes.find("NORMAL");
      if (n_it != prim.attributes.end()) {
        auto nrm = read_accessor<float>(model, n_it->second, 3);
        md.normals.resize(nv);
        for (size_t v = 0; v < nv; ++v) md.normals[v] = Vec3(nrm[3 * v], nrm[3 * v + 1], nrm[3 * v + 2]);
      }
      auto uv_it = prim.attributes.find("TEXCOORD_0");
      if (uv_it != prim.attributes.end()) {
        auto uv = read_accessor<float>(model, uv_it->second, 2);
        md.uvs.resize(nv);
        for (size_t v = 0; v < nv; ++v) md.uvs[v] = Vec2(uv[2 * v], uv[2 * v + 1]);
      }
      std::vector<uint32_t> idx;
      if (prim.indices >= 0) {
        idx = read_accessor<uint32_t>(model, prim.indices, 1);
      } else {
        idx.resize(nv);
        for (size_t v = 0; v < nv; ++v) idx[v] = uint32_t(v);
      }
      if (mode == TINYGLTF_MODE_TRIANGLES) {
        md.indices = idx;
      } else if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
        for (size_t k = 2; k < idx.size(); ++k) {
          if (k % 2 == 0) md.indices.insert(md.indices.end(), {idx[k - 2], idx[k - 1], idx[k]});
          else md.indices.insert(md.indices.end(), {idx[k - 1], idx[k - 2], idx[k]});
        }
      } else {
        for (size_t k = 2; k < idx.size(); ++k) md.indices.insert(md.indices.end(), {idx[0], idx[k - 1], idx[k]});
      }
      md.indices.resize(md.indices.size() / 3 * 3);
      if (md.indices.empty()) continue;
      md.material = prim.material >= 0 ? mat_ids[prim.material] : -1;
      mesh_prims[mi].push_back({scene.add_mesh(md), prim.material});
    }
  }

  // Node hierarchy.
  std::function<void(int, const Affine&)> visit = [&](int ni, const Affine& parent) {
    const tinygltf::Node& node = model.nodes.at(ni);
    Affine xf = compose(parent, node_local_transform(node));
    if (node.mesh >= 0) {
      for (auto& [smesh, gmat] : mesh_prims[node.mesh]) {
        GltfPart part;
        part.mesh = smesh;
        part.node_to_asset = xf;
        part.material = gmat >= 0 ? mat_ids[gmat] : -1;
        part.material_name = gmat >= 0 ? scene.material_names()[mat_ids[gmat]] : "";
        part.node_name = node.name;
        asset.parts.push_back(part);
      }
    }
    for (int c : node.children) visit(c, xf);
  };
  int sc = model.defaultScene >= 0 ? model.defaultScene : 0;
  if (!model.scenes.empty()) {
    for (int root : model.scenes.at(sc).nodes) visit(root, Affine::identity());
  } else {
    for (size_t i = 0; i < model.nodes.size(); ++i) visit(int(i), Affine::identity());
  }
  return asset;
}

}  // namespace spectral
