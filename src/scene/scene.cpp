#include "spectral/scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "spectral/kernel/material.h"

namespace spectral {

UpliftLut UpliftLut::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open uplift LUT: " + path);
  char magic[4];
  in.read(magic, 4);
  if (std::string(magic, 4) != "SPEC") throw std::runtime_error("Not an rgb2spec coefficient file: " + path);
  uint32_t res = 0;
  in.read(reinterpret_cast<char*>(&res), 4);
  if (res < 2 || res > 1024) throw std::runtime_error("Invalid LUT resolution in " + path);
  UpliftLut lut;
  lut.res = int(res);
  lut.scale.resize(res);
  lut.data.resize(size_t(3) * res * res * res * 3);
  in.read(reinterpret_cast<char*>(lut.scale.data()), std::streamsize(res * sizeof(float)));
  in.read(reinterpret_cast<char*>(lut.data.data()), std::streamsize(lut.data.size() * sizeof(float)));
  if (!in) throw std::runtime_error("Truncated uplift LUT: " + path);
  return lut;
}

UpliftLutView UpliftLut::view() const {
  UpliftLutView v;
  v.res = res;
  v.scale = scale.data();
  v.data = data.data();
  return v;
}

std::vector<AliasEntry> build_alias_table(const std::vector<double>& w_in) {
  size_t n = w_in.size();
  std::vector<AliasEntry> t(n);
  if (n == 0) return t;
  double sum = 0;
  for (double w : w_in) sum += std::max(0.0, w);
  std::vector<double> p(n);
  for (size_t i = 0; i < n; ++i) p[i] = sum > 0 ? std::max(0.0, w_in[i]) / sum : 1.0 / double(n);
  std::vector<double> scaled(n);
  std::vector<uint32_t> small, large;
  for (size_t i = 0; i < n; ++i) {
    t[i].pmf = float(p[i]);
    scaled[i] = p[i] * double(n);
    (scaled[i] < 1.0 ? small : large).push_back(uint32_t(i));
  }
  while (!small.empty() && !large.empty()) {
    uint32_t s = small.back(), l = large.back();
    small.pop_back();
    t[s].prob = float(scaled[s]);
    t[s].alias = l;
    scaled[l] = (scaled[l] + scaled[s]) - 1.0;
    if (scaled[l] < 1.0) {
      large.pop_back();
      small.push_back(l);
    }
  }
  for (uint32_t i : large) {
    t[i].prob = 1.f;
    t[i].alias = i;
  }
  for (uint32_t i : small) {
    t[i].prob = 1.f;
    t[i].alias = i;
  }
  return t;
}

Scene::Scene() { set_band_grid(380.f, 1000.f, 5.f); }

void Scene::set_band_grid(float lmin, float lmax, float step) {
  if (!spectra_.empty()) throw std::runtime_error("Band grid must be set before adding spectra");
  if (!(step > 0.f) || !(lmax >= lmin)) throw std::runtime_error("Invalid band grid");
  int n = int(std::floor((lmax - lmin) / step + 1e-3f)) + 1;
  if (n > SPECTRAL_MAX_BANDS)
    throw std::runtime_error("Band grid has " + std::to_string(n) + " bands; maximum is " +
                             std::to_string(SPECTRAL_MAX_BANDS) + " (rebuild with larger SPECTRAL_MAX_BANDS)");
  grid.lambda_min = lmin;
  grid.step = step;
  grid.n = n;
  d65n_bands_ = d65_normalized().to_bands(grid);
}

int32_t Scene::add_band_spectrum(const std::vector<float>& b) {
  if (int(b.size()) != grid.n) throw std::runtime_error("Band spectrum size mismatch");
  int32_t off = int32_t(spectra_.size());
  spectra_.insert(spectra_.end(), b.begin(), b.end());
  return off;
}

std::vector<float> Scene::band_spectrum(int32_t off) const {
  return std::vector<float>(spectra_.begin() + off, spectra_.begin() + off + grid.n);
}

int32_t Scene::add_texture(TextureData tex) {
  if (tex.width <= 0 || tex.height <= 0 || tex.rgba.size() != size_t(tex.width) * tex.height * 4)
    throw std::runtime_error("Invalid texture '" + tex.name + "'");
  textures_.push_back(std::move(tex));
  return int32_t(textures_.size() - 1);
}

int32_t Scene::add_material(const MaterialRecord& m, const std::string& name) {
  materials_.push_back(m);
  material_names_.push_back(name);
  return int32_t(materials_.size() - 1);
}

uint32_t Scene::add_mesh(const MeshData& md) {
  if (md.indices.size() % 3 != 0) throw std::runtime_error("Mesh index count not a multiple of 3: " + md.name);
  for (uint32_t i : md.indices)
    if (i >= md.positions.size()) throw std::runtime_error("Mesh index out of range: " + md.name);
  MeshRecord m;
  m.vtx_offset = uint32_t(positions_.size());
  m.vtx_count = uint32_t(md.positions.size());
  m.tri_offset = uint32_t(indices_.size() / 3);
  m.tri_count = uint32_t(md.indices.size() / 3);
  m.material = md.material;
  bool has_n = md.normals.size() == md.positions.size() && !md.normals.empty();
  bool has_uv = md.uvs.size() == md.positions.size() && !md.uvs.empty();
  m.flags = (has_n ? kMeshHasNormals : 0u) | (has_uv ? kMeshHasUVs : 0u);
  positions_.insert(positions_.end(), md.positions.begin(), md.positions.end());
  // Keep normals/uvs arrays aligned with positions.
  if (has_n) normals_.insert(normals_.end(), md.normals.begin(), md.normals.end());
  else normals_.resize(positions_.size(), Vec3(0, 1, 0));
  if (has_uv) uvs_.insert(uvs_.end(), md.uvs.begin(), md.uvs.end());
  else uvs_.resize(positions_.size(), Vec2(0, 0));
  indices_.insert(indices_.end(), md.indices.begin(), md.indices.end());
  meshes_.push_back(m);
  finalized_ = false;
  return uint32_t(meshes_.size() - 1);
}

uint32_t Scene::add_instance(uint32_t mesh, const Affine& to_world, int32_t material_override, uint32_t seg_id) {
  if (mesh >= meshes_.size()) throw std::runtime_error("add_instance: bad mesh index");
  InstanceRecord r;
  r.mesh = mesh;
  r.to_world = to_world;
  r.to_local = inverse(to_world);
  r.material = material_override >= 0 ? material_override : meshes_[mesh].material;
  r.seg_id = seg_id;
  instances_.push_back(r);
  finalized_ = false;
  return uint32_t(instances_.size() - 1);
}

int32_t Scene::add_light(const LightRecord& l) {
  lights_.push_back(l);
  finalized_ = false;
  return int32_t(lights_.size() - 1);
}

void Scene::set_environment(int w, int h, std::vector<float> data, float scale, float rotation) {
  if (w <= 0 || h <= 0 || data.size() != size_t(w) * h * grid.n) throw std::runtime_error("Invalid environment table");
  env_data_ = std::move(data);
  env_.present = 1;
  env_.width = w;
  env_.height = h;
  env_.scale = scale;
  env_.rotation = rotation;
  // Sampling distribution: band-averaged radiance * sin(theta).
  env_dist_.func.assign(size_t(w) * h, 0.f);
  for (int y = 0; y < h; ++y) {
    float sin_t = std::sin(kPi * (float(y) + 0.5f) / float(h));
    for (int x = 0; x < w; ++x) {
      const float* p = env_data_.data() + (size_t(y) * w + x) * grid.n;
      double avg = 0;
      for (int b = 0; b < grid.n; ++b) avg += std::max(0.f, p[b]);
      env_dist_.func[size_t(y) * w + x] = float(avg / grid.n) * sin_t;
    }
  }
  double total = std::accumulate(env_dist_.func.begin(), env_dist_.func.end(), 0.0);
  if (!(total > 0)) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) env_dist_.func[size_t(y) * w + x] = std::sin(kPi * (float(y) + 0.5f) / float(h));
  }
  env_dist_.cond_cdf.assign(size_t(h) * (w + 1), 0.f);
  env_dist_.row_int.assign(h, 0.f);
  env_dist_.marg_cdf.assign(h + 1, 0.f);
  for (int y = 0; y < h; ++y) {
    float* c = env_dist_.cond_cdf.data() + size_t(y) * (w + 1);
    double acc = 0;
    c[0] = 0;
    for (int x = 0; x < w; ++x) {
      acc += env_dist_.func[size_t(y) * w + x] / double(w);
      c[x + 1] = float(acc);
    }
    env_dist_.row_int[y] = float(acc);
    for (int x = 1; x <= w; ++x) c[x] = acc > 0 ? float(c[x] / acc) : float(x) / float(w);
  }
  double macc = 0;
  for (int y = 0; y < h; ++y) {
    macc += env_dist_.row_int[y] / double(h);
    env_dist_.marg_cdf[y + 1] = float(macc);
  }
  env_dist_.marg_int = float(macc);
  for (int y = 1; y <= h; ++y) env_dist_.marg_cdf[y] = macc > 0 ? float(env_dist_.marg_cdf[y] / macc) : float(y) / h;
  finalized_ = false;
}

void Scene::set_sun(Vec3 dir, float half_angle, const std::vector<float>& radiance, float scale) {
  LightRecord l;
  l.type = kLightSun;
  l.direction = normalize(dir);
  l.cos_max = std::cos(half_angle);
  l.spectrum = add_band_spectrum(radiance);
  l.scale = scale;
  add_light(l);
}

void Scene::finalize() {
  // Texture views.
  texture_views_.resize(textures_.size());
  for (size_t i = 0; i < textures_.size(); ++i) {
    TextureView& v = texture_views_[i];
    v.width = textures_[i].width;
    v.height = textures_[i].height;
    v.srgb = textures_[i].srgb ? 1 : 0;
    v.wrap = textures_[i].wrap;
    v.data = textures_[i].rgba.data();
  }
  // Remove previously generated triangle/env lights (finalize may be called repeatedly).
  lights_.erase(std::remove_if(lights_.begin(), lights_.end(),
                               [](const LightRecord& l) { return l.type == kLightTriangle || l.type == kLightEnv; }),
                lights_.end());
  for (auto& inst : instances_) inst.light_offset = -1;

  params.has_glass = 0;
  for (const auto& m : materials_)
    if (m.type != kMatPbr) params.has_glass = 1;

  // Emissive triangles.
  for (uint32_t ii = 0; ii < instances_.size(); ++ii) {
    InstanceRecord& inst = instances_[ii];
    if (inst.material < 0 || !material_emits(materials_[inst.material])) continue;
    const MeshRecord& m = meshes_[inst.mesh];
    inst.light_offset = int32_t(lights_.size());
    for (uint32_t t = 0; t < m.tri_count; ++t) {
      const uint32_t* tri = indices_.data() + 3 * size_t(m.tri_offset + t);
      Vec3 p0 = inst.to_world.point(positions_[m.vtx_offset + tri[0]]);
      Vec3 p1 = inst.to_world.point(positions_[m.vtx_offset + tri[1]]);
      Vec3 p2 = inst.to_world.point(positions_[m.vtx_offset + tri[2]]);
      LightRecord l;
      l.type = kLightTriangle;
      l.instance = ii;
      l.prim = t;
      l.area = 0.5f * length(cross(p1 - p0, p2 - p0));
      lights_.push_back(l);
    }
  }
  env_light_ = -1;
  if (env_.present) {
    LightRecord l;
    l.type = kLightEnv;
    env_light_ = int32_t(lights_.size());
    lights_.push_back(l);
  }
  sun_light_ = -1;
  for (size_t i = 0; i < lights_.size(); ++i)
    if (lights_[i].type == kLightSun) sun_light_ = int32_t(i);

  // Selection weights: estimated irradiance at a reference distance.
  SceneView sv = view();  // for emission evaluation
  const double dref2 = double(render.light_reference_distance) * render.light_reference_distance;
  auto avg_spec = [&](int32_t off) {
    if (off < 0) return 0.0;
    double a = 0;
    for (int b = 0; b < grid.n; ++b) a += spectra_[off + b];
    return a / grid.n;
  };
  std::vector<double> w(lights_.size(), 0.0);
  for (size_t i = 0; i < lights_.size(); ++i) {
    const LightRecord& l = lights_[i];
    switch (l.type) {
      case kLightPoint:
      case kLightSpot: w[i] = avg_spec(l.spectrum) * l.scale / dref2; break;
      case kLightDirectional: w[i] = avg_spec(l.spectrum) * l.scale; break;
      case kLightSun: w[i] = avg_spec(l.spectrum) * l.scale * 2.0 * kPi * (1.0 - l.cos_max); break;
      case kLightEnv: {
        double a = 0;
        for (float v : env_data_) a += v;
        w[i] = kPi * env_.scale * a / std::max<size_t>(1, env_data_.size());
        break;
      }
      case kLightTriangle: {
        const MaterialRecord& m = materials_[instances_[l.instance].material];
        Spectrum le;
        evaluate_emission(sv, m, Vec2(0.5f, 0.5f), le);
        w[i] = le.average() * l.area / dref2;
        if (m.emissive_tex >= 0) w[i] = std::max(w[i], 1e-3 * l.area / dref2);
        break;
      }
    }
    if (!(w[i] >= 0.0) || !std::isfinite(w[i])) w[i] = 0.0;
  }
  // Keep every light samplable (avoid zero pmf with nonzero emission estimates due to textures).
  double wmax = w.empty() ? 0.0 : *std::max_element(w.begin(), w.end());
  for (auto& x : w)
    if (x <= 0.0 && wmax > 0.0) x = wmax * 1e-6;
  light_alias_ = build_alias_table(w);

  finalized_ = true;
}

SceneView Scene::view() const {
  SceneView v;
  v.grid = grid;
  v.positions = positions_.data();
  v.normals = normals_.data();
  v.uvs = uvs_.data();
  v.indices = indices_.data();
  v.meshes = meshes_.data();
  v.instances = instances_.data();
  v.n_instances = uint32_t(instances_.size());
  v.materials = materials_.data();
  // texture_views_ is rebuilt in finalize(); before that, build lazily is not possible in a
  // const method, so textures are only valid after finalize().
  v.textures = texture_views_.data();
  v.spectra = spectra_.data();
  v.d65n_bands = d65n_bands_.data();
  if (uplift) v.lut = uplift->view();
  v.lights = lights_.data();
  v.n_lights = uint32_t(lights_.size());
  v.light_alias = light_alias_.empty() ? nullptr : light_alias_.data();
  v.env_light = env_light_;
  v.sun_light = sun_light_;
  v.env = env_;
  v.env.data = env_data_.data();
  v.env.dist.nu = env_.width;
  v.env.dist.nv = env_.height;
  v.env.dist.func = env_dist_.func.data();
  v.env.dist.cond_cdf = env_dist_.cond_cdf.data();
  v.env.dist.row_int = env_dist_.row_int.data();
  v.env.dist.marg_cdf = env_dist_.marg_cdf.data();
  v.env.dist.marg_int = env_dist_.marg_int;
  v.camera = camera;
  v.params = params;
  return v;
}

}  // namespace spectral
