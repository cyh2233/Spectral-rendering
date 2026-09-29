#include "spectral/io/image_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "spectral/spectrum/colorimetry.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "tinyexr.h"

namespace spectral {

std::string band_channel_name(float l) {
  char buf[32];
  if (std::fabs(l - std::round(l)) < 1e-3f) std::snprintf(buf, sizeof buf, "L_%04d", int(std::lround(l)));
  else std::snprintf(buf, sizeof buf, "L_%07.2f", double(l));
  return buf;
}

std::vector<float> SpectralImage::wavelengths() const {
  std::vector<float> w(grid.n);
  for (int i = 0; i < grid.n; ++i) w[i] = grid.center(i);
  return w;
}

// ------------------------------------------------------------------ EXR
namespace {
struct ExrAttr {
  std::string name, type;
  std::vector<uint8_t> value;
};
ExrAttr attr_float(const std::string& n, float v) {
  ExrAttr a{n, "float", std::vector<uint8_t>(4)};
  std::memcpy(a.value.data(), &v, 4);
  return a;
}
ExrAttr attr_int(const std::string& n, int v) {
  ExrAttr a{n, "int", std::vector<uint8_t>(4)};
  std::memcpy(a.value.data(), &v, 4);
  return a;
}
ExrAttr attr_string(const std::string& n, const std::string& v) {
  return ExrAttr{n, "string", std::vector<uint8_t>(v.begin(), v.end())};
}
}  // namespace

void write_spectral_exr(const SpectralImage& img, const std::string& path, bool half) {
  const int W = img.width, H = img.height, B = img.grid.n;
  struct Chan {
    std::string name;
    int type;  // TINYEXR_PIXELTYPE_*
    std::vector<uint8_t> data;
  };
  std::vector<Chan> chans;
  for (int b = 0; b < B; ++b) {
    Chan c{band_channel_name(img.grid.center(b)), TINYEXR_PIXELTYPE_FLOAT, std::vector<uint8_t>(size_t(W) * H * 4)};
    float* d = reinterpret_cast<float*>(c.data.data());
    for (size_t p = 0; p < size_t(W) * H; ++p) d[p] = img.radiance[p * B + b];
    chans.push_back(std::move(c));
  }
  if (!img.depth.empty()) {
    Chan c{"depth", TINYEXR_PIXELTYPE_FLOAT, std::vector<uint8_t>(size_t(W) * H * 4)};
    std::memcpy(c.data.data(), img.depth.data(), c.data.size());
    chans.push_back(std::move(c));
  }
  if (!img.seg_id.empty()) {
    Chan c{"seg_id", TINYEXR_PIXELTYPE_UINT, std::vector<uint8_t>(size_t(W) * H * 4)};
    std::memcpy(c.data.data(), img.seg_id.data(), c.data.size());
    chans.push_back(std::move(c));
  }
  std::sort(chans.begin(), chans.end(), [](const Chan& a, const Chan& b) { return a.name < b.name; });

  EXRHeader header;
  InitEXRHeader(&header);
  EXRImage image;
  InitEXRImage(&image);
  std::vector<unsigned char*> ptrs(chans.size());
  std::vector<EXRChannelInfo> infos(chans.size());
  std::vector<int> ptypes(chans.size()), rtypes(chans.size());
  for (size_t i = 0; i < chans.size(); ++i) {
    ptrs[i] = chans[i].data.data();
    std::memset(&infos[i], 0, sizeof(EXRChannelInfo));
    std::strncpy(infos[i].name, chans[i].name.c_str(), 255);
    ptypes[i] = chans[i].type;
    bool is_band = chans[i].name.rfind("L_", 0) == 0;
    rtypes[i] = (half && is_band) ? TINYEXR_PIXELTYPE_HALF : chans[i].type;
  }
  image.images = ptrs.data();
  image.num_channels = int(chans.size());
  image.width = W;
  image.height = H;
  header.num_channels = int(chans.size());
  header.channels = infos.data();
  header.pixel_types = ptypes.data();
  header.requested_pixel_types = rtypes.data();
  header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;

  std::vector<ExrAttr> attrs = {attr_float("spectral_lambda_min", img.grid.lambda_min),
                                attr_float("spectral_lambda_step", img.grid.step),
                                attr_int("spectral_band_count", B),
                                attr_string("spectral_units", "W/(m^2 sr nm)"),
                                attr_int("spectral_spp", img.spp),
                                attr_string("spectral_metadata", img.metadata_json)};
  std::vector<EXRAttribute> xattrs(attrs.size());
  for (size_t i = 0; i < attrs.size(); ++i) {
    std::memset(&xattrs[i], 0, sizeof(EXRAttribute));
    std::strncpy(xattrs[i].name, attrs[i].name.c_str(), 255);
    std::strncpy(xattrs[i].type, attrs[i].type.c_str(), 255);
    xattrs[i].value = attrs[i].value.data();
    xattrs[i].size = int(attrs[i].value.size());
  }
  header.num_custom_attributes = int(xattrs.size());
  header.custom_attributes = xattrs.data();

  const char* err = nullptr;
  int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
  // Header/image point to our own storage; do not call Free* on them.
  if (ret != TINYEXR_SUCCESS) {
    std::string msg = err ? err : "unknown error";
    if (err) FreeEXRErrorMessage(err);
    throw std::runtime_error("EXR write failed (" + path + "): " + msg);
  }
}

SpectralImage read_spectral_exr(const std::string& path) {
  EXRVersion version;
  if (ParseEXRVersionFromFile(&version, path.c_str()) != TINYEXR_SUCCESS)
    throw std::runtime_error("Not an EXR file: " + path);
  EXRHeader header;
  InitEXRHeader(&header);
  const char* err = nullptr;
  if (ParseEXRHeaderFromFile(&header, &version, path.c_str(), &err) != TINYEXR_SUCCESS) {
    std::string msg = err ? err : "";
    FreeEXRErrorMessage(err);
    throw std::runtime_error("EXR header parse failed: " + msg);
  }
  for (int i = 0; i < header.num_channels; ++i)
    if (header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
  EXRImage image;
  InitEXRImage(&image);
  if (LoadEXRImageFromFile(&image, &header, path.c_str(), &err) != TINYEXR_SUCCESS) {
    std::string msg = err ? err : "";
    FreeEXRErrorMessage(err);
    FreeEXRHeader(&header);
    throw std::runtime_error("EXR load failed: " + msg);
  }
  SpectralImage img;
  img.width = image.width;
  img.height = image.height;
  float lmin = 0, step = 0;
  int count = 0;
  for (int i = 0; i < header.num_custom_attributes; ++i) {
    const EXRAttribute& a = header.custom_attributes[i];
    std::string n = a.name;
    if (n == "spectral_lambda_min") std::memcpy(&lmin, a.value, 4);
    if (n == "spectral_lambda_step") std::memcpy(&step, a.value, 4);
    if (n == "spectral_band_count") std::memcpy(&count, a.value, 4);
    if (n == "spectral_spp") std::memcpy(&img.spp, a.value, 4);
    if (n == "spectral_metadata") img.metadata_json.assign(reinterpret_cast<char*>(a.value), size_t(a.size));
  }
  if (count <= 0) throw std::runtime_error("EXR lacks spectral_* attributes: " + path);
  img.grid.lambda_min = lmin;
  img.grid.step = step;
  img.grid.n = count;
  const size_t np = size_t(img.width) * img.height;
  img.radiance.assign(np * count, 0.f);
  for (int c = 0; c < header.num_channels; ++c) {
    std::string name = header.channels[c].name;
    if (name == "depth") {
      img.depth.assign(np, 0.f);
      std::memcpy(img.depth.data(), image.images[c], np * 4);
    } else if (name == "seg_id") {
      img.seg_id.assign(np, 0);
      std::memcpy(img.seg_id.data(), image.images[c], np * 4);
    } else if (name.rfind("L_", 0) == 0) {
      float l = std::stof(name.substr(2));
      int b = img.grid.nearest_band(l);
      const float* src = reinterpret_cast<const float*>(image.images[c]);
      for (size_t p = 0; p < np; ++p) img.radiance[p * count + b] = src[p];
    }
  }
  FreeEXRImage(&image);
  FreeEXRHeader(&header);
  return img;
}

// ------------------------------------------------------------------ NPZ
uint32_t crc32(const uint8_t* data, size_t n, uint32_t crc) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
    init = true;
  }
  crc = ~crc;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

namespace {

std::vector<uint8_t> npy_bytes(const NpzArray& a) {
  std::ostringstream h;
  h << "{'descr': '" << a.dtype << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < a.shape.size(); ++i) h << a.shape[i] << (a.shape.size() == 1 ? "," : (i + 1 < a.shape.size() ? ", " : ""));
  h << "), }";
  std::string hs = h.str();
  size_t total = 10 + hs.size() + 1;
  size_t pad = (64 - total % 64) % 64;
  hs.append(pad, ' ');
  hs.push_back('\n');
  std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  uint16_t hl = uint16_t(hs.size());
  out.push_back(uint8_t(hl & 0xFF));
  out.push_back(uint8_t(hl >> 8));
  out.insert(out.end(), hs.begin(), hs.end());
  out.insert(out.end(), a.data.begin(), a.data.end());
  return out;
}

void put16(std::vector<uint8_t>& b, uint16_t v) {
  b.push_back(uint8_t(v));
  b.push_back(uint8_t(v >> 8));
}
void put32(std::vector<uint8_t>& b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i)));
}
void put64(std::vector<uint8_t>& b, uint64_t v) {
  for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i)));
}
uint16_t get16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t get32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
uint64_t get64(const uint8_t* p) { return uint64_t(get32(p)) | (uint64_t(get32(p + 4)) << 32); }

}  // namespace

void write_npz(const std::string& path, const std::map<std::string, NpzArray>& arrays, bool force_zip64) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot write " + path);
  struct Entry {
    std::string name;
    uint32_t crc;
    uint64_t size, offset;
    bool zip64;
  };
  std::vector<Entry> entries;
  uint64_t offset = 0;
  for (const auto& [key, arr] : arrays) {
    std::vector<uint8_t> payload = npy_bytes(arr);
    Entry e{key + ".npy", crc32(payload.data(), payload.size()), payload.size(), offset, false};
    e.zip64 = force_zip64 || e.size >= 0xFFFFFFFFull || offset >= 0xFFFFFFFFull;
    std::vector<uint8_t> h;
    put32(h, 0x04034b50);
    put16(h, e.zip64 ? 45 : 20);
    put16(h, 0);  // flags
    put16(h, 0);  // stored
    put16(h, 0);  // time
    put16(h, 0x21);  // date 1980-01-01
    put32(h, e.crc);
    put32(h, e.zip64 ? 0xFFFFFFFFu : uint32_t(e.size));
    put32(h, e.zip64 ? 0xFFFFFFFFu : uint32_t(e.size));
    put16(h, uint16_t(e.name.size()));
    put16(h, e.zip64 ? 20 : 0);
    h.insert(h.end(), e.name.begin(), e.name.end());
    if (e.zip64) {
      put16(h, 0x0001);
      put16(h, 16);
      put64(h, e.size);
      put64(h, e.size);
    }
    out.write(reinterpret_cast<const char*>(h.data()), std::streamsize(h.size()));
    out.write(reinterpret_cast<const char*>(payload.data()), std::streamsize(payload.size()));
    offset += h.size() + payload.size();
    entries.push_back(e);
  }
  uint64_t cd_start = offset;
  std::vector<uint8_t> cd;
  bool any64 = force_zip64;
  for (const Entry& e : entries) {
    bool off64 = e.zip64 && (force_zip64 || e.offset >= 0xFFFFFFFFull);
    put32(cd, 0x02014b50);
    put16(cd, e.zip64 ? 45 : 20);  // made by
    put16(cd, e.zip64 ? 45 : 20);  // needed
    put16(cd, 0);
    put16(cd, 0);
    put16(cd, 0);
    put16(cd, 0x21);
    put32(cd, e.crc);
    put32(cd, e.zip64 ? 0xFFFFFFFFu : uint32_t(e.size));
    put32(cd, e.zip64 ? 0xFFFFFFFFu : uint32_t(e.size));
    put16(cd, uint16_t(e.name.size()));
    put16(cd, e.zip64 ? uint16_t(off64 ? 28 : 20) : 0);
    put16(cd, 0);  // comment
    put16(cd, 0);  // disk
    put16(cd, 0);  // internal attrs
    put32(cd, 0);  // external attrs
    put32(cd, off64 ? 0xFFFFFFFFu : uint32_t(e.offset));
    cd.insert(cd.end(), e.name.begin(), e.name.end());
    if (e.zip64) {
      put16(cd, 0x0001);
      put16(cd, uint16_t(off64 ? 24 : 16));
      put64(cd, e.size);
      put64(cd, e.size);
      if (off64) put64(cd, e.offset);
      any64 = true;
    }
  }
  uint64_t cd_size = cd.size();
  if (any64 || cd_start >= 0xFFFFFFFFull) {
    uint64_t z64_eocd_off = cd_start + cd_size;
    put32(cd, 0x06064b50);
    put64(cd, 44);
    put16(cd, 45);
    put16(cd, 45);
    put32(cd, 0);
    put32(cd, 0);
    put64(cd, entries.size());
    put64(cd, entries.size());
    put64(cd, cd_size);
    put64(cd, cd_start);
    put32(cd, 0x07064b50);
    put32(cd, 0);
    put64(cd, z64_eocd_off);
    put32(cd, 1);
  }
  bool big = any64 || cd_start >= 0xFFFFFFFFull;
  put32(cd, 0x06054b50);
  put16(cd, 0);
  put16(cd, 0);
  put16(cd, big ? 0xFFFF : uint16_t(entries.size()));
  put16(cd, big ? 0xFFFF : uint16_t(entries.size()));
  put32(cd, big ? 0xFFFFFFFFu : uint32_t(cd_size));
  put32(cd, big ? 0xFFFFFFFFu : uint32_t(cd_start));
  put16(cd, 0);
  out.write(reinterpret_cast<const char*>(cd.data()), std::streamsize(cd.size()));
  if (!out) throw std::runtime_error("Write error: " + path);
}

std::map<std::string, NpzArray> read_npz(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open " + path);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::map<std::string, NpzArray> out;
  size_t pos = 0;
  while (pos + 30 <= buf.size() && get32(&buf[pos]) == 0x04034b50) {
    uint16_t method = get16(&buf[pos + 8]);
    uint32_t crc = get32(&buf[pos + 14]);
    uint64_t csize = get32(&buf[pos + 18]);
    uint16_t nlen = get16(&buf[pos + 26]), xlen = get16(&buf[pos + 28]);
    std::string name(reinterpret_cast<char*>(&buf[pos + 30]), nlen);
    size_t xp = pos + 30 + nlen;
    for (size_t q = xp; q + 4 <= xp + xlen;) {
      uint16_t id = get16(&buf[q]), sz = get16(&buf[q + 2]);
      if (id == 0x0001 && sz >= 16) csize = get64(&buf[q + 12]);
      q += 4 + sz;
    }
    if (method != 0) throw std::runtime_error("read_npz: only stored (uncompressed) entries supported");
    size_t dp = xp + xlen;
    if (dp + csize > buf.size()) throw std::runtime_error("read_npz: truncated");
    if (crc32(&buf[dp], csize) != crc) throw std::runtime_error("read_npz: CRC mismatch in " + name);
    // Parse NPY.
    const uint8_t* p = &buf[dp];
    if (p[0] != 0x93 || std::memcmp(p + 1, "NUMPY", 5) != 0) throw std::runtime_error("read_npz: bad npy magic");
    uint16_t hl = get16(p + 8);
    std::string header(reinterpret_cast<const char*>(p + 10), hl);
    NpzArray a;
    auto d0 = header.find("'descr': '") + 10;
    a.dtype = header.substr(d0, header.find('\'', d0) - d0);
    auto s0 = header.find("'shape': (") + 10;
    std::string shp = header.substr(s0, header.find(')', s0) - s0);
    std::stringstream ss(shp);
    std::string tok;
    while (std::getline(ss, tok, ','))
      if (tok.find_first_not_of(' ') != std::string::npos) a.shape.push_back(std::stoull(tok));
    a.data.assign(p + 10 + hl, p + csize);
    std::string key = name.size() > 4 ? name.substr(0, name.size() - 4) : name;
    out[key] = std::move(a);
    pos = dp + csize;
  }
  return out;
}

void write_spectral_npz(const SpectralImage& img, const std::string& path, bool force_zip64) {
  std::map<std::string, NpzArray> m;
  auto f32 = [](const std::vector<float>& v, std::vector<size_t> shape) {
    NpzArray a{"<f4", std::move(shape), std::vector<uint8_t>(v.size() * 4)};
    std::memcpy(a.data.data(), v.data(), a.data.size());
    return a;
  };
  size_t H = size_t(img.height), W = size_t(img.width), B = size_t(img.grid.n);
  m["radiance"] = f32(img.radiance, {H, W, B});
  m["wavelengths"] = f32(img.wavelengths(), {B});
  if (!img.depth.empty()) m["depth"] = f32(img.depth, {H, W});
  if (!img.seg_id.empty()) {
    NpzArray a{"<u4", {H, W}, std::vector<uint8_t>(img.seg_id.size() * 4)};
    std::memcpy(a.data.data(), img.seg_id.data(), a.data.size());
    m["seg_id"] = std::move(a);
  }
  NpzArray meta{"|S" + std::to_string(std::max<size_t>(1, img.metadata_json.size())), {},
                std::vector<uint8_t>(img.metadata_json.begin(), img.metadata_json.end())};
  if (meta.data.empty()) meta.data.push_back(0);
  m["metadata"] = std::move(meta);
  NpzArray spp{"<i4", {}, std::vector<uint8_t>(4)};
  std::memcpy(spp.data.data(), &img.spp, 4);
  m["spp"] = std::move(spp);
  write_npz(path, m, force_zip64);
}

SpectralImage read_spectral_npz(const std::string& path) {
  auto m = read_npz(path);
  SpectralImage img;
  const NpzArray& r = m.at("radiance");
  if (r.shape.size() != 3) throw std::runtime_error("radiance must be 3-D");
  img.height = int(r.shape[0]);
  img.width = int(r.shape[1]);
  img.radiance.resize(r.data.size() / 4);
  std::memcpy(img.radiance.data(), r.data.data(), r.data.size());
  const NpzArray& w = m.at("wavelengths");
  std::vector<float> wl(w.data.size() / 4);
  std::memcpy(wl.data(), w.data.data(), w.data.size());
  img.grid.n = int(wl.size());
  img.grid.lambda_min = wl.front();
  img.grid.step = wl.size() > 1 ? wl[1] - wl[0] : 1.f;
  if (m.count("depth")) {
    img.depth.resize(m["depth"].data.size() / 4);
    std::memcpy(img.depth.data(), m["depth"].data.data(), m["depth"].data.size());
  }
  if (m.count("seg_id")) {
    img.seg_id.resize(m["seg_id"].data.size() / 4);
    std::memcpy(img.seg_id.data(), m["seg_id"].data.data(), m["seg_id"].data.size());
  }
  if (m.count("metadata")) {
    auto& d = m["metadata"].data;
    img.metadata_json.assign(d.begin(), d.end());
  }
  if (m.count("spp")) std::memcpy(&img.spp, m["spp"].data.data(), 4);
  return img;
}

// ------------------------------------------------------------------ preview
void write_preview_png(const SpectralImage& img, const std::string& path, float exposure) {
  const size_t np = size_t(img.width) * img.height;
  std::vector<float> w = band_cmf_weights(img.grid);
  std::vector<Vec3d> rgb(np);
  // Normalize so that a constant spectrum of 1 has Y = 1.
  double ynorm = 0;
  for (int b = 0; b < img.grid.n; ++b) ynorm += w[img.grid.n + b];
  ynorm = ynorm > 0 ? 1.0 / ynorm : 1.0;
  double log_sum = 0;
  size_t count = 0;
  for (size_t p = 0; p < np; ++p) {
    Vec3d xyz = bands_to_xyz(img.radiance.data() + p * img.grid.n, img.grid, w);
    for (auto& c : xyz) c *= ynorm;
    rgb[p] = xyz_to_linear_srgb(xyz);
    if (xyz[1] > 0) {
      log_sum += std::log(xyz[1] + 1e-6);
      ++count;
    }
  }
  double scale = std::exp2(double(exposure));
  if (exposure == 0.f) scale = count ? 0.18 / std::exp(log_sum / double(count)) : 1.0;
  std::vector<uint8_t> px(np * 3);
  for (size_t p = 0; p < np; ++p)
    for (int c = 0; c < 3; ++c) {
      float v = srgb_encode(float(std::min(1.0, std::max(0.0, rgb[p][c] * scale))));
      px[p * 3 + c] = uint8_t(std::lround(v * 255.f));
    }
  if (!stbi_write_png(path.c_str(), img.width, img.height, 3, px.data(), img.width * 3))
    throw std::runtime_error("PNG write failed: " + path);
}

RgbImage load_rgb_image(const std::string& path) {
  RgbImage out;
  std::string lower = path;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower.size() > 4 && lower.substr(lower.size() - 4) == ".exr") {
    float* rgba = nullptr;
    const char* err = nullptr;
    if (LoadEXR(&rgba, &out.width, &out.height, path.c_str(), &err) != TINYEXR_SUCCESS) {
      std::string msg = err ? err : "";
      FreeEXRErrorMessage(err);
      throw std::runtime_error("Cannot load EXR " + path + ": " + msg);
    }
    out.rgb.resize(size_t(out.width) * out.height * 3);
    for (size_t p = 0; p < size_t(out.width) * out.height; ++p)
      for (int c = 0; c < 3; ++c) out.rgb[p * 3 + c] = rgba[p * 4 + c];
    free(rgba);
    return out;
  }
  int n = 0;
  if (stbi_is_hdr(path.c_str())) {
    float* d = stbi_loadf(path.c_str(), &out.width, &out.height, &n, 3);
    if (!d) throw std::runtime_error("Cannot load image " + path);
    out.rgb.assign(d, d + size_t(out.width) * out.height * 3);
    stbi_image_free(d);
  } else {
    unsigned char* d = stbi_load(path.c_str(), &out.width, &out.height, &n, 3);
    if (!d) throw std::runtime_error("Cannot load image " + path);
    out.rgb.resize(size_t(out.width) * out.height * 3);
    for (size_t i = 0; i < out.rgb.size(); ++i) out.rgb[i] = srgb_decode(d[i] / 255.f);
    stbi_image_free(d);
  }
  return out;
}

}  // namespace spectral
