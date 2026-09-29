/* C ABI for external spectral sky modules (e.g. a proprietary/encrypted sky model).
 * Build a shared library exporting these symbols and reference it from the scene JSON:
 *   { "type": "sky", "model": "plugin", "library": "libmysky.so", "options": {...} }
 * Directions are unit vectors in the renderer world frame (+Y up).
 * Radiance is spectral radiance in W / (m^2 sr nm). */
#ifndef SPECTRAL_SKY_PLUGIN_H
#define SPECTRAL_SKY_PLUGIN_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define SPECTRAL_SKY_EXPORT __declspec(dllexport)
#else
#define SPECTRAL_SKY_EXPORT __attribute__((visibility("default")))
#endif

#define SPECTRAL_SKY_ABI_VERSION 1

typedef struct spectral_sky spectral_sky;

/* Must return SPECTRAL_SKY_ABI_VERSION. */
SPECTRAL_SKY_EXPORT int spectral_sky_abi_version(void);
/* options_json: the "options" object of the light, serialized. NULL on failure. */
SPECTRAL_SKY_EXPORT spectral_sky* spectral_sky_create(const char* options_json);
/* sun_dir: unit vector towards the sun. Returns 0 on success. */
SPECTRAL_SKY_EXPORT int spectral_sky_configure(spectral_sky* sky, double sun_dir_x, double sun_dir_y,
                                               double sun_dir_z, double visibility_km, double ground_albedo,
                                               double altitude_m);
SPECTRAL_SKY_EXPORT double spectral_sky_radiance(const spectral_sky* sky, double dir_x, double dir_y, double dir_z,
                                                 double lambda_nm);
SPECTRAL_SKY_EXPORT double spectral_sky_sun_radiance(const spectral_sky* sky, double lambda_nm);
SPECTRAL_SKY_EXPORT double spectral_sky_sun_half_angle(const spectral_sky* sky);
SPECTRAL_SKY_EXPORT void spectral_sky_destroy(spectral_sky* sky);

#ifdef __cplusplus
}
#endif

#endif
