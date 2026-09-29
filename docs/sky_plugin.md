# Sky models and plugins

The sky is **tabulated once per scene** (lat-long table × output bands, plus an explicit sun disk
light) and importance-sampled; kernels never call the sky model, so CPU and GPU behave identically.

| Model | Notes |
|---|---|
| `simple` | Constant sky spectrum + sun disk (radiance or irradiance). Tests, quick previews. |
| `prague` | Prague Sky Model (Wilkie et al. 2021, Apache-2.0, vendored). Download a dataset: SWIR (547 MB, 280–2480 nm, ground level) for the default 380–1000 nm grid; visible-only datasets stop at 760 nm. Tables are cached in the cache directory keyed by all parameters. |
| `plugin` | Any shared library implementing the C ABI below (e.g. an encrypted in-house sky). |

Plugin ABI (`include/spectral/sky/sky_plugin.h`, version 1):
```c
int          spectral_sky_abi_version(void);                       /* return 1 */
spectral_sky* spectral_sky_create(const char* options_json);
int          spectral_sky_configure(spectral_sky*, double sun_x, double sun_y, double sun_z,
                                    double visibility_km, double ground_albedo, double altitude_m);
double       spectral_sky_radiance(const spectral_sky*, double dx, double dy, double dz, double lambda_nm);
double       spectral_sky_sun_radiance(const spectral_sky*, double lambda_nm);
double       spectral_sky_sun_half_angle(const spectral_sky*);
void         spectral_sky_destroy(spectral_sky*);
```
Directions are unit vectors in the renderer world frame (+Y up); radiance in W/(m² sr nm).
Scene JSON: `{"type": "sky", "model": "plugin", "library": "libmysky.so", "options": {...},
"sun": {"elevation_deg": 35, "azimuth_deg": 120}}`. A minimal example is
`tests/plugins/test_sky_plugin.c`. The Prague integration was compiled here but could not be run
against a dataset (download blocked in the development container); run the gated test with
`SPECTRAL_PRAGUE_DATASET=/path/PragueSkyModelDatasetSWIR.dat build/tests/spectral_tests`.
