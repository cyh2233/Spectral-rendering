/* Test sky plugin: constant sky radiance 0.5, sun radiance 1000 * (lambda/500). */
#include <stdlib.h>
#include "spectral/sky/sky_plugin.h"

struct spectral_sky {
  double sun[3];
};

int spectral_sky_abi_version(void) { return SPECTRAL_SKY_ABI_VERSION; }
spectral_sky* spectral_sky_create(const char* options_json) {
  (void)options_json;
  return (spectral_sky*)calloc(1, sizeof(spectral_sky));
}
int spectral_sky_configure(spectral_sky* s, double x, double y, double z, double vis, double alb, double alt) {
  (void)vis; (void)alb; (void)alt;
  s->sun[0] = x; s->sun[1] = y; s->sun[2] = z;
  return 0;
}
double spectral_sky_radiance(const spectral_sky* s, double x, double y, double z, double l) {
  (void)s; (void)x; (void)z; (void)l;
  return y >= 0 ? 0.5 : 0.0;
}
double spectral_sky_sun_radiance(const spectral_sky* s, double l) { (void)s; return 1000.0 * l / 500.0; }
double spectral_sky_sun_half_angle(const spectral_sky* s) { (void)s; return 0.01; }
void spectral_sky_destroy(spectral_sky* s) { free(s); }
