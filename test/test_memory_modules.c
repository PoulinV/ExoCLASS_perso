/* Native repeated-init/cleanup diagnostic; no Python, likelihood, or network.
 * Usage: memory_modules LAST_MODULE REPEATS
 * Modules: 0=input, 1=background, 2=thermodynamics, 3=perturbations,
 * 4=primordial, 5=fourier, 6=transfer, 7=harmonic, 8=lensing, 9=distortions.
 * A partial-module run deliberately follows classy's cleanup ordering;
 * input allocations owned by later modules may remain in such a control.
 */
#include "class.h"
#ifdef __APPLE__
#include <mach/mach.h>
#endif

static double resident_kib(void) {
#ifdef __APPLE__
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                (task_info_t)&info, &count) != KERN_SUCCESS) return -1;
  return info.resident_size / 1024.;
#else
  return -1.;
#endif
}

#define CHECK(call, error) do { \
  if ((call) == _FAILURE_) { fprintf(stderr, "%s\n", error); return 1; } \
} while (0)

int main(int argc, char **argv) {
  int level = argc > 1 ? atoi(argv[1]) : 9;
  int repeats = argc > 2 ? atoi(argv[2]) : 5;
  if (level < 0 || level > 9 || repeats < 1 || repeats > 200) return 2;
  struct precision pr = {0};
  struct background ba = {0};
  struct thermodynamics th = {0};
  struct perturbations pt = {0};
  struct primordial pm = {0};
  struct fourier fo = {0};
  struct transfer tr = {0};
  struct harmonic hr = {0};
  struct lensing le = {0};
  struct distortions sd = {0};
  struct output op = {0};
  struct file_content fc = {0};
  ErrorMsg errmsg;
  const char *names[] = {"output", "l_max_scalars", "lensing",
                        "sd_branching_approx", "sd_only_exotic", "exact_y",
                        "h", "omega_b", "omega_cdm"};
  const char *values[] = {"tCl,pCl,lCl,sd", "300", "yes", "sharp_sharp",
                         "yes", "yes", "0.675", "0.0224", "0.12"};
  CHECK(parser_init(&fc, 9, "memory-test", errmsg), errmsg);
  for (int i = 0; i < 9; ++i) {
    strcpy(fc.name[i], names[i]);
    strcpy(fc.value[i], values[i]);
  }
  for (int i = 0; i < repeats; ++i) {
    CHECK(input_read_from_file(&fc, &pr, &ba, &th, &pt, &tr, &pm,
                               &hr, &fo, &le, &sd, &op, errmsg), errmsg);
    if (level >= 1) CHECK(background_init(&pr, &ba), ba.error_message);
    if (level >= 2) CHECK(thermodynamics_init(&pr, &ba, &th), th.error_message);
    if (level >= 3) CHECK(perturbations_init(&pr, &ba, &th, &pt), pt.error_message);
    if (level >= 4) CHECK(primordial_init(&pr, &pt, &pm), pm.error_message);
    if (level >= 5) CHECK(fourier_init(&pr, &ba, &th, &pt, &pm, &fo), fo.error_message);
    if (level >= 6) CHECK(transfer_init(&pr, &ba, &th, &pt, &fo, &tr), tr.error_message);
    if (level >= 7) CHECK(harmonic_init(&pr, &ba, &pt, &pm, &fo, &tr, &hr), hr.error_message);
    if (level >= 8) CHECK(lensing_init(&pr, &pt, &hr, &fo, &le), le.error_message);
    if (level >= 9) CHECK(distortions_init(&pr, &ba, &th, &pt, &pm, &sd), sd.error_message);
    if (i == 0 && level >= 6) {
      size_t sample = ((size_t)tr.index_tt_t0 * (size_t)tr.l_size[0] + 3) * tr.q_size + 10;
      printf("FINGERPRINT q_size=%zu q10=%.17g transfer_sample=%.17g",
             tr.q_size, tr.q[10], tr.transfer[0][sample]);
      if (level >= 9) {
        printf(" sd_x_size=%d sd_di10=%.17g", sd.x_size, sd.DI[10]);
      }
      puts("");
    }
    if (level >= 9) CHECK(distortions_free(&sd), sd.error_message);
    if (level >= 8) CHECK(lensing_free(&le), le.error_message);
    if (level >= 7) CHECK(harmonic_free(&hr), hr.error_message);
    if (level >= 6) CHECK(transfer_free(&tr), tr.error_message);
    if (level >= 5) CHECK(fourier_free(&fo), fo.error_message);
    if (level >= 4) CHECK(primordial_free(&pm), pm.error_message);
    if (level >= 3) CHECK(perturbations_free(&pt), pt.error_message);
    if (level >= 2) CHECK(thermodynamics_free(&th), th.error_message);
    if (level >= 1) CHECK(background_free(&ba), ba.error_message);
    printf("MEMORY level=%d iteration=%d rss_KiB=%.0f\n", level, i, resident_kib());
    fflush(stdout);
  }
  CHECK(parser_free(&fc), errmsg);
  return 0;
}
