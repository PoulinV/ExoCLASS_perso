#!/usr/bin/env python3
"""Synthetic allocation-ownership regression tests, without a CLASS build.

Run: python3 test/test_distortions_cleanup.py
Requires a C compiler. Extracts the actual two C functions and uses actual
headers. Redshifts below the output cutoff exercise allocation and cleanup
without calling physics modules. This does not test initialization failures
or numerical spectra. All fixtures are synthetic and compilation is temporary.
"""

from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREFIX = r'''
#include <stdlib.h>
#include <stdio.h>
#include "distortions.h"

static void *allocations[32];
static size_t sizes[32];
static size_t outstanding;
static void *tracked_malloc(size_t size) {
  void *pointer = malloc(size);
  if (pointer == NULL) abort();
  for (int i = 0; i < 32; ++i) {
    if (allocations[i] == NULL) {
      allocations[i] = pointer;
      sizes[i] = size;
      outstanding += size;
      return pointer;
    }
  }
  abort();
}
static void tracked_free(void *pointer) {
  if (pointer == NULL) return;
  for (int i = 0; i < 32; ++i) {
    if (allocations[i] == pointer) {
      outstanding -= sizes[i];
      allocations[i] = NULL;
      free(pointer);
      return;
    }
  }
  abort(); /* An unowned or already freed pointer is a regression. */
}
#define malloc tracked_malloc
#define free tracked_free

/* No physics calls should be reached for z < z_output_sd. */
#define background_tau_of_z(...) (abort(), _FAILURE_)
#define background_at_tau(...) (abort(), _FAILURE_)
#define thermodynamics_at_z(...) (abort(), _FAILURE_)
#define injection_deposition_at_z(...) (abort(), _FAILURE_)
#define noninjection_photon_heating_at_z(...) (abort(), _FAILURE_)
#define noninjection_init(...) _SUCCESS_
#define noninjection_free(...) _SUCCESS_
'''
SUFFIX = r'''
int main(void) {
  for (int sm = 0; sm <= 1; ++sm) {
    for (int exact_y = 0; exact_y <= 1; ++exact_y) {
      for (int exotic = 0; exotic <= 1; ++exotic) {
        for (int repeat = 0; repeat < 32; ++repeat) {
          struct precision pr = {0};
          struct background ba = {0};
          struct thermodynamics th = {0};
          struct perturbations pt = {0};
          struct primordial pm = {0};
          struct distortions sd = {0};
          ba.bg_size = 16;
          th.th_size = 32;
          sd.has_distortions = _TRUE_;
          sd.include_only_exotic = exotic;
          sd.include_DH_SMresidual_distortions = sm;
          sd.exact_y = exact_y;
          sd.z_size = 1;
          sd.z_output_sd = 1.;
          sd.z = malloc(sizeof(double));
          sd.z[0] = 0.;
          if (sm) sd.DH_SMresiduals_table = malloc(3 * 2000 * sizeof(double));
          if (distortions_compute_heating_rate(&pr, &ba, &th, &pt, &pm, &sd)
              != _SUCCESS_) return 2;
          if (sd.dQrho_dz_tot[0] != 0.) return 3;
          if (exact_y && sd.exact_integrand_y[0] != 0.) return 4;
          if (distortions_free(&sd) != _SUCCESS_) return 5;
          if (outstanding != 0) {
            fprintf(stderr, "sm=%d exact_y=%d exotic=%d: %zu bytes retained\n",
                    sm, exact_y, exotic, outstanding);
            return 1;
          }
        }
      }
    }
  }
  puts("256 synthetic heating/cleanup cycles: zero outstanding allocations");
  return 0;
}
'''


class DistortionsCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "source/distortions.c").read_text()
        functions = []
        for name in ("distortions_free", "distortions_compute_heating_rate"):
            match = re.search(r"^int " + name + r"\([^;]*?\{.*?^\}", source, re.M | re.S)
            if not match:
                raise RuntimeError("Cannot extract function: " + name)
            functions.append(match.group())
        cls.functions = "\n".join(functions)

    def compile_and_run(self, functions):
        with tempfile.TemporaryDirectory(prefix="class-cleanup-test-") as tmp:
            cfile = Path(tmp) / "test.c"
            binary = Path(tmp) / "test"
            cfile.write_text(PREFIX + functions + SUFFIX)
            command = ["cc", "-std=gnu99", "-O0"]
            for directory in ("include", "external/heating", "external/HyRec2020",
                              "external/RecfastCLASS"):
                command.extend(["-I", str(ROOT / directory)])
            compiled = subprocess.run(command + [str(cfile), str(ROOT / "tools/common.c"),
                                                 "-lm", "-o", str(binary)],
                                      capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            return subprocess.run([str(binary)], capture_output=True, text=True)

    def test_successful_cleanup_repeated(self):
        result = self.compile_and_run(self.functions)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_detects_missing_sm_table_free(self):
        mutant = self.functions.replace("free(psd->DH_SMresiduals_table);", "")
        self.assertNotEqual(mutant, self.functions)
        result = self.compile_and_run(mutant)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("48000 bytes retained", result.stderr)

    def test_detects_missing_thermo_workspace_free(self):
        mutant = self.functions.replace("free(pvecthermo);", "")
        self.assertNotEqual(mutant, self.functions)
        result = self.compile_and_run(mutant)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("256 bytes retained", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
