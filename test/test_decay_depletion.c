/** Regression test for decay-depletion bookkeeping at the DarkAges interface. */

#include "class.h"

static int check_rate(struct injection * pin,
                      double z,
                      double expected,
                      const char * label) {
  double actual;

  if (injection_rate_DM_decay(pin,z,&actual) == _FAILURE_) {
    fprintf(stderr,"%s: injection_rate_DM_decay failed\n",label);
    return _FAILURE_;
  }

  if (fabs(actual-expected) > 1.e-12*MAX(1.,fabs(expected))) {
    fprintf(stderr,"%s: got %.17g, expected %.17g\n",label,actual,expected);
    return _FAILURE_;
  }

  return _SUCCESS_;
}

static int check_deposition_rate(struct injection * pin,
                                 double z,
                                 double expected,
                                 const char * label) {
  double actual;

  if (injection_rate_DM_decay_for_deposition(pin,z,&actual) == _FAILURE_) {
    fprintf(stderr,"%s: injection_rate_DM_decay_for_deposition failed\n",label);
    return _FAILURE_;
  }

  if (fabs(actual-expected) > 1.e-12*MAX(1.,fabs(expected))) {
    fprintf(stderr,"%s: got %.17g, expected %.17g\n",label,actual,expected);
    return _FAILURE_;
  }

  return _SUCCESS_;
}

int main(void) {
  struct injection in = {0};
  double reference_rate;
  double depleted_rate;

  in.rho_cdm = 10.;
  in.DM_decay_fraction = 0.2;
  in.DM_decay_Gamma = 0.5;
  in.t = 2.*log(2.)/in.DM_decay_Gamma;
  in.z_start_chi_approx = 2000.;
  in.DM_decay_table_max_z = 2999.;

  reference_rate = in.rho_cdm*in.DM_decay_fraction*in.DM_decay_Gamma;
  depleted_rate = reference_rate*0.25;

  in.f_eff_type = f_eff_on_the_spot;
  in.chi_type = no_factorization;
  in.DM_decay_table_uses_undepleted_rate = _FALSE_;
  if (check_rate(&in,100.,depleted_rate,"on-the-spot") == _FAILURE_) return _FAILURE_;

  in.f_eff_type = DarkAges;
  in.DM_decay_table_uses_undepleted_rate = _TRUE_;
  if (check_rate(&in,100.,depleted_rate,"physical rate with DarkAges") == _FAILURE_) return _FAILURE_;
  if (check_deposition_rate(&in,100.,reference_rate,"raw DarkAges table") == _FAILURE_) return _FAILURE_;
  if (check_deposition_rate(&in,2000.,reference_rate,"raw DarkAges table at boundary") == _FAILURE_) return _FAILURE_;
  if (check_deposition_rate(&in,2500.,depleted_rate,"high-z no-factorization") == _FAILURE_) return _FAILURE_;
  in.z_start_chi_approx = 4000.;
  if (check_deposition_rate(&in,3500.,depleted_rate,"beyond transfer-table range") == _FAILURE_) return _FAILURE_;
  in.z_start_chi_approx = 2000.;

  in.chi_type = chi_CK;
  if (check_deposition_rate(&in,100.,depleted_rate,"unsupported factorized path") == _FAILURE_) return _FAILURE_;

  in.f_eff_type = f_eff_from_file;
  if (check_rate(&in,100.,depleted_rate,"ordinary f_eff file") == _FAILURE_) return _FAILURE_;

  return _SUCCESS_;
}
