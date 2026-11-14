#!/usr/bin/env bash
# Run all verification scripts against the shipped frozen data.
# Each script prints PASS/FAIL per check and exits non-zero on failure.
set -u
cd "$(dirname "$0")"
fail=0
for s in verify_webgen_maintable verify_webgen_judge_ablation verify_webgen_noise \
         verify_webgen_bon verify_webgen_cost verify_v2w_evolution \
         verify_v2w_routing verify_v2w_gan_rounds verify_i2c_case verify_revision_checks; do
  echo "================================================================"
  echo "### $s"
  echo "================================================================"
  python3 "$s.py" || fail=1
  echo
done
if [ "$fail" -eq 0 ]; then echo "== ALL VERIFICATION SCRIPTS PASSED =="; else echo "== SOME CHECKS FAILED =="; fi
exit $fail
