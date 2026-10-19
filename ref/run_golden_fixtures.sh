#!/bin/bash
# Regenerate tests/fixtures/rcicr_seed1.npz. Run from the repo root:
#   docker run --rm -v "$PWD/ref":/tests -v "$PWD":/repo -w /tests \
#     hvalev/rcpyci:latest /bin/bash /tests/run_golden_fixtures.sh
export R_HOME=/usr/local/lib/R
export LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/lib/R/modules
cd /tests || exit 1
PY=/pyrcicr/bin/python3   # the image ships numpy in this venv, not in system python

echo "════ generating the stimulus pair (R + pyrcicr_ref) ════"
"$PY" test_generate_functions.py >/tmp/gen.log 2>&1 || { echo "  generate FAILED"; tail -20 /tmp/gen.log; exit 1; }
echo "  ok"

echo "════ writing the golden fixture ════"
"$PY" make_golden_fixtures.py || exit 1

chown -R "$(stat -c %u:%g /repo)" /repo/tests/fixtures 2>/dev/null
rm -rf /tests/stimuli /tests/cis /tests/zmaps
