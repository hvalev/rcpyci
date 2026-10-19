#!/bin/bash
# Drive the experiment inside the R container: build an env with the versions rcpyci's
# lock resolves for Python 3.11, generate the stimulus pair, then compare R vs ref vs rcpyci.
export R_HOME=/usr/local/lib/R
export LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/lib/R/modules
cd /tests || exit 1

python3 -m venv /up >/dev/null && /up/bin/pip -q install --upgrade pip >/dev/null 2>&1
/up/bin/pip -q install "numpy==2.4.6" "scipy==1.17.1" "pandas==2.3.3" "pillow==12.3.0" \
  "tqdm==4.70.1" joblib matplotlib "rpy2[all]==3.5.15" >/dev/null 2>&1
cp -a /repo /repo_rw && /up/bin/pip -q install --no-deps /repo_rw >/dev/null 2>&1
echo "  env: $(/up/bin/python3 -c "import numpy,scipy,rcpyci,joblib;from importlib.metadata import version as V;print('numpy',numpy.__version__,'scipy',scipy.__version__,'| rcpyci',V('rcpyci'))" 2>&1 | tail -1)"

echo "════ step 1: generate suite creates the paired .Rdata/.npz (R vs pyrcicr_ref) ════"
out=$(/pyrcicr/bin/python3 test_generate_functions.py 2>&1); echo "  exit=$?"
echo "$out" | grep -iE "identical|deviation" | sed 's/^/       /' | tail -4

echo "════ step 2: R vs pyrcicr_ref vs rcpyci ════"
out=$(timeout 900 /up/bin/python3 test_rcpyci_vs_r.py 2>&1); rc=$?
echo "$out" | grep -viE "ScriptRunContext|^Loading required|^ *$" | sed 's/^/  /'
echo "  ----> experiment exit: $rc"
