"""EXPERIMENT: equivalence between R's rcicr and the CURRENT rcpyci implementation.

The existing suite (test_ci_functions.py) compares R against pyrcicr_ref, the 1:1 port
that was used for the first python implementation. This script runs the same R call and
feeds the identical inputs to rcpyci's own API, so the drift introduced by the pipeline
re-architecture becomes visible.

Run order matters: test_generate_functions.py must run first in the same container, it
writes the paired .Rdata/.npz stimulus files this script consumes.
"""

import glob
import os
import sys

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

sys.path.append("/tests/pyrcicr_ref/pyrcicr_ref/")

package = importr("rcicr")

STIM_DIR = "/tests/stimuli"
BASE_LABEL = "aName"
STIMULI_R = [1, 2, 3, 4, 5]  # R is 1-based
STIMULI_PY = [0, 1, 2, 3, 4]
RESPONSES = [1, 1, 1, 1, 1]


def newest_pair():
    """Pick the newest .Rdata/.npz pair that belongs to the SAME run.

    The original tests take glob(...)[0], which pairs leftovers from different runs.
    """

    def token(path):
        return os.path.basename(path).split("_time_")[1].rsplit(".", 1)[0]

    rdatas = {token(p): p for p in glob.glob(f"{STIM_DIR}/*.Rdata")}
    npzs = {token(p): p for p in glob.glob(f"{STIM_DIR}/*.npz")}
    shared = sorted(set(rdatas) & set(npzs))
    if not shared:
        raise SystemExit("no matching .Rdata/.npz pair found - run test_generate_functions.py first")
    last = shared[-1]
    return rdatas[last], npzs[last]


def report(label, a, b, tol=None):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        print(f"  {label:<34} SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    dev = float(np.nanmax(np.abs(a - b)))
    ok = np.allclose(a, b, equal_nan=True) if tol is None else dev < tol
    print(f"  {label:<34} max dev {dev:<12.6g} {'OK' if ok else 'DIVERGED'}")
    return ok


print("════ inputs ════")
rdata_file, npz_file = newest_pair()
print(f"  rdata: {os.path.basename(rdata_file)}")
print(f"  npz:   {os.path.basename(npz_file)}")

npz = np.load(npz_file)
patches = npz["patches"]
# R stores patch indices 1-based; pyrcicr_ref compensated with `astype(int) - 1`.
# rcpyci's own __generate_noise_pattern counts from 0 (core.py `idx = 0`), so R artifacts
# have to be shifted before they can be handed to rcpyci.
patch_idx = npz["patchIdx"] - 1
base = npz[f"base_faces_{BASE_LABEL}"]
stimuli_params = npz[f"stimuli_params_{BASE_LABEL}"]
params = stimuli_params[np.array(STIMULI_PY)]
if params.shape[1] == 4096:
    params = params[:, :4092]
print(f"  base {base.shape} | patches {patches.shape} | patch_idx {patch_idx.shape} | params {params.shape}")

print("\n════ R reference call ════")
result_r = np.array(
    robjects.conversion.rpy2py(
        package.generateCI(
            stimuli=robjects.IntVector(STIMULI_R),
            responses=robjects.IntVector(RESPONSES),
            baseimage=BASE_LABEL,
            antiCI=False,
            sigma=3,
            zmap=True,
            zmapmethod="quick",
            threshold=1,
            rdata=rdata_file,
            rdatapath=True,
        )
    )
)
print(f"  result_r {result_r.shape}  [ci, scaled, base, combined, zmap]")

print("\n════ pyrcicr_ref (1:1 port, what the old suite compared against) ════")
from generateCI import generate_CI  # noqa: E402

ref = generate_CI(
    stimuli=STIMULI_PY,
    responses=RESPONSES,
    baseimage=BASE_LABEL,
    anti_CI=False,
    sigma=3,
    zmap=True,
    zmapmethod="quick",
    threshold=1,
    rdata=npz_file,
)
ok_ref = [
    report("ref 'ci' vs R", result_r[0], ref["ci"]),
    report("ref 'scaled' vs R", result_r[1], ref["scaled"]),
    report("ref 'base' vs R", result_r[2], ref["base"], tol=0.04),
    report("ref 'combined' vs R", result_r[3], ref["combined"], tol=0.04),
    report("ref 'zmap' vs R", result_r[4], ref["zmap"]["zmap"], tol=1e-3),
]

print("\n════ rcpyci (current architecture) ════")
from rcpyci.core import compute_ci  # noqa: E402
from rcpyci.im_ops import apply_independent_scaling, combine  # noqa: E402
from rcpyci.pipelines import compute_zscore_ci  # noqa: E402

# real callers (utils.extract_sorted_responses_*) hand core a (n_trials, 1) column
resp = np.array(RESPONSES).reshape(-1, 1)
ci = compute_ci(
    base_image=base,
    responses=resp,
    stimuli_params=params,
    patches=patches,
    patch_idx=patch_idx,
    anti_ci=False,
)
scaled = apply_independent_scaling(ci)
combined = combine(base, scaled)
zmap = compute_zscore_ci(ci, sigma=3)["zscore_image"]
zmap = np.where((zmap > -1) & (zmap < 1), np.nan, zmap)  # R's zmap is thresholded at +/- threshold

ok_rc = [
    report("rcpyci ci vs R", result_r[0], ci),
    report("rcpyci scaled vs R", result_r[1], scaled),
    report("rcpyci combined vs R", result_r[3], combined, tol=0.04),
    report("rcpyci zmap vs R", result_r[4], zmap, tol=1e-3),
]

print("\n════ rcpyci vs pyrcicr_ref (python-to-python drift) ════")
ok_pp = [
    report("rcpyci ci vs ref ci", ref["ci"], ci),
    report("rcpyci scaled vs ref scaled", ref["scaled"], scaled),
    report("rcpyci combined vs ref", ref["combined"], combined),
]

print(f"\n==> ref vs R: {sum(ok_ref)}/{len(ok_ref)} | rcpyci vs R: {sum(ok_rc)}/{len(ok_rc)} | rcpyci vs ref: {sum(ok_pp)}/{len(ok_pp)}")
sys.exit(0 if all(ok_ref) and all(ok_rc) else 1)
