"""Regenerate tests/fixtures/rcicr_seed1.npz from R's rcicr.

Runs inside the equivalence container (see run_rcpyci_vs_r.sh for the invocation), after
test_generate_functions.py has produced a matching .Rdata/.npz stimulus pair.

The fixture deliberately holds only what is small and non-derivable:

  r_ci            R's classification image, float64 (512, 512) - the golden numbers
  stimuli_params  the selected + trimmed (n_trials, 4092) parameter rows R used
  responses       (n_trials, 1) column, the shape rcpyci's callers pass
  meta            versions and settings R was called with

patches / patch_idx are NOT stored: they are 125 MB for a 512x512 image, and
rcpyci.__generate_noise_pattern reproduces them deterministically (the container suite
checks the match at 2.0e-15 for patches and exactly for patch_idx).
"""

import glob
import os
import sys

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

STIM_DIR = "/tests/stimuli"
BASE_LABEL = "aName"
STIMULI_R = [1, 2, 3, 4, 5]
STIMULI_PY = [0, 1, 2, 3, 4]
RESPONSES = [1, 1, 1, 1, 1]
OUT = "/repo/tests/fixtures/rcicr_seed1.npz"

package = importr("rcicr")


def newest_pair():
    def token(path):
        return os.path.basename(path).split("_time_")[1].rsplit(".", 1)[0]

    rdatas = {token(p): p for p in glob.glob(f"{STIM_DIR}/*.Rdata")}
    npzs = {token(p): p for p in glob.glob(f"{STIM_DIR}/*.npz")}
    shared = sorted(set(rdatas) & set(npzs))
    if not shared:
        raise SystemExit("no matching .Rdata/.npz pair - run test_generate_functions.py first")
    return rdatas[shared[-1]], npzs[shared[-1]]


rdata_file, npz_file = newest_pair()
print(f"  pair: {os.path.basename(rdata_file)}")

result_r = np.array(
    robjects.conversion.rpy2py(
        package.generateCI(
            stimuli=robjects.IntVector(STIMULI_R),
            responses=robjects.IntVector(RESPONSES),
            baseimage=BASE_LABEL,
            antiCI=False,
            sigma=3,
            zmap=False,
            rdata=rdata_file,
            rdatapath=True,
        )
    )
)

npz = np.load(npz_file)
params = npz[f"stimuli_params_{BASE_LABEL}"][np.array(STIMULI_PY)]
if params.shape[1] == 4096:
    params = params[:, :4092]

rcicr_version = robjects.conversion.rpy2py(robjects.r("as.character(packageVersion('rcicr'))"))[0]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(
    OUT,
    r_ci=result_r[0].astype(np.float64),
    stimuli_params=params.astype(np.float64),
    responses=np.array(RESPONSES, dtype=np.float64).reshape(-1, 1),
    meta=np.array(
        {
            "source": "R rcicr generateCI()",
            "rcicr_version": str(rcicr_version),
            "r_base_image_label": BASE_LABEL,
            "img_size": str(result_r.shape[1]),
            "n_scales": "5",
            "noise_type": "sinusoid",
            "gabor_sigma": "25",  # unused by the sinusoid noise pattern
            "patch_indices": "R stimuli are 1-based; rcpyci's generator counts from 0",
        }
    ),
)
size_mb = os.path.getsize(OUT) / 1e6
print(f"  wrote {OUT} ({size_mb:.2f} MB) rcicr {rcicr_version}, r_ci {result_r[0].shape}")
