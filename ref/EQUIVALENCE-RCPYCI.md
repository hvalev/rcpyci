# Comparing R's rcicr with the current `rcpyci` implementation

`test_ci_functions.py` only proves the 1:1 port (`pyrcicr_ref`) still matches R. This
adds the comparison that matters after the pipeline re-architecture: R against
`rcpyci` itself.

## Running it

Needs R, so it runs in the equivalence image, not in CI:

```bash
docker run --rm -v "$PWD/ref":/tests -v "$PWD":/repo:ro -w /tests \
  hvalev/rcpyci:latest /bin/bash /tests/run_rcpyci_vs_r.sh
```

## Three translation steps R's artifacts need before `rcpyci` will accept them

1. **Patch indices are 1-based in R.** `pyrcicr_ref` compensated with
   `p['patchIdx'].astype(int) - 1`; `rcpyci.core.__generate_individual_noise_stimulus`
   does not subtract, because its own `__generate_noise_pattern` counts from `idx = 0`.
   Feeding an `.Rdata`/`.npz` straight in raises `IndexError: index 4092 is out of
   bounds for axis 0 with size 4092`. Pass `patch_idx - 1`.
2. **`responses` must be a `(n_trials, 1)` column.** `utils.extract_sorted_responses_*`
   already reshape that way, but a flat `(n,)` array makes `stimuli * responses`
   broadcast `(5, 4092)` against `(5,)` and fail.
3. **Thresholding the z-map is the caller's job now.** `pipelines.compute_zscore_ci`
   returns the blurred z-score; the port's `process_quick_zmap` additionally blanks
   `|z| < threshold` and returns `{'blurred_ci', 'scaled_image', 'zmap'}`.

## Result on seed 1 / n_trials 5 / nscales 5 / sigma 3 / threshold 1

| comparison | max deviation |
| --- | --- |
| `pyrcicr_ref` vs R: ci / scaled | 4.2e-17 / 3.3e-16 |
| `pyrcicr_ref` vs R: base / combined | 0.0375 / 0.0188 |
| `pyrcicr_ref` vs R: zmap | 1.2e-4 |
| `rcpyci` vs R: ci / scaled | 4.2e-17 / 3.3e-16 |
| `rcpyci` vs R: combined / zmap | 0.0188 / 1.2e-4 |
| `rcpyci` vs `pyrcicr_ref`: ci / scaled / combined | **0 / 0 / 0** |

The numbers did not drift with the re-architecture: `rcpyci` reproduces the port exactly
and matches R to the same precision the port does. What changed is the calling
convention, not the arithmetic.

Note `base`/`combined`/`zmap` never agree exactly with R: the base face goes through a
JPEG/PNG round trip, and the R z-map's nan count differs from python's by a few pixels.
The existing suite already tolerates that (`< 0.04`, `equal_nan`).

## Pinning the result so it actually gates pushes

`ref/` can never run in CI, so the numbers above are frozen into
`tests/fixtures/rcicr_seed1.npz` (2.2 MB) and asserted by `tests/test_vs_rcicr_golden.py`
with `atol=1e-13`: four orders of magnitude above the 4e-17 floating point noise between
the two implementations, and five below anything a real algorithmic change produces.

`make_golden_fixtures.py` stores only R's output plus the small parameter arrays.
`patches` and `patch_idx` are left out on purpose - 125 MB each for a 512x512 image -
`compute_ci` regenerates them from `n_scales`/`noise_type` instead.

```bash
docker run --rm -v "$PWD/ref":/tests -v "$PWD":/repo -w /tests \
  hvalev/rcpyci:latest /bin/bash /tests/run_golden_fixtures.sh
```

The container keeps its scientific stack in `/pyrcicr/bin/python3`, not in `python3` on
`PATH`; the runner scripts use the absolute interpreter path.
