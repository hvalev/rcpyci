import os

import numpy as np
import pandas as pd


def computeInfoVal2IFC(target_ci, rdata, iter=10000, force_gen_ref_dist=False):
    # RD: To suppress notes from R CMD CHECK, but this should not be necessary -- debug
    ref_seed = None
    ref_img_size = None
    ref_n_trials = None

    # Load parameter file (created when generating stimuli)
    df = pd.read_csv(rdata)
    seed = df['seed'].values[0]
    img_size = df['img_size'].values[0]
    n_trials = df['n_trials'].values[0]

    if not force_gen_ref_dist and not 'reference_norms' in globals():
        # Pre-computed reference distribution table
        ref_lookup = pd.DataFrame({
            'ref_seed': [1],
            'ref_img_size': [512],
            'ref_ref_iter': [10000],
            'ref_n_trials': [100],
            'ref_median': [1097.7394],
            'ref_mad': [52.54232],
        })

        ref_values = ref_lookup[
            (ref_lookup['ref_seed'] == seed) &
            (ref_lookup['ref_img_size'] == img_size) &
            (ref_lookup['ref_ref_iter'] == iter) &
            (ref_lookup['ref_n_trials'] == n_trials)
        ]

        if ref_values.shape[0] == 1:
            # We have a match, use the values
            print("Pre-computed reference values matching your exact parameters found.")
            ref_median = ref_values['ref_median'].values[0]
            ref_mad = ref_values['ref_mad'].values[0]
            ref_ref_iter = ref_values['ref_ref_iter'].values[0]
        else:
            # Check whether at least seed, img_size, and n_trials match
            ref_values = ref_lookup[
                (ref_lookup['ref_seed'] == seed) &
                (ref_lookup['ref_img_size'] == img_size) &
                (ref_lookup['ref_n_trials'] == n_trials)
            ]

            if ref_values.shape[0] > 0:
                print("I found pre-computed reference values that matched seed, image size, and number of trials, but not the number of reference distribution iterations.")
                max_ref_iter = ref_values['ref_ref_iter'].max()
                user_response = input(f"I did find pre-computed values for {max_ref_iter} iterations matching all other parameters. Do you want to use those instead? (yes/no): ")
                if user_response.lower() == 'yes':
                    print(f"Using pre-computed reference values for {max_ref_iter} instead of {iter} iterations.")
                    ref_values = ref_lookup[
                        (ref_lookup['ref_seed'] == seed) &
                        (ref_lookup['ref_img_size'] == img_size) &
                        (ref_lookup['ref_n_trials'] == n_trials) &
                        (ref_lookup['ref_ref_iter'] == max_ref_iter)
                    ]

                    ref_median = ref_values['ref_median'].values[0]
                    ref_mad = ref_values['ref_mad'].values[0]
                    ref_ref_iter = ref_values['ref_ref_iter'].values[0]
            else:
                ref_median = None
                ref_mad = None
                ref_ref_iter = None
    else:
        ref_median = None
        ref_mad = None
        ref_ref_iter = None

    if 'ref_median' not in globals():
        if 'reference_norms' not in globals():
            # Reference norms not present in rdata file, re-generate
            os.system(f"Rscript -e 'source(\"generateReferenceDistribution2IFC.R\"); generateReferenceDistribution2IFC(\"{rdata}\", iter={iter})'")

            # Re-load rdata file
            df = pd.read_csv(rdata)

            print("Note that now that this simulated reference distribution has been saved to the .Rdata file, the next time you call computeInfoVal2IFC(), it will not need to be computed again.")
        else:
            print("Using reference distribution found in rdata file.")

        reference_norms = df['reference_norms'].values
        ref_median = np.median(reference_norms)
        ref_mad = np.median(np.abs(reference_norms - ref_median))
        ref_ref_iter = len(reference_norms)

    # Compute informational value metric
    cinorm = np.linalg.norm(target_ci["ci"], 'fro')
    infoVal = (cinorm - ref_median) / ref_mad

    print(f"Informational value: z = {infoVal:.4f} (ci norm = {cinorm:.4f}; reference median = {ref_median:.4f}; MAD = {ref_mad:.4f}; iterations = {ref_ref_iter})")

    return infoVal
