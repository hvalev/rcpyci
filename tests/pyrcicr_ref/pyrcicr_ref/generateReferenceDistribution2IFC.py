import os
import pickle

import numpy as np


def generateReferenceDistribution2IFC(rdata, iter=10000):
    # Load parameter file (created when generating stimuli)
    with open(rdata, 'rb') as f:
        data = pickle.load(f)

    # Re-generate stimuli based on rdata parameters in matrix form
    base_face_files = data['base_face_files']
    n_trials = data['n_trials']
    img_size = data['img_size']
    seed = data['seed']
    noise_type = data['noise_type']
    
    print("Re-generating stimuli based on rdata file, please wait...")
    stimuli = generateStimuli2IFC(base_face_files, n_trials, img_size, seed=seed, noise_type=noise_type, return_as_dataframe=True, save_as_png=False, save_rdata=False)

    # Simulate random responding in 2IFC task with ntrials trials across iter iterations
    print("Computing reference distribution, please wait...")

    if iter < 10000:
        print("You should set iter >= 10000 for InfoVal statistic to be reliable")
    
    # Run simulation
    reference_norms = np.zeros(iter)

    for i in range(iter):
        print(f"Simulation iteration {i+1}/{iter}")

        # Generate random responses for this iteration
        responses = np.random.choice([-1, 1], size=n_trials)

        # Compute classification image for this iteration
        ci = np.dot(stimuli, responses) / stimuli.shape[1]

        # Save norm for this iteration
        reference_norms[i] = np.linalg.norm(ci, 'fro')
    
    # Save reference norms to rdata file
    print("Saving simulated reference distribution to rdata file...")
    del stimuli, responses
    data['reference_norms'] = reference_norms
    with open(rdata, 'wb') as f:
        pickle.dump(data, f)

def generateStimuli2IFC(base_face_files, n_trials, img_size, seed, noise_type, return_as_dataframe=False, save_as_png=False, save_rdata=False):
    # Implement your generateStimuli2IFC function here
    return
# Call generateReferenceDistribution2IFC with your parameters
# rdata_file = 'your_rdata_file.rdata'
# generateReferenceDistribution2IFC(rdata_file, iter=10000)
