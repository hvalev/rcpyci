from scipy.stats import pearsonr


def computeCumulativeCICorrelation(stimuli, responses, baseimage, rdata, targetci=None, step=1):
    # Load parameter file (created when generating stimuli)
    with open(rdata, 'rb') as f:
        data = pickle.load(f)
    
    if 's' not in data and 'p' not in data:
        raise ValueError("File specified in rdata argument did not contain 's' or 'p' variable.")
    
    p = {'patches': data['s']['sinusoids'], 'patchIdx': data['s']['sinIdx'], 'noise_type': 'sinusoid'}
    
    # Get base image
    base_faces = data['base_faces']
    if baseimage not in base_faces:
        raise ValueError(f"File specified in rdata argument did not contain any reference to base image label: {baseimage} (NOTE: file contains references to the following base image label(s): {', '.join(base_faces.keys())})")
    
    base = base_faces[baseimage]

    # Retrieve parameters of actually presented stimuli
    stimuli_params = data['stimuli_params']
    params = stimuli_params[baseimage][stimuli]

    if not params:
        raise ValueError(f"No parameters found for base image: {base}")
    
    # Compute final classification image if necessary
    if not targetci:
        finalCI = generateCINoise(params, responses, p)
    else:
        finalCI = targetci['ci']

    # Compute correlations with final CI with cumulative CI
    correlations = []

    for trial in range(0, len(responses), step):
        cumCI = generateCINoise(params[:trial+1], responses[:trial+1], p)
        corr, _ = pearsonr(cumCI.flatten(), finalCI.flatten())
        correlations.append(corr)
    
    return correlations
