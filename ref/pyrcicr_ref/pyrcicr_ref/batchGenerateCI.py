
from joblib import Parallel, delayed


def batchGenerateCI(data, by, stimuli, responses, baseimage, rdata, save_as_png=True, targetpath='./cis', label='', antiCI=False, scaling='autoscale', constant=0.1):

    if scaling == 'autoscale':
        do_autoscale = True
        scaling = 'none'
    else:
        do_autoscale = False

    unique_units = data[by].unique()
    num_units = len(unique_units)

    def process_unit(unit):
        unit_data = data[data[by] == unit]
        if label == '':
            filename = f"{baseimage}_{by}_{unit_data.iloc[0][by]}"
        else:
            filename = f"{baseimage}_{label}_{by}_{unit_data.iloc[0][by]}"

        unit_cis = generateCI(
            stimuli=unit_data[stimuli],
            responses=unit_data[responses],
            baseimage=baseimage,
            rdata=rdata,
            save_as_png=save_as_png,
            filename=filename,
            targetpath=targetpath,
            antiCI=antiCI,
            scaling=scaling,
            scaling_constant=constant,
            participants=None
        )

        return (filename, unit_cis)

    with Parallel(n_jobs=-1, verbose=10) as parallel:
        result = parallel(delayed(process_unit)(unit) for unit in unique_units)

    cis = {filename: unit_cis for filename, unit_cis in result}

    if do_autoscale:
        cis = autoscale(cis, save_as_pngs=save_as_png, targetpath=targetpath)

    return cis
