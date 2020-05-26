import os
import shutil
import tempfile



from joblib import load, dump, Parallel, delayed
from joblib import Parallel, delayed
import numpy as np

from .simulateLogNormal import simulateLogNormal
from .power_analysis_workers import anova_oneway_simulation, plsda_simulation
import warnings


def power_analysis(data, effect_size, sample_size, alpha=0.05, model='ANOVA', simmodel='lognormal',
                   fakedata_size=5000, n_repeats=10, variables_to_calculate=None, n_jobs=-1, **kwargs):
    """

    :param data:
    :param effect_size:
    :param sample_size:
    :param alpha:
    :param model:
    :param simmodel:
    :param fakedata_size:
    :param n_repeats:
    :param variables_to_calculate:
    :param n_jobs:
    :param kwargs:
    :return:
    """
    try:

        # Check if the maximum sample sizes not exceed the requested number of simulated samples
        # Add input for simulated Data generation algorithm?
        if 2 * max(sample_size) >= fakedata_size:
            fakedata_size = max(sample_size) + 500

        n_vars = data.shape[1]
        # Generate the simulated data

        ##Simulation of a new data set based on multivariate normal distribution
        # add option here
        simulated_data, correlation_matrix = simulateLogNormal(data, 'Estimate', fakedata_size, **kwargs)


        # Generate a shared memory array to avoid duplicating the simulated data
        temp_folder = tempfile.mkdtemp()
        data_fname = os.path.join(temp_folder, 'simdata_mmap.mmap')

        if os.path.exists(data_fname):
            os.unlink(data_fname)
        dump(simulated_data, data_fname)
        simdata_memmap = load(data_fname, mmap_mode='r+')

        if variables_to_calculate is None:
            variables_to_calculate = range(n_vars)

        # Run the simulation in parallel - each worker will handle 1 variable
        if model == 'ANOVA':
            output = Parallel(n_jobs=n_jobs, verbose=10)(delayed(anova_oneway_simulation)(data=simdata_memmap,
                                                                                          variables=variable,
                                                                                          effect_size=effect_size,
                                                                                          sample_size=sample_size,
                                                                                          alpha=alpha, n_repeats=n_repeats,
                                                                                          modification_type='correlation',
                                                                                          weight_values=correlation_matrix[:, variable],
                                                                                          weight_threshold=0.8, **kwargs) for variable in variables_to_calculate)
        elif model == 'PLS-DA':
            output = Parallel(n_jobs=n_jobs, verbose=10)(delayed(plsda_simulation)(data=simdata_memmap,
                                                                                variables=variable,
                                                                                effect_size=effect_size,
                                                                                sample_size=sample_size,
                                                                                alpha=alpha, n_repeats=n_repeats,
                                                                                modification_type='correlation',
                                                                                weight_values=correlation_matrix[:, variable],
                                                                                weight_threshold=0.8, **kwargs) for variable in variables_to_calculate)
    # Remove the temporary directory used to
        # store the memmaps
        try:
            shutil.rmtree(temp_folder)
        except OSError:
            pass

        return output

    except TypeError as terp:
        raise terp
    except ValueError as verr:
        raise verr
    except Exception as exp:
        raise exp
