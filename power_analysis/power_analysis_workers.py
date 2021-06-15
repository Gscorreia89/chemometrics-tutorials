from copy import deepcopy

import numpy as np
import scipy.stats as scistats
from statsmodels.stats.multitest import multipletests
from pyChemometrics.ChemometricsPLSDA import ChemometricsPLSDA

from .simulateLogNormal import simulateLogNormal
from .simulateEffect import effect_cohen_d
from .scoreResults import score_confusionmetrics, score_metrics, score_classification_metrics

"""
Code templates for each type of simulation, containing data selection, effect size addition or outcome simulation, 
analysis with each specific model and scoring, for specific effect and sample size combinations.
"""


def anova_oneway_simulation(data, variables, effect_size, sample_size, alpha=0.05, n_repeats=15, weight_values=None,
                             weight_threshold=0.8, modification_type='correlation', class_balance=0.5,
                             multiple_testing_correction='fdr_bh'):
    """
    Worker function to perform power calculations for a one-way ANOVA model, with effect size added parametrized
    using Cohen's d measure.

    :param numpy.ndarray data: X data matrix (real or simulated) to use in th
    :param int, float or numpy.ndarray variables: List of variables to modify. In case of an `int` value or numpy.ndarray with dtype=`int` only variable
    with If a single `Float` value is provided interpreted as a proportion will all be modified by their effect size
    :param numpy.ndarray effect_size: array with effect size values to test
    :param numpy.ndarray sample_size: array with sample sizes to test
    :param float alpha:
    :param int n_repeats:
    :param numpy.ndarray weight: Can be
    :param numpy.ndarray weight_threshold: Used in all modification methods invol
    :param str modification_type: How to mo. Single means only the variables requested are modified. Proportion means
    that a set of
    :param float class_balance:
    :return:
    """

    try:
        import warnings
        warnings.filterwarnings('ignore')
        if modification_type not in ['correlation', 'manual', 'proportion', 'correlation_weighted']:
            raise ValueError("modification_type argument not supported")
        if modification_type == 'proportion' and not isinstance(variables, float):
            raise TypeError("When using \'proportion\' as modification_type \'variables\' must be a float")

        # get the list of metrics calculated in scoreResults and update
        results = dict.fromkeys(score_metrics)
        for key in results.keys():
            results[key] = np.zeros((effect_size.size, sample_size.size, n_repeats))

        if multiple_testing_correction is not None:
            adjusted_results = dict.fromkeys(score_metrics)
            for key in adjusted_results.keys():
                adjusted_results[key] = np.zeros((effect_size.size, sample_size.size, n_repeats))
            adjusted_results['method'] = multiple_testing_correction

        n_vars = data.shape[1]
        # Loop over effect size, sample size and finally each monte carlo repeat
        for eff_idx, curr_effect in np.ndenumerate(effect_size):
            for ssize_idx, curr_ssize in np.ndenumerate(sample_size):
                for rep_idx in range(n_repeats):
                    # Select samples to use
                    ## Select a subset of the simulated spectra
                    mod_data = np.copy(data[np.random.choice(data.shape[0], curr_ssize, replace=False), :])
                    # if any option other than proportion
                    if modification_type != 'proportion':
                        # Modify only variables above a certain threshold of correlation
                        var_to_mod = np.zeros(n_vars, dtype='int')
                        var_to_mod[variables] = 1

                        expected_hits = np.zeros(n_vars, dtype='int')
                        expected_hits[var_to_mod == 1] = 1
                        # If correlation and correlation_weighted
                        if weight_values is not None and modification_type in ["correlation", "correlation_weighted"]:
                            if weight_values.ndim == 1:
                                var_to_mod |= abs(weight_values) >= weight_threshold
                            else:
                                var_to_mod |= np.any(abs(weight_values) >= weight_threshold, axis=1)

                        expected_hits = var_to_mod
                    # Select a subset of samples to add the effect on
                    which_samples = np.random.choice(range(curr_ssize), int(np.floor(class_balance * curr_ssize)),
                                                     replace=False)

                    if modification_type == 'correlation_weighted':
                        mod_data = effect_cohen_d(mod_data, curr_effect, which_vars=var_to_mod,
                                                  which_samples=which_samples, standardized=True,
                                                  noise=0, weight=weight_values)
                    else:
                        mod_data = effect_cohen_d(mod_data, curr_effect, which_vars=var_to_mod,
                                                  which_samples=which_samples, standardized=True,
                                                  noise=0, weight=None)

                    # Would it be possible to pass a model selection criteria?
                    # P-values for the one-way ANOVA
                    pvals = scistats.f_oneway(np.delete(mod_data, which_samples, axis=0),
                                              mod_data[which_samples, :])[1]

                    if modification_type == 'correlation_weighted':
                        scored_res = score_confusionmetrics(result_vector=pvals, expected_hits=expected_hits,
                                               weight_vector=weight_values,
                                               alpha=alpha)
                    else:
                        scored_res = score_confusionmetrics(result_vector=pvals, expected_hits=expected_hits,
                                               weight_vector=None,
                                               alpha=alpha)

                    for key in scored_res.keys():
                        results[key][eff_idx, ssize_idx, rep_idx] = scored_res[key]
                    # Would it be possible to pass a model selection criteria?
                    # P-values for the one-way ANOVA
                    if multiple_testing_correction is not None:
                        adjusted_pvalues = multipletests(pvals, alpha=0.05, method=multiple_testing_correction)[1]

                        scored_res = score_confusionmetrics(result_vector=adjusted_pvalues, expected_hits=expected_hits,
                                                weight_vector=None,
                                                alpha=alpha)
                        for key in scored_res.keys():
                            adjusted_results[key][eff_idx, ssize_idx, rep_idx] = scored_res[key]

        results['Sample Size'] = sample_size
        results['Effect Size'] = effect_size

        if multiple_testing_correction is not None:
            adjusted_results['Sample Size'] = sample_size
            adjusted_results['Effect Size'] = effect_size

        # process the results...
        if multiple_testing_correction is None:
            return results
        else:
            return results, adjusted_results

    except TypeError as terp:
        raise terp
    except ValueError as verr:
        raise verr
    except Exception as exp:
        raise exp


def plsda_simulation(data, variables, effect_size, sample_size, alpha=0.05, n_repeats=15, weight_values=None,
                             weight_threshold=0.8, modification_type='correlation', class_balance=0.5,
                      test_set_proportion=1, n_components=10, n_components_criteria='fixed'):
    """

    :param data:
    :param variables:
    :param effect_size:
    :param sample_size:
    :param alpha:
    :param n_repeats:
    :param weight_values:
    :param weight_threshold:
    :param modification_type:
    :param class_balance:
    :param test_set_proportion:
    :param n_comps:
    :param n_components_criteria:
    :return:
    """

    try:
        import warnings
        warnings.filterwarnings('ignore')
        if modification_type not in ['correlation', 'manual', 'proportion', 'correlation_weighted']:
            raise ValueError("modification_type argument not supported")
        if modification_type == 'proportion' and not isinstance(variables, float):
            raise TypeError("When using \'proportion\' as modification_type \'variables\' must be a float")

        # get the list of metrics calculated in scoreResults and update
        results = dict.fromkeys(score_metrics)
        for key in results.keys():
            results[key] = np.zeros((effect_size.size, sample_size.size, n_repeats))

        n_vars = data.shape[1]
        # Loop over effect size, sample size and finally each monte carlo repeat
        for eff_idx, curr_effect in np.ndenumerate(effect_size):
            for ssize_idx, curr_ssize in np.ndenumerate(sample_size):
                for rep_idx in range(n_repeats):
                    # Select samples to use
                    ## Select a subset of the simulated spectra to make up training and test sets
                    train_x = np.copy(data[np.random.choice(data.shape[0], curr_ssize, replace=False), :])
                    test_x = np.copy(data[np.random.choice(data.shape[0],
                                                              int(np.floor(test_set_proportion*curr_ssize)), replace=False), :])

                    # Select a subset of samples to assign to class 2
                    which_samples_train = np.random.choice(range(curr_ssize), int(np.floor(class_balance * curr_ssize)),
                                                     replace=False)
                    which_samples_test = np.random.choice(test_x.shape[0], int(np.floor(class_balance * test_x.shape[0])),
                                                     replace=False)

                    train_y = np.zeros(train_x.shape[0])
                    train_y[which_samples_train] = 1
                    test_y = np.zeros(test_x.shape[0])
                    test_y[which_samples_test] = 1

                    if modification_type != 'proportion':
                        # Modify only variables above a certain threshold of correlation
                        var_to_mod = np.zeros(n_vars, dtype='int')
                        var_to_mod[variables] = 1

                        expected_hits = np.zeros(n_vars, dtype='int')
                        expected_hits[var_to_mod == 1] = 1
                        # If correlation and correlation_weighted
                        if weight_values is not None and modification_type in ["correlation", "correlation_weighted"]:
                            if weight_values.ndim == 1:
                                var_to_mod |= abs(weight_values) >= weight_threshold
                            else:
                                var_to_mod |= np.any(abs(weight_values) >= weight_threshold, axis=1)
                    else:
                        var_to_mod = np.random.choice(n_vars, int(np.floor(variables*n_vars)))

                    if modification_type == 'correlation_weighted':
                        train_x = effect_cohen_d(train_x, curr_effect, which_vars=var_to_mod,
                                                 which_samples=which_samples_train, standardized=True,
                                                 noise=0, weight=weight_values)

                        test_x = effect_cohen_d(test_x, curr_effect, which_vars=var_to_mod,
                                                  which_samples=which_samples_test, standardized=True,
                                                  noise=0, weight=weight_values)
                    else:
                        train_x = effect_cohen_d(train_x, curr_effect, which_vars=var_to_mod,
                                                  which_samples=which_samples_train, standardized=True,
                                                  noise=0, weight=None)
                        test_x = effect_cohen_d(test_x, curr_effect, which_vars=var_to_mod,
                                                  which_samples=which_samples_test, standardized=True,
                                                  noise=0, weight=None)

                    if n_components_criteria == 'scree':

                        # Fit a PLS-DA model
                        pls_da_model = ChemometricsPLSDA(ncomps=1)
                        # Automatically assess number of components

                        models = list()
                        for ncomps in range(1, n_components + 1):
                            currmodel = deepcopy(pls_da_model)
                            currmodel.ncomps = ncomps
                            currmodel.fit(train_x, train_y)
                            currmodel.cross_validation(train_x, train_y)
                            models.append(currmodel)

                        q2 = np.array([x.cvParameters['PLS']['Q2Y'] for x in models])
                        if q2.size == 2:
                            plateau_index = np.where(np.diff(q2) / q2[0] < 0.05)[0]
                            if plateau_index.size == 0:
                                n_components = n_components
                            else:
                                n_components = np.min(np.where(np.diff(q2) / q2[0] < 0.05)[0]) + 1
                        else:
                            plateau_index = np.where((np.diff(q2) / q2[0:-1]) < 0.05)[0]
                            if plateau_index.size == 0:
                                n_components = n_components
                            else:
                                n_components = np.min(plateau_index) + 1

                    pls_da_model = ChemometricsPLSDA(ncomps=n_components)
                    pls_da_model.fit(train_x, train_y)
                    predicted_y = pls_da_model.predict(test_x)

                    scored_res = score_classification_metrics(predicted_y, test_y, None, 1)

                    for key in scored_res.keys():
                        results[key][eff_idx, ssize_idx, rep_idx] = scored_res[key]

        results['Sample Size'] = sample_size
        results['Effect Size'] = effect_size
        results['Test Set Size'] = sample_size * test_set_proportion
        return results

    except TypeError as terp:
        raise terp
    except ValueError as verr:
        raise verr
    except Exception as exp:
        raise exp


power_analysis_types = {'ANOVA': anova_oneway_simulation, 'PLS-DA': plsda_simulation}

