import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Define a labeling convention
metric_label = {'True Positive Rate': 'Power', 'False Positive Rate': 'False Positive Rate',
                'True Negative Rate': 'True Negative Rate', 'False Negative Rate': 'False Negative Rate',
                'Positive Predictive Value': 'Positive Predictive Value',
                'Negative Predictive Value': 'Negative Predictive Value',
                'False Discovery Rate': 'False Discovery Rate', 'False Omission Rate': 'False Omission Rate',
                'Accuracy': 'Accuracy', 'F1': 'F1'}


def plot_metric_curve(analysis_output, metric='True Positive Rate',
                  plot_multiple_testing=False, fixed_effect_size=None,
                     fixed_sample_size=100, sd=2):
    """

    :param analysis_output:
    :param metric:
    :param plot_multiple_testing:
    :param fixed_effect_size:
    :param fixed_sample_size:
    :param sd:
    :return:
    """

    if plot_multiple_testing is True:
        results_idx = 1
    else:
        results_idx = 0

    if fixed_effect_size is not None:
        eff_index = np.where(analysis_output[0][0]['Effect Size'] == fixed_effect_size)[0]
        means_metric = np.array([np.nanmean(analysis_output[x][results_idx][metric], axis=2)[eff_index, :] for x in range(len(analysis_output))])
        stdev_metric = np.array([np.nanstd(analysis_output[x][results_idx][metric], axis=2)[eff_index, :] for x in range(len(analysis_output))])
        x = analysis_output[0][0]['Sample Size']
        y = np.nanstd(means_metric, axis=1)
        y_err = np.nanmean(stdev_metric, axis=1) * sd
        x_lab = 'Sample Size'

    elif fixed_sample_size is not None:
        samp_index = np.where(analysis_output[0][0]['Sample Size'] == fixed_sample_size)[0]
        means_metric = np.array([np.nanmean(analysis_output[x][results_idx][metric], axis=2)[:, samp_index] for x in range(len(analysis_output))])
        stdev_metric = np.array([np.nanstd(analysis_output[x][results_idx][metric], axis=2)[:, samp_index] for x in range(len(analysis_output))])
        x = analysis_output[0][0]['Effect Size']
        y = np.nanmean(means_metric, axis=0)
        y_err = np.nanmean(stdev_metric, 0) * sd
        # Truncate negative values for confidence intervals
        x_lab = 'Effect Size'

    lower_error = np.amax(np.c_[y - y_err, np.zeros(y.size)])
    upper_error = np.amin(np.c_[y + y_err, np.ones(y.size)])
    error_bar = [lower_error, upper_error]
    fig, ax = plt.subplots()
    ax.errorbar(x=x, y=y, yerr=error_bar)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(metric_label[metric])
    return ax


def plot_metric_heatmap(analysis_output, which_var, metric='True Positive Rate',
                       plot_multiple_testing=False, interpolation='bicubic', contour_level=0.8):
    """

    :param analysis_output:
    :param which_var:
    :param metric:
    :param plot_multiple_testing:
    :param interpolation:
    :param contour_level:
    :return:
    """

    if plot_multiple_testing is True:
        results_idx = 1
    else:
        results_idx = 0

    fig, ax = plt.subplots()

    if (len(analysis_output[0])) == 2:
        means_metric = np.nanmean(analysis_output[which_var][results_idx][metric], axis=2)
        n_samp_size = len(analysis_output[which_var][results_idx]['Sample Size'])
        samp_size = analysis_output[which_var][results_idx]['Sample Size']
        n_eff_size = len(analysis_output[which_var][results_idx]['Effect Size'])
        eff_size = analysis_output[which_var][results_idx]['Effect Size']
    else:
        means_metric = np.nanmean(analysis_output[which_var][metric], axis=2)
        n_samp_size = len(analysis_output[which_var]['Sample Size'])
        samp_size = analysis_output[which_var]['Sample Size']
        n_eff_size = len(analysis_output[which_var]['Effect Size'])
        eff_size = analysis_output[which_var]['Effect Size']
    color_norm = plt.Normalize(0, 1)
    heatmap = ax.imshow(means_metric, interpolation=interpolation, cmap='jet', norm=color_norm)
    if contour_level is not None:
        ax.contour(means_metric, np.array([contour_level]), linewidths=1)

    ax.set_xticks(np.arange(0, n_samp_size))
    ax.set_yticks(np.arange(0, n_eff_size))
    ax.set_xticklabels(samp_size)
    ax.set_yticklabels(eff_size)
    ax.set_ylabel('Effect Size')
    ax.set_xlabel('Sample Size')
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label(metric_label[metric], rotation=270)

    return ax


def plot_metric_spectrum(analysis_output, ref_spectrum, xvar, metric='True Positive Rate', plot_multiple_testing=False):
    """

    :param analysis_output:
    :param ref_spectrum:
    :param xvar:
    :param metric:
    :param plot_multiple_testing:
    :return:
    """

    if plot_multiple_testing is True:
        results_idx = 1
    else:
        results_idx = 0

    points = np.array([xvar, ref_spectrum]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    metric_value = np.array([np.nanmean(analysis_output[x][results_idx][metric], axis=2) for x in range(len(analysis_output))]).squeeze()

    fig, ax = plt.subplots()

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(metric_value)
    ax.plot(xvar, ref_spectrum, alpha=0)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_xlabel('Metabolic Variable')
    cbar.set_label(metric_label[metric], rotation=270)

    return ax