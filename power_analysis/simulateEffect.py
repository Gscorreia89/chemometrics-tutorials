import numpy as np


def effect_cohen_d(x, effect_size, which_vars=0, which_samples=0, standardized=True, noise=0, weight=None):
    """

    Modify a data matrix adding an artificial effect parametrized by Cohen's d measure.
    The pooled standard deviation is calculated from the whole matrix, for the required variables.

    :param numpy.ndarray x: Data matrix
    :param numpy.ndarray or float effect_size: Value or vector of effect size to add
    :param numpy.ndarray, int which_vars: Index of variables to modify.
    :param numpy.ndarray or int which_samples: Index of samples to modify.
    :param boolean standardized: If True, use standardized effect size.
    :param float or numpy.ndarray noise: Noise value to add or noise covariance matrix.
    :param numpy.ndarray weight: Vector of values to weight the effect size (for example, pearson correlation).
    :return: Modified data matrix, with the added effect in the requested rows and columns
    and spiked in gaussian noise (optional).
    :rtype: numpy.ndarray
    """

    try:

        # if float, which_vars is interpreted as a proportion of variables, randomly selected
        if isinstance(which_vars, float):
            n_vars = which_samples * x.shape[1]
            which_vars = np.random.choice(x.shape[1], n_vars, replace=False)
        if which_vars.dtype in [float, int]:
            which_vars = np.where(which_vars)[0]
        # Use a standardized effect size
        if standardized is True:
            effect_size *= np.std(x[:, which_vars], axis=0)
        # Option to use weights  - ie, correlation weights
        if weight is not None:
            effect_size *= weight[which_vars]

        # Detect if noise is constant or a covariance matrix
        if isinstance(noise, np.ndarray):
            noise = np.dot(np.random.randn(np.array(which_samples).size, np.array(which_vars).size), noise)
        else:
            noise = np.random.randn(np.array(which_samples).size, np.array(which_vars).size) * noise
        # Add the effect and generate the output matrix
        x_modified = np.copy(x)
        x_modified[np.ix_(which_samples, which_vars)] = x_modified[np.ix_(which_samples, which_vars)] \
                                                        + effect_size + noise

        return x_modified

    except TypeError as terr:
        raise terr
    except ValueError as verr:
        raise verr
    except Exception as exp:
        raise exp
