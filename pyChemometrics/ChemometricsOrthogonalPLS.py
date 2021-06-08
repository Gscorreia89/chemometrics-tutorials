from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._split import BaseShuffleSplit
from .ChemometricsScaler import ChemometricsScaler
from ._ortho_filter_pls import OrthogonalPLSRegression
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.stats as st
from .plotting_utils import _lineplots, _barplots

__author__ = 'gscorreia89'


class ChemometricsOrthogonalPLS(BaseEstimator, RegressorMixin, TransformerMixin):
    """

    ChemometricsOrthogonalPLS object.

    :param int ncomps: Number of Orthogonal PLS components desired. Must be 2 or greater.
    :param xscaler: Scaler object for X data matrix.
    :type xscaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None.
    :param yscaler: Scaler object for the Y data vector/matrix.
    :type yscaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None.
    :param kwargs pls_type_kwargs: Keyword arguments to be passed during initialization of pls_algorithm.
    :raise TypeError: If the pca_algorithm or scaler objects are not of the right class.
    """

    def __init__(self, ncomps=2,xscaler=ChemometricsScaler(), yscaler=None,
                 **pls_type_kwargs):

        try:

            # use custom orthogonal PLS regression code
            pls_algorithm = OrthogonalPLSRegression(ncomps, scale=False, **pls_type_kwargs)

            if not (isinstance(xscaler, TransformerMixin) or xscaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if not (isinstance(yscaler, TransformerMixin) or yscaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            # 2 blocks of data = two scaling options
            if xscaler is None:
                xscaler = ChemometricsScaler(0, with_std=False)
                # Force scaling to false, as this will be handled by the provided scaler or not
            if yscaler is None:
                yscaler = ChemometricsScaler(0, with_std=False)

            self.pls_algorithm = pls_algorithm
            # Most initialized as None, before object is fitted...

            # Orthogonal component parameters
            self.t_ortho = None
            self.w_ortho = None
            self.p_ortho = None
            self.u_ortho = None
            self.c_ortho = None
            self.q_ortho = None

            # "predictive" component parameters
            self.t_pred = None
            self.w_pred = None
            self.p_pred = None
            self.u_pred = None
            self.q_pred = None
            self.c_pred = None

            self.t = None
            self.u = None

            self.b_t = None
            self.b_u = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.beta_coeffs = None

            self._ncomps = ncomps
            self._x_scaler = xscaler
            self._y_scaler = yscaler
            self.cvParameters = None
            self.modelParameters = None
            self._isfitted = False

        except TypeError as terp:
            print(terp.args[0])

    def fit(self, x, y, **fit_params):
        """

        Perform model fitting on the provided x and y data and calculate basic goodness-of-fit metrics.
        Similar to scikit-learn's BaseEstimator method.

        :param x: Data matrix to fit the Orthogonal PLS model.
        :type x: numpy.ndarray, shape [n_samples, n_features].
        :param y: Data matrix to fit the Orthogonal PLS model.
        :type y: numpy.ndarray, shape [n_samples, n_features].
        :param kwargs fit_params: Keyword arguments to be passed to the .fit() method of the core sklearn model.
        :raise ValueError: If any problem occurs during fitting.
        """
        try:
            # To comply with the sklearn-scaler behaviour convention regarding 1d arrays
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            # Not so important as don't expect a user applying a single x variable to a multivariate regression
            # method, but for consistency/testing purposes
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            xscaled = self.x_scaler.fit_transform(x)
            yscaled = self.y_scaler.fit_transform(y)

            self.pls_algorithm.fit(xscaled, yscaled, **fit_params)

            # Expose the model parameters
            self.p_pred = self.pls_algorithm.predictive_p
            self.q_pred = self.pls_algorithm.predictive_q
            self.w_pred = self.pls_algorithm.predictive_w
            self.c_pred = self.pls_algorithm.predictive_c
            self.t_pred = self.pls_algorithm.predictive_t
            self.u_pred = self.pls_algorithm.predictive_u

            self.rotations_ws = self.pls_algorithm.x_rotations_
            self.rotations_cs = self.pls_algorithm.y_rotations_

            # orthogonal Parameters
            self.t_ortho = self.pls_algorithm.t_ortho
            self.w_ortho = self.pls_algorithm.w_ortho
            self.p_ortho = self.pls_algorithm.p_ortho
            self.c_ortho = self.pls_algorithm.c_ortho
            self.u_ortho = self.pls_algorithm.u_ortho
            self.q_ortho = self.pls_algorithm.q_ortho

            self.b_t = self.pls_algorithm.b_t
            self.b_u = self.pls_algorithm.b_u
            # scores so everything can be calculated easily
            self.t = np.dot(xscaled, self.pls_algorithm.x_rotations_)
            self.u = np.dot(yscaled, self.pls_algorithm.y_rotations_)

            self.beta_coeffs = self.pls_algorithm.coef_
            # Needs to come here for the method shortcuts down the line to work...
            self._isfitted = True

            # Calculate RSSy/RSSx, R2Y/R2X
            R2Y = ChemometricsOrthogonalPLS.score(self, x=x, y=y, block_to_score='y')
            R2X = ChemometricsOrthogonalPLS.score(self, x=x, y=y, block_to_score='x')

            self.modelParameters = {'R2Y': R2Y, 'R2X': R2X}

        except ValueError as verr:
            raise verr

    def fit_transform(self, x, y, **fit_params):
        """

        Fit a model to supplied data and return the scores. Equivalent to scikit-learn's TransformerMixin method.

        :param x: Data matrix to fit the PLS model.
        :type x: numpy.ndarray, shape [n_samples, n_features].
        :param y: Data matrix to fit the PLS model.
        :type y: numpy.ndarray, shape [n_samples, n_features].
        :param kwargs fit_params: Optional keyword arguments to be passed to the pls_algorithm .fit() method.
        :return: Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).
        :rtype: tuple of numpy.ndarray, shape [[n_tscores], [n_uscores]]
        :raise ValueError: If any problem occurs during fitting.
        """

        try:
            self.fit(x, y, **fit_params)
            return self.transform(x, y=None), self.transform(x=None, y=y)

        except ValueError as verr:
            raise verr

    def transform(self, x=None, y=None):
        """

        Calculate the scores for a data block from the original data. Equivalent to sklearn's TransformerMixin method.

        :param x: Data matrix to fit the PLS model.
        :type x: numpy.ndarray, shape [n_samples, n_features] or None
        :param y: Data matrix to fit the PLS model.
        :type y: numpy.ndarray, shape [n_samples, n_features] or None
        :return: Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).
        :rtype: tuple with 2 numpy.ndarray, shape [n_samples, n_comps]
        :raise ValueError: If dimensions of input data are mismatched.
        :raise AttributeError: When calling the method before the model is fitted.
        """

        try:
            # Check if model is fitted
            if self._isfitted is True:
                # If X and Y are passed, complain and do nothing
                if (x is not None) and (y is not None):
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif (x is None) and (y is None):
                    raise ValueError('yy')
                # If Y is given, return U
                elif x is None:
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)
                    yscaled = self.y_scaler.transform(y)
                    # Taking advantage of rotations_y
                    # Otherwise this would be the full calculation U = Y*pinv(CQ')*C
                    U = np.dot(yscaled, self.rotations_cs)
                    return U

                # If X is given, return T
                elif y is None:
                    # Not so important as don't expect a user applying a single x variable to a multivariate regression
                    # method, but for consistency/testing purposes
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    xscaled = self.x_scaler.transform(x)
                    # Taking advantage of already calculated rotation_x
                    # Otherwise this would be would the full calculation T = X*pinv(WP')*W
                    T = np.dot(xscaled, self.rotations_ws)
                    return T

            else:
                raise AttributeError('Model not fitted')

        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def inverse_transform(self, t=None, u=None):
        """

        Transform scores to the original data space using their corresponding loadings.
        Same logic as in scikit-learn's TransformerMixin method.

        :param t: T scores corresponding to the X data matrix.
        :type t: numpy.ndarray, shape [n_samples, n_comps] or None
        :param u: Y scores corresponding to the Y data vector/matrix.
        :type u: numpy.ndarray, shape [n_samples, n_comps] or None
        :return x: X Data matrix in the original data space.
        :rtype: numpy.ndarray, shape [n_samples, n_features] or None
        :return y: Y Data matrix in the original data space.
        :rtype: numpy.ndarray, shape [n_samples, n_features] or None
        :raise ValueError: If dimensions of input data are mismatched.
        """

        try:
            if self._isfitted is True:
                if t is not None and u is not None:
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif t is None and u is None:
                    raise ValueError('yy')
                # If T is given, return U
                elif t is not None:
                    # Calculate X from T using X = TP'
                    if self.p_ortho is not None:
                        p_loadings = np.c_[self.p_ortho, self.p_pred]
                    else:
                        p_loadings = self.p_pred
                    xpred = np.dot(t, p_loadings.T)
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.inverse_transform(xpred)
                    else:
                        xscaled = xpred
                    return xscaled
                # If U is given, return T
                elif u is not None:
                    # Calculate Y from U - using Y = UQ'
                    if self.q_ortho is not None:
                        q_loadings = np.c_[self.q_ortho, self.q_pred]
                    else:
                        q_loadings = self.q_pred
                    ypred = np.dot(u, q_loadings.T)
                    if self.y_scaler is not None:
                        yscaled = self.y_scaler.inverse_transform(ypred)
                    else:
                        yscaled = ypred

                    return yscaled

        except ValueError as verr:
            raise verr

    def score(self, x, y, block_to_score='y', sample_weight=None):
        """

        Predict and calculate the R2 for the model using one of the data blocks (X or Y) provided.
        Equivalent to the scikit-learn RegressorMixin score method.

        :param x: Data matrix to fit the PLS model.
        :type x: numpy.ndarray, shape [n_samples, n_features] or None
        :param y: Data matrix to fit the PLS model.
        :type y: numpy.ndarray, shape [n_samples, n_features] or None
        :param str block_to_score: Which of the data blocks (X or Y) to calculate the R2 goodness of fit.
        :param sample_weight: Optional sample weights to use in scoring.
        :type sample_weight: numpy.ndarray, shape [n_samples] or None
        :return R2Y: The model's R2Y, calculated by predicting Y from X and scoring.
        :rtype: float
        :return R2X: The model's R2X, calculated by predicting X from Y and scoring.
        :rtype: float
        :raise ValueError: If block to score argument is not acceptable or date mismatch issues with the provided data.
        """

        try:
            if block_to_score not in ['x', 'y']:
                raise ValueError("x or y are the only accepted values for block_to_score")
            # Comply with the sklearn scaler behaviour
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            # Not so important as don't expect a user applying a single x variable to a multivariate regression
            # method, but for consistency/testing purposes
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            # Calculate RSSy/RSSx, R2Y/R2X
            if block_to_score == 'y':
                yscaled = deepcopy(self.y_scaler).fit_transform(y)
                # Calculate total sum of squares of X and Y for R2X and R2Y calculation
                tssy = np.sum(yscaled ** 2)
                ypred = self.y_scaler.transform(ChemometricsOrthogonalPLS.predict(self, x, y=None))
                rssy = np.sum((yscaled - ypred) ** 2)
                R2Y = 1 - (rssy / tssy)
                return R2Y
            # The prediction here of both X and Y is done using the other block of data only
            # so these R2s can be interpreted as as a "classic" R2, and not as a proportion of variance modelled
            # Here we use X = Ub_uW', as opposed to (X = TP').
            else:
                # Kept here for easier adaptation from sklearn
                xscaled = deepcopy(self.x_scaler).fit_transform(x)
                # Calculate total sum of squares of X and Y for R2X and R2Y calculation
                tssx = np.sum(xscaled ** 2)
                xpred = self.x_scaler.transform(ChemometricsOrthogonalPLS.predict(self, x=None, y=y))
                tssx = np.sum(xscaled ** 2)
                rssx = np.sum((xscaled - xpred) ** 2)
                R2X = 1 - (rssx / tssx)
                return R2X

        except ValueError as verr:
            raise verr

    def predict(self, x=None, y=None):
        """

        Predict the values in one data block using the other. Same as its scikit-learn's RegressorMixin namesake method.

        :param x: Data matrix to fit the PLS model.
        :type x: numpy.ndarray, shape [n_samples, n_features] or None
        :param y: Data matrix to fit the PLS model.
        :type y: numpy.ndarray, shape [n_samples, n_features] or None
        :return: Predicted data block (X or Y) obtained from the other data block.
        :rtype: numpy.ndarray, shape [n_samples, n_features]
        :raise ValueError: If no data matrix is passed, or dimensions mismatch issues with the provided data.
        :raise AttributeError: Calling the method without fitting the model before.
        """
        try:
            if self._isfitted is True:
                if (x is not None) and (y is not None):
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif (x is None) and (y is None):
                    raise ValueError('yy')
                # Predict Y from X
                elif x is not None:
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    xscaled = self.x_scaler.transform(x)
                    # Using Betas to predict Y directly
                    predicted = np.dot(xscaled, self.beta_coeffs)
                    if predicted.ndim == 1:
                        predicted = predicted.reshape(-1, 1)
                    predicted = self.y_scaler.inverse_transform(predicted)
                    return predicted
                # Predict X from Y
                elif y is not None:
                    # Comply with the sklearn scaler behaviour
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)
                    # Going through calculation of U and then X = Ub_uW'
                    u_scores = self.transform(x=None, y=y)
                    if self.w_ortho is not None:
                        w = np.c_[self.w_ortho, self.w_pred]
                    else:
                        w = self.w_pred
                    predicted = np.dot(np.dot(u_scores, self.b_u), w.T)
                    if predicted.ndim == 1:
                        predicted = predicted.reshape(-1, 1)
                    predicted = self.x_scaler.inverse_transform(predicted)
                    return predicted
            else:
                raise AttributeError("Model is not fitted")
        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def hotelling_T2(self, orth_comps=[0], alpha=0.05):
        """

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        :param list orth_comps: List of components to calculate the Hotelling T2.
        :param float alpha: Significant level for the F statistic.
        :return: List with the Hotelling T2 ellipse radii
        :rtype: list
        :raise ValueError: If the dimensions request
        """
        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")

            nsamples = self.t_pred.shape[0]

            if orth_comps is None:
                ncomps = self.ncomps
                ellips = np.c_[self.t_pred, self.t_ortho[:, range(self.ncomps - 1)]] ** 2
            else:
                ncomps = 1 + len(orth_comps)
                ellips = np.c_[self.t_pred, self.t_ortho[:, orth_comps]] ** 2

            ellips = 1 / nsamples * (ellips.sum(0))

            # F stat
            a = (nsamples - 1) / nsamples * ncomps * (nsamples ** 2 - 1) / (nsamples * (nsamples - ncomps))
            a = a * st.f.ppf(1-alpha, ncomps, nsamples - ncomps)

            hoteling_t2 = list()
            for comp in range(ncomps):
                hoteling_t2.append(np.sqrt((a * ellips[comp])))

            return np.array(hoteling_t2)

        except AttributeError as atre:
            raise atre
        except ValueError as valerr:
            raise valerr
        except TypeError as typerr:
            raise typerr

    def dmodx(self, x):
        """

        Normalised DmodX measure

        :param x: data matrix [n samples, m variables]
        :return: The Normalised DmodX measure for each sample
        """
        resids_ssx = self._residual_ssx(x)
        s = np.sqrt(resids_ssx/(self.p_pred.shape[1] - self.ncomps))
        dmodx = np.sqrt((s/self.modelParameters['S0X'])**2)
        return dmodx

    def outlier(self, x, comps=None, measure='T2', alpha=0.05):
        """

        Use the Hotelling T2 or DmodX measure and F statistic to screen for outlier candidates.

        :param x: Data matrix [n samples, m variables]
        :param comps: Which components to use (for Hotelling T2 only)
        :param measure: Hotelling T2 or DmodX
        :param alpha: Significance level
        :return: List with row indices of X matrix
        """
        try:
            if comps is None:
                comps = range(self.t.shape[1])
            if measure == 'T2':
                scores = self.transform(x)
                t2 = self.hotelling_T2(comps=comps)
                outlier_idx = np.where(((scores[:, comps] ** 2) / t2 ** 2).sum(axis=1) > 1)[0]
            elif measure == 'DmodX':
                dmodx = self.dmodx(x)
                dcrit = st.f.ppf(1 - alpha, x.shape[1] - self.ncomps,
                                 (x.shape[0] - self.ncomps - 1) * (x.shape[1] - self.ncomps))
                outlier_idx = np.where(dmodx > dcrit)[0]
            else:
                print("Select T2 (Hotelling T2) or DmodX as outlier exclusion criteria")
            return outlier_idx
        except Exception as exp:
            raise exp

    def plot_scores(self, orthogonal_component=1, color=None, discrete=False, label_outliers=False, plot_title=None):
        """

        Score plot figure wth an Hotelling T2.

        :param comps: Components to use in the 2D plot
        :param color: Variable used to color points
        :return: Score plot figure
        """
        try:
            fig, ax = plt.subplots()

            orthogonal_component = np.array([orthogonal_component - 1])

            t2 = self.hotelling_T2(alpha=0.05, orth_comps=orthogonal_component)

            score_mat = np.c_[self.t_pred, self.t_ortho[:, orthogonal_component]]

            outlier_idx = np.where(((score_mat ** 2) / t2 ** 2).sum(axis=1) > 1)[0]

            # Sort out colormaps if color argument is passed
            if color is None:
                # Set default color to 1
                color = np.ones(self.t_pred.shape[0])

            if discrete is False:
                cmap = cm.jet
                cnorm = Normalize(vmin=min(color), vmax=max(color))

                ax.scatter(score_mat[:, 0], score_mat[:, 1], c=color, cmap=cmap, norm=cnorm)
                #ax.scatter(score_mat[outlier_idx, 0], score_mat[outlier_idx, 1],
                #            c=color[outlier_idx], cmap=cmap, norm=cnorm, marker='x',
                #            s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
                ax.colorbar()

            else:
                cmap = cm.Set1
                subtypes = np.unique(color)
                for subtype in subtypes:
                    subset_index = np.where(color == subtype)
                    ax.scatter(score_mat[subset_index, 0], score_mat[subset_index, 1],
                                c=cmap(subtype), label=subtype)

                ax.legend()
                #ax.scatter(score_mat[outlier_idx, 0], score_mat[outlier_idx, 1],
                #            c=color[outlier_idx], cmap=cmap, marker='x',
                #            s=1.5 * mpl.rcParams['lines.markersize'] ** 2)

            if label_outliers:
                for outlier in outlier_idx:
                    ax.annotate(outlier, (score_mat[outlier, 0] + score_mat[outlier, 0]*0.05,
                                          score_mat[outlier, 1] + score_mat[outlier, 1]*0.05))

            angle = np.arange(-np.pi, np.pi, 0.01)
            x = t2[0] * np.cos(angle)
            y = t2[1] * np.sin(angle)
            ax.axhline(c='k')
            ax.axvline(c='k')
            ax.plot(x, y, c='k')

            xmin = np.minimum(min(score_mat[:, 0]), np.min(x))
            xmax = np.maximum(max(score_mat[:, 0]), np.max(x))
            ymin = np.minimum(min(score_mat[:, 1]), np.min(y))
            ymax = np.maximum(max(score_mat[:, 1]), np.max(y))

            #axes = plt.gca()
            ax.set_xlim([(xmin + (0.2 * xmin)), xmax + (0.2 * xmax)])
            ax.set_ylim([(ymin + (0.2 * ymin)), ymax + (0.2 * ymax)])

            if plot_title is None:
                fig.suptitle("OrthogonalPLS score plot")
            else:
                fig.suptitle(plot_title)

            ax.set_xlabel("Tpred")
            ax.set_ylabel("Tortho{0}".format((orthogonal_component + 1)))
            plt.show()

        except (ValueError, IndexError) as verr:
            print("The number of components to plot must not exceed 2 and the component choice cannot "
                  "exceed the number of components in the model")
            raise Exception

        return ax

    def plot_model_parameters(self, parameter='w_pred', orthogonal_component=1, cross_val=False, sigma=2, bar=False, xaxis=None):

        choices = {'w_pred': self.w_pred, 'p_pred': self.p_pred, 'w_ortho': self.w_ortho, 'p_ortho': self.p_ortho}
        choices_cv = {'wpred': 'Wpred_w_pred', }

        # decrement component to adjust for python indexing
        orthogonal_component -= 1
        # Beta and VIP don't depend on components so have an exception status here
        if cross_val is True:
            if parameter in ['w_pred', 'p_pred']:
                mean = self.cvParameters['Mean_' + choices_cv[parameter]].squeeze()
                error = sigma * self.cvParameters['Stdev_' + choices_cv[parameter]].squeeze()
            else:
                mean = self.cvParameters['Mean_' + choices_cv[parameter]][:, orthogonal_component]
                error = sigma * self.cvParameters['Stdev_' + choices_cv[parameter]][:, orthogonal_component]
        else:
            error = None
            if parameter in ['w_pred', 'p_pred']:
                mean = choices[parameter]
            else:
                mean = choices[parameter][:, orthogonal_component]
        if bar is False:
            _lineplots(mean, error=error, xaxis=xaxis)
        # To use with barplots for other types of data
        else:
            _barplots(mean, error=error, xaxis=xaxis)

        plt.xlabel("Variable No")
        if parameter in ['w_pred', 'p_pred']:
            plt.ylabel("{0} for Orthogonal PLS model".format(parameter))
        else:
            plt.ylabel("{0} for Orthogonal PLS component {1}".format(parameter, (orthogonal_component + 1)))
        plt.show()

        return None

    @property
    def ncomps(self):
        try:
            return self._ncomps
        except AttributeError as atre:
            raise atre

    @ncomps.setter
    def ncomps(self, ncomps=1):
        """

        Setter for number of components. Re-sets the model.

        :param int ncomps: Number of PLS components to use in the model.
        :raise AttributeError: If there is a problem changing the number of components and resetting the model.
        """
        # To ensure changing number of components effectively resets the model
        try:

            self._ncomps = ncomps
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.pls_algorithm.n_components = ncomps

            # Orthogonal component parameters
            self.t_ortho = None
            self.w_ortho = None
            self.p_ortho = None
            self.u_ortho = None
            self.c_ortho = None
            self.q_ortho = None

            # "predictive" component parameters
            self.t_pred = None
            self.w_pred = None
            self.p_pred = None
            self.u_pred = None
            self.q_pred = None
            self.c_pred = None

            self.b_t = None
            self.b_u = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.beta_coeffs = None

            self.cvParameters = None
            self.modelParameters = None
            self.b_t = None
            self.b_u = None

            return None
        except AttributeError as atre:
            raise atre

    @property
    def x_scaler(self):
        try:
            return self._x_scaler
        except AttributeError as atre:
            raise atre

    @x_scaler.setter
    def x_scaler(self, scaler):
        """

        Setter for the X data block scaler.

        :param scaler: The object which will handle data scaling.
        :type scaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None
        :raise AttributeError: If there is a problem changing the scaler and resetting the model.
        :raise TypeError: If the new scaler provided is not a valid object.
        """

        try:

            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = ChemometricsScaler(0, with_std=False)

            self._x_scaler = scaler
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.modelParameters = None
            self.cvParameters = None

            # Orthogonal component parameters
            self.t_ortho = None
            self.w_ortho = None
            self.p_ortho = None
            self.u_ortho = None
            self.c_ortho = None
            self.q_ortho = None

            # "predictive" component parameters
            self.t_pred = None
            self.w_pred = None
            self.p_pred = None
            self.u_pred = None
            self.q_pred = None
            self.c_pred = None

            self.b_t = None
            self.b_u = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.beta_coeffs = None

            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    @property
    def y_scaler(self):
        try:
            return self._y_scaler
        except AttributeError as atre:
            raise atre

    @y_scaler.setter
    def y_scaler(self, scaler):
        """

        Setter for the Y data block scaler.

        :param scaler: The object which will handle data scaling.
        :type scaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None
        :raise AttributeError: If there is a problem changing the scaler and resetting the model.
        :raise TypeError: If the new scaler provided is not a valid object.
        """
        try:
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = ChemometricsScaler(0, with_std=False)

            self._y_scaler = scaler
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.modelParameters = None
            self.cvParameters = None

            # Orthogonal component parameters
            self.t_ortho = None
            self.w_ortho = None
            self.p_ortho = None
            self.u_ortho = None
            self.c_ortho = None
            self.q_ortho = None

            # "predictive" component parameters
            self.t_pred = None
            self.w_pred = None
            self.p_pred = None
            self.u_pred = None
            self.q_pred = None
            self.c_pred = None

            self.b_t = None
            self.b_u = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.beta_coeffs = None

            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    def cross_validation(self, x, y, cv_method=KFold(7, shuffle=True), outputdist=False,
                         **crossval_kwargs):
        """

        Cross-validation method for the model. Calculates Q2 and cross-validated estimates for all model parameters.

        :param x: Data matrix to fit the PLS model.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param y: Data matrix to fit the PLS model.
        :type y: numpy.ndarray, shape [n_samples, n_features]
        :param cv_method: An instance of a scikit-learn CrossValidator object.
        :type cv_method: BaseCrossValidator or BaseShuffleSplit
        :param bool outputdist: Output the whole distribution for. Useful when ShuffleSplit or CrossValidators other than KFold.
        :param kwargs crossval_kwargs: Keyword arguments to be passed to the sklearn.Pipeline during cross-validation
        :return:
        :rtype: dict
        :raise TypeError: If the cv_method passed is not a scikit-learn CrossValidator object.
        :raise ValueError: If the x and y data matrices are invalid.
        """

        try:
            if not (isinstance(cv_method, BaseCrossValidator) or isinstance(cv_method, BaseShuffleSplit)):
                raise TypeError("Scikit-learn cross-validation object please")

            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False:
                self.fit(x, y)

            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            cv_pipeline = deepcopy(self)
            ncvrounds = cv_method.get_n_splits()

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            if y.ndim > 1:
                y_nvars = y.shape[1]
            else:
                y_nvars = 1
                y = y.reshape(-1, 1)

            # Initialize arrays
            cv_pred_w = np.zeros((ncvrounds, x_nvars, 1))
            cv_ortho_w = np.zeros((ncvrounds, x_nvars, self.ncomps -1))
            cv_pred_p = np.zeros((ncvrounds, x_nvars, 1))
            cv_ortho_p = np.zeros((ncvrounds, x_nvars, self.ncomps - 1))

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            pressy = 0
            pressx = 0

            # Calculate Sum of Squares SS in whole dataset for future calculations
            ssx = np.sum(np.square(cv_pipeline.x_scaler.fit_transform(x)))
            ssy = np.sum(np.square(cv_pipeline.y_scaler.fit_transform(y)))

            # As assessed in the test set..., opposed to PRESS
            R2X_training = np.zeros(ncvrounds)
            R2Y_training = np.zeros(ncvrounds)
            # R2X and R2Y assessed in the test set
            R2X_test = np.zeros(ncvrounds)
            R2Y_test = np.zeros(ncvrounds)

            for cvround, train_testidx in enumerate(cv_method.split(x, y)):
                # split the data explicitly
                train = train_testidx[0]
                test = train_testidx[1]

                # Check dimensions for the indexing
                if y_nvars == 1:
                    ytrain = y[train]
                    ytest = y[test]
                else:
                    ytrain = y[train, :]
                    ytest = y[test, :]
                if x_nvars == 1:
                    xtrain = x[train]
                    xtest = x[test]
                else:
                    xtrain = x[train, :]
                    xtest = x[test, :]

                cv_pipeline.fit(xtrain, ytrain, **crossval_kwargs)
                # Prepare the scaled X and Y test data
                # If testset_scale is True, these are scaled individually...

                # Comply with the sklearn scaler behaviour
                if ytest.ndim == 1:
                    ytest = ytest.reshape(-1, 1)
                    ytrain = ytrain.reshape(-1, 1)
                if xtest.ndim == 1:
                    xtest = xtest.reshape(-1, 1)
                    xtrain = xtrain.reshape(-1, 1)
                # Fit the training data

                xtest_scaled = cv_pipeline.x_scaler.transform(xtest)
                ytest_scaled = cv_pipeline.y_scaler.transform(ytest)

                R2X_training[cvround] = cv_pipeline.score(xtrain, ytrain, 'x')
                R2Y_training[cvround] = cv_pipeline.score(xtrain, ytrain, 'y')
                ypred = cv_pipeline.predict(x=xtest, y=None)
                xpred = cv_pipeline.predict(x=None, y=ytest)

                xpred = cv_pipeline.x_scaler.transform(xpred).squeeze()

                ypred = cv_pipeline.y_scaler.transform(ypred).squeeze()
                ytest_scaled = ytest_scaled.squeeze()

                curr_pressx = np.sum(np.square(xtest_scaled - xpred))
                curr_pressy = np.sum(np.square(ytest_scaled - ypred))

                R2X_test[cvround] = cv_pipeline.score(xtest, ytest, 'x')
                R2Y_test[cvround] = cv_pipeline.score(xtest, ytest, 'y')

                pressx += curr_pressx
                pressy += curr_pressy

                cv_loadings_p[cvround, :, :] = cv_pipeline.loadings_p
                cv_loadings_q[cvround, :, :] = cv_pipeline.loadings_q
                cv_weights_w[cvround, :, :] = cv_pipeline.weights_w
                cv_weights_c[cvround, :, :] = cv_pipeline.weights_c

            for cvround in range(0, ncvrounds):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = np.argmin(
                        np.array([np.sum(np.abs(self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload])),
                                  np.sum(np.abs(
                                      self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload] * -1))]))
                    if choice == 1:
                        cv_loadings_p[cvround, :, currload] = -1 * cv_loadings_p[cvround, :, currload]
                        cv_loadings_q[cvround, :, currload] = -1 * cv_loadings_q[cvround, :, currload]
                        cv_weights_w[cvround, :, currload] = -1 * cv_weights_w[cvround, :, currload]
                        cv_weights_c[cvround, :, currload] = -1 * cv_weights_c[cvround, :, currload]
                        cv_rotations_ws[cvround, :, currload] = -1 * cv_rotations_ws[cvround, :, currload]
                        cv_rotations_cs[cvround, :, currload] = -1 * cv_rotations_cs[cvround, :, currload]

            # Calculate total sum of squares
            q_squaredy = 1 - (pressy / ssy)
            q_squaredx = 1 - (pressx / ssx)

            # Store everything...
            self.cvParameters = {'Q2X': q_squaredx, 'Q2Y': q_squaredy, 'MeanR2X_Training': np.mean(R2X_training),
                                 'MeanR2Y_Training': np.mean(R2Y_training), 'StdevR2X_Training': np.std(R2X_training),
                                 'StdevR2Y_Training': np.std(R2Y_training), 'MeanR2X_Test': np.mean(R2X_test),
                                 'MeanR2Y_Test': np.mean(R2Y_test), 'StdevR2X_Test': np.std(R2X_test),
                                 'StdevR2Y_Test': np.std(R2Y_test), 'Mean_Loadings_q': cv_loadings_q.mean(0),
                                 'Stdev_Loadings_q': cv_loadings_q.std(0), 'Mean_Loadings_p': cv_loadings_p.mean(0),
                                 'Stdev_Loadings_p': cv_loadings_q.std(0), 'Mean_Weights_c': cv_weights_c.mean(0),
                                 'Stdev_Weights_c': cv_weights_c.std(0), 'Mean_Weights_w': cv_weights_w.mean(0),
                                 'Stdev_Weights_w': cv_weights_w.std(0), 'Mean_Rotations_ws': cv_rotations_ws.mean(0),
                                 'Stdev_Rotations_ws': cv_rotations_ws.std(0),
                                 'Mean_Rotations_cs': cv_rotations_cs.mean(0),
                                 'Stdev_Rotations_cs': cv_rotations_cs.std(0), 'Mean_Beta': cv_betacoefs.mean(0),
                                 'Stdev_Beta': cv_betacoefs.std(0), 'Mean_VIP': cv_vipsw.mean(0),
                                 'Stdev_VIP': cv_vipsw.std(0)}

            # Save everything found during CV
            if outputdist is True:
                self.cvParameters['CVR2X_Training'] = R2X_training
                self.cvParameters['CVR2Y_Training'] = R2Y_training
                self.cvParameters['CVR2X_Test'] = R2X_test
                self.cvParameters['CVR2Y_Test'] = R2Y_test
                self.cvParameters['CV_Loadings_q'] = cv_loadings_q
                self.cvParameters['CV_Loadings_p'] = cv_loadings_p
                self.cvParameters['CV_Weights_c'] = cv_weights_c
                self.cvParameters['CV_Weights_w'] = cv_weights_w
                self.cvParameters['CV_Rotations_ws'] = cv_rotations_ws
                self.cvParameters['CV_Rotations_cs'] = cv_rotations_cs
                self.cvParameters['CV_Train_Scores_t'] = cv_train_scores_t
                self.cvParameters['CV_Train_Scores_u'] = cv_test_scores_u
                self.cvParameters['CV_Beta'] = cv_betacoefs
                self.cvParameters['CV_VIPw'] = cv_vipsw

            return None

        except TypeError as terp:
            raise terp

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
