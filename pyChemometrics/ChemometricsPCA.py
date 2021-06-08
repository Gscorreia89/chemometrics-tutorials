from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA as skPCA
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._split import BaseShuffleSplit
from .ChemometricsScaler import ChemometricsScaler
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import scipy.stats as st
import matplotlib.cm as cm

__author__ = 'gscorreia89'

from copy import deepcopy

class ChemometricsPCA(BaseEstimator):
    """

    ChemometricsPCA object - Wrapper for sklearn.decomposition PCA algorithms, with tailored methods
    for Chemometric Data analysis.

    :param ncomps: Number of PCA components desired.
    :type ncomps: int
    :param sklearn.decomposition._BasePCA pca_algorithm: scikit-learn PCA algorithm to use (inheriting from _BasePCA).
    :param scaler: The object which will handle data scaling.
    :type scaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None
    :param kwargs pca_type_kwargs: Keyword arguments to be passed during initialization of pca_algorithm.
    :raise TypeError: If the pca_algorithm or scaler objects are not of the right class.
    """

    # Constant usage of kwargs might look excessive but ensures that most things from scikit-learn can be used directly
    # no matter what PCA algorithm is used
    def __init__(self, ncomps=2, pca_algorithm=skPCA, scaler=ChemometricsScaler(), **pca_type_kwargs):

        try:
            # Perform the check with is instance but avoid abstract base class runs. PCA needs number of comps anyway!
            init_pca_algorithm = pca_algorithm(n_components=ncomps, **pca_type_kwargs)
            if not isinstance(init_pca_algorithm, (BaseEstimator, TransformerMixin)):
                raise TypeError("Use a valid scikit-learn PCA model please")
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = ChemometricsScaler(0, with_std=False)

            self.pca_algorithm = init_pca_algorithm

            # Most initialized as None, before object is fitted.
            self.scores = None
            self.loadings = None
            self._ncomps = ncomps
            self._scaler = scaler
            self.cvParameters = None
            self.modelParameters = None
            self._isfitted = False

        except TypeError as terp:
            print(terp.args[0])
            raise terp

    def fit(self, x, **fit_params):
        """

        Perform model fitting on the provided x data matrix and calculate basic goodness-of-fit metrics.
        Equivalent to scikit-learn's default BaseEstimator method.

        :param x: Data matrix to fit the PCA model.
        :type x: numpy.ndarray, shape [n_samples, n_features].
        :param kwargs fit_params: Keyword arguments to be passed to the .fit() method of the core sklearn model.
        :raise ValueError: If any problem occurs during fitting.
        """

        try:
            # This scaling check is always performed to ensure running model with scaling or with scaling == None
            # always give consistent results (same type of data scale expected for fitting,
            # returned by inverse_transform, etc
            if self.scaler is not None:
                xscaled = self.scaler.fit_transform(x)
                self.pca_algorithm.fit(xscaled, **fit_params)
                self.scores = self.pca_algorithm.transform(xscaled)
                ss = np.sum((xscaled - np.mean(xscaled, 0)) ** 2)
                predicted = self.pca_algorithm.inverse_transform(self.scores)
                rss = np.sum((xscaled - predicted) ** 2)
                # variance explained from scikit-learn stored as well
            else:
                self.pca_algorithm.fit(x, **fit_params)
                self.scores = self.pca_algorithm.transform(x)
                ss = np.sum((x - np.mean(x, 0)) ** 2)
                predicted = self.pca_algorithm.inverse_transform(self.scores)
                rss = np.sum((x - predicted) ** 2)
            self.modelParameters = {'R2X': 1 - (rss / ss), 'VarExp': self.pca_algorithm.explained_variance_,
                                    'VarExpRatio': self.pca_algorithm.explained_variance_ratio_}

            # For "Normalised" DmodX calculation
            resid_ssx = self._residual_ssx(x)
            s0 = np.sqrt(resid_ssx.sum()/((self.scores.shape[0] - self.ncomps - 1)*(x.shape[1] - self.ncomps)))
            self.modelParameters['S0'] = s0
            # Kernel PCA and other non-linear methods might not have explicit loadings - safeguard against this
            if hasattr(self.pca_algorithm, 'components_'):
                self.loadings = self.pca_algorithm.components_
            self._isfitted = True

        except ValueError as verr:
            raise verr

    def fit_transform(self, x, **fit_params):
        """

        Fit a model and return the scores, as per the scikit-learn's TransformerMixin method.

        :param x: Data matrix to fit and project.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param kwargs fit_params: Optional keyword arguments to be passed to the fit method.
        :return: PCA projections (scores) corresponding to the samples in X.
        :rtype: numpy.ndarray, shape [n_samples, n_comps]
        :raise ValueError: If there are problems with the input or during model fitting.
        """

        try:
            self.fit(x, **fit_params)
            return self.transform(x)
        except ValueError as exp:
            raise exp

    def transform(self, x):
        """

        Calculate the projections (scores) of the x data matrix. Similar to scikit-learn's TransformerMixin method.

        :param x: Data matrix to fit and project.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param kwargs transform_params: Optional keyword arguments to be passed to the transform method.
        :return: PCA projections (scores) corresponding to the samples in X.
        :rtype: numpy.ndarray, shape [n_samples, n_comps]
        :raise ValueError: If there are problems with the input or during model fitting.
        """
        try:
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.transform(xscaled)
            else:
                return self.pca_algorithm.transform(x)
        except ValueError as verr:
            raise verr

    def score(self, x, sample_weight=None):
        """

        Return the average log-likelihood of all samples. Same as the underlying score method from the scikit-learn
        PCA objects.

        :param x: Data matrix to score model on.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param numpy.ndarray sample_weight: Optional sample weights during scoring.
        :return: Average log-likelihood over all samples.
        :rtype: float
        :raises ValueError: if the data matrix x provided is invalid.
        """
        try:
            # Not all sklearn pca objects have a "score" method...
            score_method = getattr(self.pca_algorithm, "score", None)
            if not callable(score_method):
                raise NotImplementedError
            # Scaling check for consistency
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.score(xscaled, sample_weight)
            else:
                return self.pca_algorithm.score(x, sample_weight)
        except ValueError as verr:
            raise verr

    def inverse_transform(self, scores):
        """

        Transform scores to the original data space using the principal component loadings.
        Similar to scikit-learn's default TransformerMixin method.

        :param scores: The projections (scores) to be converted back to the original data space.
        :type scores: numpy.ndarray, shape [n_samples, n_comps]
        :return: Data matrix in the original data space.
        :rtype: numpy.ndarray, shape [n_samples, n_features]
        :raises ValueError: If the dimensions of score mismatch the number of components in the model.
        """
        # Scaling check for consistency
        if self.scaler is not None:
            xinv_prescaled = self.pca_algorithm.inverse_transform(scores)
            xinv = self.scaler.inverse_transform(xinv_prescaled)
            return xinv
        else:
            return self.pca_algorithm.inverse_transform(scores)

    @property
    def ncomps(self):
        try:
            return self._ncomps
        except AttributeError as atre:
            raise atre

    @ncomps.setter
    def ncomps(self, ncomps=1):
        """

        Setter for number of components.

        :param int ncomps: Number of components to use in the model.
        :raise AttributeError: If there is a problem changing the number of components and resetting the model.
        """
        # To ensure changing number of components effectively resets the model
        try:
            self._ncomps = ncomps
            self.pca_algorithm = clone(self.pca_algorithm, safe=True)
            self.pca_algorithm.n_components = ncomps
            self.modelParameters = None
            self.loadings = None
            self.scores = None
            self.cvParameters = None
            return None
        except AttributeError as atre:
            raise atre

    @property
    def scaler(self):
        try:
            return self._scaler
        except AttributeError as atre:
            raise atre

    @scaler.setter
    def scaler(self, scaler):
        """

        Setter for the model scaler.

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

            self._scaler = scaler
            self.pca_algorithm = clone(self.pca_algorithm, safe=True)
            self.modelParameters = None
            self.loadings = None
            self.scores = None
            self.cvParameters = None
            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    def hotelling_T2(self, comps=None, alpha=0.05):
        """

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        :param list comps:
        :param float alpha: Significance level
        :return: The Hotelling T2 ellipsoid radii at vertex
        :rtype: numpy.ndarray
        :raise AtributeError: If the model is not fitted
        :raise ValueError: If the components requested are higher than the number of components in the model
        :raise TypeError: If comps is not None or list/numpy 1d array and alpha a float
        """

        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")
            nsamples = self.scores.shape[0]
            if comps is None:
                ncomps = self.ncomps
                ellips = self.scores[:, range(self.ncomps)] ** 2
                ellips = 1 / nsamples * (ellips.sum(0))
            else:
                ncomps = len(comps)
                ellips = self.scores[:, comps] ** 2
                ellips = 1 / nsamples * (ellips.sum(0))

            # F stat
            fs = (nsamples - 1) / nsamples * ncomps * (nsamples ** 2 - 1) / (nsamples * (nsamples - ncomps))
            fs = fs * st.f.ppf(1-alpha, ncomps, nsamples - ncomps)

            hoteling_t2 = list()
            for comp in range(ncomps):
                hoteling_t2.append(np.sqrt((fs * ellips[comp])))

            return np.array(hoteling_t2)

        except AttributeError as atre:
            raise atre
        except ValueError as valerr:
            raise valerr
        except TypeError as typerr:
            raise typerr

    def _residual_ssx(self, x):
        """

        :param x: Data matrix [n samples, m variables]
        :return: The residual Sum of Squares per sample
        """
        pred_scores = self.transform(x)

        x_reconstructed = self.scaler.transform(self.inverse_transform(pred_scores))
        xscaled = self.scaler.transform(x)
        residuals = np.sum((xscaled - x_reconstructed)**2, axis=1)
        return residuals

    def x_residuals(self, x, scale=True):
        """

        :param x: data matrix [n samples, m variables]
        :param scale: Return the residuals in the scale the model is using or in the raw data scale
        :return: X matrix model residuals
        """
        pred_scores = self.transform(x)
        x_reconstructed = self.scaler.transform(self.inverse_transform(pred_scores))
        xscaled = self.scaler.transform(x)

        x_residuals = np.sum((xscaled - x_reconstructed)**2, axis=1)
        if scale:
            x_residuals = self.scaler.inverse_transform(x_residuals)

        return x_residuals

    def dmodx(self, x):
        """

        Normalised DmodX measure

        :param x: data matrix [n samples, m variables]
        :return: The Normalised DmodX measure for each sample
        """
        resids_ssx = self._residual_ssx(x)
        s = np.sqrt(resids_ssx/(self.loadings.shape[1] - self.ncomps))
        dmodx = np.sqrt((s/self.modelParameters['S0'])**2)
        return dmodx

    def leverages(self):
        """

        Calculate the leverages for each observation

        :return: The leverage (H) for each observation
        :rtype: numpy.ndarray
        """
        return np.diag(np.dot(self.scores, np.dot(np.linalg.inv(np.dot(self.scores.T, self.scores)), self.scores.T)))

    def cross_validation(self, x, cv_method=KFold(7, shuffle=True), outputdist=False):
        """

        Cross-validation method for the model. Calculates cross-validated estimates for Q2X and other
        model parameters using row-wise cross validation.

        :param x: Data matrix.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param cv_method: An instance of a scikit-learn CrossValidator object.
        :type cv_method: BaseCrossValidator
        :param bool outputdist: Output the whole distribution for the cross validated parameters.
        Useful when using ShuffleSplit or CrossValidators other than KFold.
        :return: Adds a dictionary cvParameters to the object, containing the cross validation results
        :rtype: dict
        :raise TypeError: If the cv_method passed is not a scikit-learn CrossValidator object.
        :raise ValueError: If the x data matrix is invalid.
        """

        try:

            if not (isinstance(cv_method, BaseCrossValidator) or isinstance(cv_method, BaseShuffleSplit)):
                raise TypeError("Scikit-learn cross-validation object please")

            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False or self.loadings is None:
                self.fit(x)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            cv_pipeline = deepcopy(self)

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            total_press = 0
            # Calculate Sum of Squares SS in whole dataset
            ss = np.sum((cv_pipeline.scaler.transform(x)) ** 2)
            # Initialise list for loadings and for the VarianceExplained in the test set values
            # Check if model has loadings, as in case of kernelPCA these are not available
            if hasattr(self.pca_algorithm, 'components_'):
                loadings = []

            # cv_varexplained_training is a list containing lists with the SingularValue/Variance Explained metric
            # as obtained in the training set during fitting.
            # cv_varexplained_test is a single R2X measure obtained from using the
            # model fitted with the training set in the test set.
            cv_varexplained_training = []
            cv_varexplained_test = []

            # Performs Row/Observation-Wise CV - Faster computationally, but has some limitations
            # See Bro R. et al, Cross-validation of component models: A critical look at current methods,
            # Analytical and Bioanalytical Chemistry 2008
            for xtrain, xtest in cv_method.split(x):
                cv_pipeline.fit(x[xtrain, :])
                # Calculate R2/Variance Explained in test set
                # To calculate an R2X in the test set

                xtest_scaled = cv_pipeline.scaler.transform(x[xtest, :])

                tss = np.sum((xtest_scaled) ** 2)
                # Append the var explained in training set for this round and loadings for this round
                cv_varexplained_training.append(cv_pipeline.pca_algorithm.explained_variance_ratio_)
                if hasattr(self.pca_algorithm, 'components_'):
                    loadings.append(cv_pipeline.loadings)

                # RSS for row wise cross-validation
                pred_scores = cv_pipeline.transform(x[xtest, :])
                pred_x = cv_pipeline.scaler.transform(cv_pipeline.inverse_transform(pred_scores))
                rss = np.sum(np.square(xtest_scaled - pred_x))
                total_press += rss
                cv_varexplained_test.append(1 - (rss / tss))

            # Create matrices for each component loading containing the cv values in each round
            # nrows = nrounds, ncolumns = n_variables
            # Check that the PCA model has loadings
            if hasattr(self.pca_algorithm, 'components_'):
                cv_loads = []
                for comp in range(0, self.ncomps):
                    cv_loads.append(np.array([x[comp] for x in loadings]))

                # Align loadings due to sign indeterminacy.
                # The solution followed here is to select the sign that gives a more similar profile to the
                # Loadings calculated with the whole data.
                for cvround in range(0, cv_method.n_splits):
                    for currload in range(0, self.ncomps):
                        choice = np.argmin(np.array([np.sum(np.abs(self.loadings - cv_loads[currload][cvround, :])),
                                                     np.sum(
                                                         np.abs(self.loadings - cv_loads[currload][cvround, :] * -1))]))
                        if choice == 1:
                            cv_loads[currload][cvround, :] = -1 * cv_loads[currload][cvround, :]

            # Calculate total sum of squares
            # Q^2X
            q_squared = 1 - (total_press / ss)
            # Assemble the dictionary and data matrices

            self.cvParameters = {'Mean_VarExpRatio_Training': np.array(cv_varexplained_training).mean(axis=0),
                                 'Stdev_VarExpRatio_Training': np.array(cv_varexplained_training).std(axis=0),
                                 'Mean_VarExp_Test': np.mean(cv_varexplained_test),
                                 'Stdev_VarExp_Test': np.std(cv_varexplained_test),
                                 'Q2': q_squared}

            if outputdist is True:
                self.cvParameters['CV_VarExpRatio_Training'] = cv_varexplained_training
                self.cvParameters['CV_VarExp_Test'] = cv_varexplained_test
            # Check that the PCA model has loadings
            if hasattr(self.pca_algorithm, 'components_'):
                self.cvParameters['Mean_Loadings'] = [np.mean(x, 0) for x in cv_loads]
                self.cvParameters['Stdev_Loadings'] = [np.std(x, 0) for x in cv_loads]
                if outputdist is True:
                    self.cvParameters['CV_Loadings'] = cv_loads
            return None

        except TypeError as terp:
            raise terp
        except ValueError as verr:
            raise verr

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
                comps = range(self.scores.shape[1])
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

    def plot_scores(self, comps=[0, 1], color=None, discrete=False, plot_title=None, label_outliers=False):
        """

        Score plot figure wth an Hotelling T2.

        :param comps: Components to use in the 2D plot
        :param color: Variable used to color points
        :return: Score plot figure
        """
        try:
            fig, ax = plt.subplots()

            # Use a constant color if no color argument is passed

            t2 = self.hotelling_T2(alpha=0.05, comps=comps)
            outlier_idx = np.where(((self.scores[:, comps] ** 2) / t2 ** 2).sum(axis=1) > 1)[0]

            if len(comps) == 1:
                x_coord = np.arange(0, self.scores.shape[0])
                y_coord = self.scores[:, comps[0]]
            else:
                x_coord = self.scores[:, comps[0]]
                y_coord = self.scores[:, comps[1]]

            if color is None:
                ax.scatter(x_coord, y_coord)
                #ax.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
                #            marker='x', s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
            else:
                if discrete is False:
                    cmap = cm.jet
                    cnorm = Normalize(vmin=min(color), vmax=max(color))

                    ax.scatter(x_coord, y_coord, c=color, cmap=cmap, norm=cnorm)
                    #ax.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
                    #            c=color[outlier_idx], cmap=cmap, norm=cnorm, marker='x',
                    #            s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
                    ax.colorbar()
                else:
                    cmap = cm.Set1
                    subtypes = np.unique(color)
                    for subtype in subtypes:
                        subset_index = np.where(color == subtype)
                        ax.scatter(x_coord[subset_index], y_coord[subset_index],
                                    c=cmap(subtype), label=subtype)
                    ax.legend()
                    #ax.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
                    #            c=color[outlier_idx], cmap=cmap, marker='x',
                    #            s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
            if label_outliers:
                for outlier in outlier_idx:
                    ax.annotate(outlier, (x_coord[outlier] + x_coord[outlier]*0.05, y_coord[outlier] + y_coord[outlier]*0.05))

            if len(comps) == 2:
                angle = np.arange(-np.pi, np.pi, 0.01)
                x = t2[0] * np.cos(angle)
                y = t2[1] * np.sin(angle)
                ax.axhline(c='k')
                ax.axvline(c='k')
                ax.plot(x, y, c='k')

                xmin = np.minimum(min(x_coord), np.min(x))
                xmax = np.maximum(max(x_coord), np.max(x))
                ymin = np.minimum(min(y_coord), np.min(y))
                ymax = np.maximum(max(y_coord), np.max(y))

                #axes = plt.gca()
                ax.set_xlim([(xmin + (0.2 * xmin)), xmax + (0.2 * xmax)])
                ax.set_ylim([(ymin + (0.2 * ymin)), ymax + (0.2 * ymax)])
            else:
                ax.axhline(y=t2, c='k', ls='--')
                ax.axhline(y=-t2, c='k', ls='--')
                ax.legend(['Hotelling $T^{2}$ 95% limit'])

        except (ValueError, IndexError) as verr:
            print("The number of components to plot must not exceed 2 and the component choice cannot "
                  "exceed the number of components in the model")
            raise Exception

        if plot_title is None:
            fig.suptitle("PCA score plot")
        else:
            fig.suptitle(plot_title)

        if len(comps) == 1:
            ax.set_xlabel("PC[{0}] - Variance Explained : {1:.2f} %".format((comps[0] + 1), self.modelParameters['VarExpRatio']*100))
        else:
            ax.set_xlabel("PC[{0}] - Variance Explained : {1:.2f} %".format((comps[0] + 1), self.modelParameters['VarExpRatio'][comps[0]]*100))
            ax.set_ylabel("PC[{0}] - Variance Explained : {1:.2f} %".format((comps[1] + 1), self.modelParameters['VarExpRatio'][comps[1]]*100))
        plt.show()
        return ax

    def scree_plot(self, x, total_comps=5, cv_method=KFold(7, shuffle=True)):
        """

        Plot of the R2X and Q2X per number of component to aid in the selection of the component number.

        :param x: Data matrix [n samples, m variables]
        :param total_comps: Maximum number of components to fit
        :param cv_method: scikit-learn Base Cross-Validator to use
        :return: Figure with R2X and Q2X Goodness of fit metrics per component
        """
        fig, ax = plt.subplots()
        models = list()

        for ncomps in range(1, total_comps + 1):
            currmodel = deepcopy(self)
            currmodel.ncomps = ncomps
            currmodel.fit(x)
            currmodel.cross_validation(x, outputdist=False, cv_method=cv_method)
            models.append(currmodel)

        q2 = np.array([x.cvParameters['Q2'] for x in models])
        r2 = np.array([x.modelParameters['R2X'] for x in models])

        ax.bar([x - 0.1 for x in range(1, total_comps + 1)], height=r2, width=0.2)
        ax.bar([x + 0.1 for x in range(1, total_comps + 1)], height=q2, width=0.2)
        ax.legend(['R2', 'Q2'])
        ax.set_xlabel("Number of components")
        ax.set_ylabel("R2/Q2X")

        # Specific case where n comps = 2
        if len(q2) == 2:
            plateau = np.min(np.where(np.diff(q2)/q2[0] < 0.05)[0])
        else:
            percent_cutoff = np.where(np.diff(q2) / q2[0:-1] < 0.05)[0]
            if percent_cutoff.size == 0:
                print("Consider exploring a higher level of components")
            else:
                plateau = np.min(percent_cutoff)
                ax.vlines(x= (plateau + 1), ymin=0, ymax=1, colors='red', linestyles ='dashed')
                print("Q2X measure stabilizes (increase of less than 5% of previous value or decrease) "
                      "at component {0}".format(plateau + 1))
        plt.show()

        return ax

    def repeated_cv(self, x, total_comps=7, repeats=15, cv_method=KFold(7, shuffle=True)):
        """

        Perform repeated cross-validation and plot Q2X values and their distribution (violin plot) per component
        number to help select the appropriate number of components.

        :param x: Data matrix [n samples, m variables]
        :param total_comps: Maximum number of components to fit
        :param repeats: Number of CV procedure repeats
        :param cv_method: scikit-learn Base Cross-Validator to use
        :return: Violin plot with Q2X values and distribution per component number.
        """

        q2x = np.zeros((total_comps, repeats))

        for ncomps in range(1, total_comps + 1):
            for rep in range(repeats):
                currmodel = deepcopy(self)
                currmodel.ncomps = ncomps
                currmodel.fit(x)
                currmodel.cross_validation(x, cv_method=cv_method, outputdist=False)
                q2x[ncomps - 1, rep] = currmodel.cvParameters['Q2']

        fig, ax = plt.subplots()
        ax = sns.violinplot(data=q2x.T, palette="Set1")
        ax = sns.swarmplot(data=q2x.T, edgecolor="black", color='black')
        ax.set_xticklabels(range(1, total_comps + 1))
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Q2X")
        plt.show()

        return q2x, ax

    def plot_loadings(self, component=1, bar=False, sigma=2, x=None):
        """
        Loading plot figure for the selected component. With uncertainty estimation if the cross validation method
        has been called before.

        :param float component: Component to plot loadings
        :param boolean bar: Whether to use line or bar plot
        :param float sigma: Multiple of standard deviation to plot
        :return: Loading plot figure
        """
        # Adjust the indexing so user can refer to component 1 as component 1 instead of 0
        component -= 1
        fig, ax = plt.subplots()

        if x is None:
            x_to_fill = range(self.loadings[component, :].size)
        else:
            x_to_fill = x

        # For "spectrum/continuous like plotting"
        if bar is False:
            if x is None:
                ax.plot(self.loadings[component, :])
            else:
                ax.plot(x, self.loadings[component, :])

            if self.cvParameters is not None:
                ax.fill_between(x_to_fill,
                self.cvParameters['Mean_Loadings'][component] - sigma*self.cvParameters['Stdev_Loadings'][component],
                self.cvParameters['Mean_Loadings'][component] + sigma*self.cvParameters['Stdev_Loadings'][component],
                alpha=0.2, color='red')

        # To use with barplots for other types of data
        else:
            if self.cvParameters is not None:
                ax.errorbar(x_to_fill,
                             height=self.cvParameters['Mean_Loadings'][:, component],
                             yerr=2 * self.cvParameters['Stdev_Loadings'][:, component],
                             width=0.2)
            else:
                ax.bar(x_to_fill, height=self.loadings[component, :], width=0.2)

        ax.set_xlabel("Variable No")
        ax.set_ylabel("Loading for PC{0}".format((component + 1)))
        plt.show()

        return ax

    def plot_dmodx(self, x, label_outliers=False, alpha=0.05):
        """

        Plot a figure with DmodX values and the F-statistic critical line.

        :param numpy.ndarray x: Data matrix [n samples, m variables]
        :param float alpha: Significance level
        :return: Plot with DmodX values and critical line
        """

        try:
            dmodx = self.dmodx(x)
            # Degrees of freedom for the PCA model (denominator in F-stat) calculated as suggested in
            # Faber, Nicolaas (Klaas) M., Degrees of freedom for the residuals of a
            # principal component analysis - A clarification, Chemometrics and Intelligent Laboratory Systems 2008
            dcrit = st.f.ppf(1-alpha, x.shape[1] - self.ncomps - 1, (x.shape[0] - self.ncomps - 1)*(x.shape[1] - self.ncomps))
            outlier_idx = self.outlier(x, measure='DmodX')
            fig, ax = plt.subplots()
            x_axis = np.array([x for x in range(x.shape[0])])
            ax.plot(x_axis, dmodx, 'o')
            ax.plot(x_axis[outlier_idx], dmodx[outlier_idx], 'rx')

            if label_outliers:
                for outlier in outlier_idx:
                    ax.annotate(outlier, (
                    x_axis[outlier] + x_axis[outlier] * 0.05, dmodx[outlier] + dmodx[outlier] * 0.05))

            ax.set_xlabel('Sample Index')
            ax.set_ylabel('DmodX')
            ax.hlines(dcrit, xmin=0, xmax= x.shape[0], color='r', linestyles='--')
            plt.show()
            return ax
        except TypeError as terr:
            raise terr
        except ValueError as verr:
            raise verr

    def plot_leverages(self):
        """
        Leverage (h) per observation, with a red line plotted at y = 1/Number of samples (expected
        :return: Plot with observation leverages (h)
        """
        fig, ax = plt.subplots()
        lev = self.leverages()
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Leverage')
        ax.bar(left=range(lev.size), height=lev)
        ax.hlines(y=1/lev.size, xmin=0, xmax=lev.size, colors='r', linestyles='--')
        plt.show()
        return ax

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def plotScoresInteractive(self, components=[1, 2], colour=None, label=None, discrete=True, alpha=0.05):
        """
        Interactively visualise PCA scores (coloured by a given sampleMetadata field, and for a given pair of components) with plotly, provides tooltips to allow identification of samples.

        :param Dataset dataTrue: Dataset
        :param PCA object pcaModel: PCA model object (scikit-learn based)
        :param str colourBy: **sampleMetadata** field name to of which values to colour samples by
        :param list components: List of two integers, components to plot
        :param float alpha: Significance value for plotting Hotellings ellipse
        :param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks; must match between data and pcaModel
        """

        values = self.scores
        ns, nc = values.shape
        sampleMetadata = dataTrue.sampleMetadata.copy()

        components = [component - 1 for i, component in
                      enumerate(components)]  # Reduce components by one (account for python indexing)


        classes = sampleMetadata[colourBy]

        if label is None:
            hovertext = sampleMetadata['Sample File Name'].str.cat(classes.astype(str),
                                                                   sep='; ' + colourBy + ': ')  # Save text to show in tooltips
        else:
            hovertext = label

        data = []

        # Ensure all values in column have the same type
        if discrete:
            # list of all types in column; and set of unique types
            mylist = list(type(classes[i]) for i in range(ns))
            myset = set(mylist)

        # else if mixed type convert to string
        if len(myset) > 1:
            classes = classes.astype(str)

        # Plot NaN values in gray
        plotnans = classes.isnull().values
        if sum(plotnans != 0):
            NaNplot = go.Scattergl(
                x=values[plotnans == True, components[0]],
                y=values[plotnans == True, components[1]],
                mode='markers',
                marker=dict(
                    color='rgb(180, 180, 180)',
                    symbol='circle',
                ),
                text=hovertext[plotnans == True],
                hoverinfo='text',
                showlegend=False
            )
            data.append(NaNplot)

        # Plot numeric values with a colorbar
        if discrete is False:
            CLASSplot = go.Scattergl(
                x=values[plotnans == False, components[0]],
                y=values[plotnans == False, components[1]],
                mode='markers',
                marker=dict(
                    colorscale='Portland',
                    color=classes[plotnans == False],
                    symbol='circle',
                    showscale=True
                ),
                text=hovertext[plotnans == False],
                hoverinfo='text',
                showlegend=False
            )

        # Plot categorical values by unique groups
        else:
            uniq, indices = numpy.unique(classes, return_inverse=True)
            CLASSplot = go.Scattergl(
                x=values[plotnans == False, components[0]],
                y=values[plotnans == False, components[1]],
                mode='markers',
                marker=dict(
                    colorscale='Portland',
                    color=indices[plotnans == False],
                    symbol='circle',
                ),
                text=hovertext[plotnans == False],
                hoverinfo='text',
                showlegend=False
            )

        data.append(CLASSplot)

        hotelling_ellipse = self.hotelling_T2(comps=numpy.array([components[0], components[1]]), alpha=alpha)

        layout = {
            'shapes': [
                {
                    'type': 'circle',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': 0 - hotelling_ellipse[0],
                    'y0': 0 - hotelling_ellipse[1],
                    'x1': 0 + hotelling_ellipse[0],
                    'y1': 0 + hotelling_ellipse[1],
                }
            ],
            'xaxis': dict(
                title='PC' + str(components[0] + 1) + ' (' + '{0:.2f}'.format(
                    self.modelParameters['VarExpRatio'][components[0]] * 100) + '%)'
            ),
            'yaxis': dict(
                title='PC' + str(components[1] + 1) + ' (' + '{0:.2f}'.format(
                    self.modelParameters['VarExpRatio'][components[1]] * 100) + '%)'
            ),
            'title': 'Coloured by ' + colour,
            'legend': dict(
                yanchor='middle',
                xanchor='right'
            ),
            'hovermode': 'closest'
        }

        figure = go.Figure(data=data, layout=layout)

        return figure
