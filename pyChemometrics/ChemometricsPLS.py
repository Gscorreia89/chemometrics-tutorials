from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._split import BaseShuffleSplit
from .ChemometricsScaler import ChemometricsScaler
import scipy.stats as st
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from .plotting_utils import _lineplots, _barplots, _scatterplots
import seaborn as sns
__author__ = 'gd2212'


class ChemometricsPLS(BaseEstimator, RegressorMixin, TransformerMixin):
    """

    ChemometricsPLS object - Wrapper for sklearn.cross_decomposition PLS algorithms, with tailored methods
    for Chemometric Data analysis.

    :param int n_components: Number of PLS components desired.
    :param sklearn._PLS pls_algorithm: Scikit-learn PLS algorithm to use - PLSRegression or PLSCanonical are supported.
    :param xscaler: Scaler object for X data matrix.
    :type xscaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None.
    :param yscaler: Scaler object for the Y data vector/matrix.
    :type yscaler: ChemometricsScaler object, scaling/preprocessing objects from scikit-learn or None.
    :param kwargs pls_type_kwargs: Keyword arguments to be passed during initialization of pls_algorithm.
    :raise TypeError: If the pca_algorithm or scaler objects are not of the right class.
    """

    """
    This object is designed to fit flexibly both PLSRegression with one or multiple Y and PLSCanonical, both
    with either NIPALS or SVD. PLS-SVD doesn't calculate the same type of model parameters, and should
    not be used with this object.
    For PLSRegression/PLS1/PLS2 and PLSCanonical/PLS-C2A/PLS-W2A, the actual components
    found may differ (depending on type of deflation, etc), and this has to be taken into consideration,
    but the actual nomenclature/definitions should be the "same".
    Nomenclature is as follows:
    X - T Scores - Projections of X, called T
    Y - U Scores - Projections of Y, called U
    X - Loadings P - Vector/multivariate directions associated with T on X are called P (equivalent to PCA)
    Y - Loadings Q - Vector/multivariate directions associated with U on Y are called q
    X - Weights W - Weights/directions of maximum covariance with Y of the X block are called W
    Y - Weights C - Weights/directions of maximum covariance with X of the Y block block are called C
    X - Rotations W*/Ws/R - The rotation of X variables to LV space pinv(WP')W
    Y - Rotations C*/Cs - The rotation of Y variables to LV space pinv(CQ')C
    T = X W(P'W)^-1 = XW* (W* : p x k matrix)
    U = Y C(Q'C)^-1 = YC* (C* : q x k matrix)
    Loadings and weights after the first component do not represent
    the original variables. The SIMPLS-style (similar interpretation but not the same Rotations that would be obtained from 
    using the SIMPLS algorithm) W*/Ws and C*/Cs act as weight vectors
    which relate to the original X and Y variables, and not to their deflated versions.
    For more information see Sijmen de Jong, "SIMPLS: an alternative approach to partial least squares regression", Chemometrics
    and Intelligent Laboratory Systems 1992
    "Inner" relation regression coefficients of T b_t: U = Tb_t
    "Inner" relation regression coefficients of U b_U: T = Ub_u
    These are obtained by regressing the U's and T's, applying standard linear regression to them.
    B = pinv(X'X)X'Y
    b_t = pinv(T'T)T'U
    b_u = pinv(U'U)U'T
    or in a form usually seen in PLS NIPALS algorithms: b_t are the betas from regressing T on U - t'u/u'u
    and b_u are the betas from regressing U on T - u't/t't

    In summary, there are various ways to approach the model. Following a general nomenclature applicable
    for both single and block Y:
    For predictions, the model assumes the Latent variable formulation and uses an "inner relation"
    between the latent variable projections, where U = Tb_t and T = Ub_u.
    Therefore, prediction using the so-called "mixed relations" (relate T with U and subsequently Y/relate
    U with T and subsequently X), works through the following formulas
    Y = T*b_t*C' + G
    X = U*b_u*W' + H
    The b_u and b_s are effectively "regression coefficients" between the latent variable scores
    
    In parallel, we can think in terms of "outer relations", data decompositions or linear approximations to
    the original data blocks, similar to PCA components
    Y = UQ' + F
    X = TP' + E
    For PLS regression with single y, Y = UC' + F = Y = UQ' + F, due to Q = C, but not necessarily true for
    multi Y, so Q' is used here. Notice that this formula cannot be used directly to
    predict Y from X and vice-versa, the inner relation regression using latent variable scores is necessary.
    
    Finally, assuming PLSRegression (single or multi Y, but asymmetric deflation):
    The PLS model can be approached from a multivariate regression/regularized regression point of view,
    where Y is related to the original X variables, through regression coefficients Beta,
    bypassing the latent variable definition and concepts.
    Y = XBQ', Y = XB, where B are the regression coefficients and B = W*Q' (the W*/ws is the SIMPLS-like R rotation,
    the x_rotation in sklearn default PLS algorithms).
    The Betas (regression coefficients) obtained in this manner directly relate the original X variables
    to the prediction of Y.
    
    This MLR (multivariate linear regression) approach to PLS has the advantage of exposing the PLS betas and PLS mechanism
    as a biased regression applying a degree of shrinkage, which decreases with the number of components
    all the way up to B(OLS), when Number of Components = number of variables/columns.
    
    See:
    Frank, Ildiko E. Friedman, Jerome H., A Statistical View of Some Chemometrics Regression Tools, 1993
    de Jong, PLS shrinks, Journal of Chemometrics, 1995 
    Nicole Kramer, An Overview on the Shrinkage Properties of Partial Least Squares Regression, 
    Computational Statistics, 2007
    """

    def __init__(self, n_components=2, pls_algorithm=PLSRegression, x_scaler=ChemometricsScaler(), y_scaler=None,
                 **pls_type_kwargs):

        try:

            # Perform the check with is instance but avoid abstract base class runs.
            pls_algorithm = pls_algorithm(n_components, scale=False, **pls_type_kwargs)
            if not isinstance(pls_algorithm, (BaseEstimator, PLSRegression)):
                raise TypeError("Scikit-learn model please")
            if not (isinstance(x_scaler, TransformerMixin) or x_scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if not (isinstance(y_scaler, TransformerMixin) or y_scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            # 2 blocks of data = two scaling options
            if x_scaler is None:
                x_scaler = ChemometricsScaler(0, with_std=False)
                # Force scaling to false by default
            if y_scaler is None:
                y_scaler = ChemometricsScaler(0, with_std=False)

            self.pls_algorithm = pls_algorithm
            # Most initialized as None, before object is fitted...
            self.scores_t = None
            self.scores_u = None
            self.weights_w = None
            self.weights_c = None
            self.loadings_p = None
            self.loadings_q = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.b_u = None
            self.b_t = None
            self.beta_coeffs = None

            self.n_components = n_components
            self.x_scaler = x_scaler
            self.y_scaler = y_scaler
            self.cvParameters = None
            self.modelParameters = None
            self._isfitted = False

        except TypeError as terp:
            print(terp.args[0])

    @property
    def n_components(self):
        try:
            return self._n_components
        except AttributeError as atre:
            raise atre

    @n_components.setter
    def n_components(self, n_components=1):
        """

        Setter for number of components. Re-sets the model.

        :param int ncomps: Number of PLS components to use in the model.
        :raise AttributeError: If there is a problem changing the number of components and resetting the model.
        """
        # To ensure changing number of components effectively resets the model
        try:
            self._n_components = n_components
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.pls_algorithm.n_components = n_components
            self.loadings_p = None
            self.scores_t = None
            self.scores_u = None
            self.loadings_q = None
            self.weights_c = None
            self.weights_w = None
            self.rotations_cs = None
            self.rotations_ws = None
            self.cvParameters = None
            self.modelParameters = None
            self.b_t = None
            self.b_u = None
            self.beta_coeffs = None
            self._isfitted = False

            return None
        except AttributeError as atre:
            raise atre

    def fit(self, x, y, **fit_params):
        """

        Perform model fitting on the provided x and y data and calculate basic goodness-of-fit metrics.
        Similar to scikit-learn's BaseEstimator method.

        :param x: Data matrix to fit the PLS model.
        :type x: numpy.ndarray, shape [n_samples, n_features].
        :param y: Data matrix to fit the PLS model.
        :type y: numpy.ndarray, shape [n_samples, n_features].
        :param kwargs fit_params: Keyword arguments to be passed to the .fit() method of the core sklearn model.
        :raise ValueError: If any problem occurs during fitting.
        """
        try:
            # This scaling check is always performed to ensure running model with scaling or with scaling == None
            # always gives consistent results (the same type of data scale used fitting will be expected or returned
            # by all methods of the ChemometricsPLS object)
            # For no scaling, mean centering is performed nevertheless - sklearn objects
            # do this by default, this is solely to make everything ultra clear and to expose the
            # interface for potential future modification
            # Comply with the sklearn-scaler behaviour convention

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
            self.loadings_p = self.pls_algorithm.x_loadings_
            self.loadings_q = self.pls_algorithm.y_loadings_
            self.weights_w = self.pls_algorithm.x_weights_
            self.weights_c = self.pls_algorithm.y_weights_
            self.rotations_ws = self.pls_algorithm.x_rotations_
            # scikit learn sets the rotation, causing a discrepancy between the scores calculated during fitting and the transform method
            # for now, we calculate the rotation and override it: C* = pinv(CQ')C
            self.rotations_cs = np.dot(np.linalg.pinv(np.dot(self.weights_c, self.loadings_q.T)), self.weights_c)
            self.scores_t = self.pls_algorithm.x_scores_
            self.scores_u = self.pls_algorithm.y_scores_
            self.b_u = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_u.T, self.scores_u)), self.scores_u.T),
                              self.scores_t)
            self.b_t = np.dot(np.dot(np.linalg.pinv(np.dot(self.scores_t.T, self.scores_t)), self.scores_t.T),
                              self.scores_u)
            self.beta_coeffs = self.pls_algorithm.coef_
            # Needs to come here for the method shortcuts down the line to work...
            self._isfitted = True
            self.cvParameters = None
            # Calculate RSSy/RSSx, R2Y/R2X
            R2Y = ChemometricsPLS.score(self, x=x, y=y, block_to_score='y')
            R2X = ChemometricsPLS.score(self, x=x, y=y, block_to_score='x')

            self.modelParameters = {'R2Y': R2Y, 'R2X': R2X}

            resid_ssx = self._residual_ssx(x)
            s0 = np.sqrt(resid_ssx.sum() / ((self.scores_t.shape[0] - self.n_components - 1) * (x.shape[1] - self.n_components)))
            self.modelParameters['S0X'] = s0

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
                    xpred = np.dot(t, self.loadings_p.T)
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.inverse_transform(xpred)
                    else:
                        xscaled = xpred

                    return xscaled
                # If U is given, return T
                elif u is not None:
                    # Calculate Y from U - using Y = UQ'
                    ypred = np.dot(u, self.loadings_q.T)
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
                tssy = np.sum(np.square(yscaled))
                ypred = self.y_scaler.transform(ChemometricsPLS.predict(self, x, y=None))
                rssy = np.sum(np.square(yscaled - ypred))
                R2Y = 1 - (rssy / tssy)
                return R2Y
            # The prediction here of both X and Y is done using the other block of data only
            # so these R2s can be interpreted as as a "classic" R2, and not as a proportion of variance modelled
            # Here we use X = Ub_uW', as opposed to (X = TP').
            else:
                xscaled = deepcopy(self.x_scaler).fit_transform(x)
                # Calculate total sum of squares of X and Y for R2X and R2Y calculation
                xpred = self.x_scaler.transform(ChemometricsPLS.predict(self, x=None, y=y))
                tssx = np.sum(np.square(xscaled))
                rssx = np.sum(np.square(xscaled - xpred))
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
                    # Going through calculation of U and then X = Ub_uW'
                    u_scores = ChemometricsPLS.transform(self, x=None, y=y)
                    predicted = np.dot(np.dot(u_scores, self.b_u), self.weights_w.T)
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

    def VIP(self):
        """

        Output the Variable importance for projection metric (VIP). With the default values it is calculated
        using the x variable weights and the variance explained of y.

        Note: Code not adequate to obtain a VIP for each individual variable in the multi-Y case, as SSY should be changed
        so that it is calculated for each y and not for the whole Y matrix

        :param mode: The type of model parameter to use in calculating the VIP. Default value is weights (w), and other acceptable arguments are p, ws, cs, c and q.
        :type mode: str
        :param str direction: The data block to be used to calculated the model fit and regression sum of squares.
        :return numpy.ndarray VIP: The vector with the calculated VIP values.
        :rtype: numpy.ndarray, shape [n_features]
        :raise ValueError: If mode or direction is not a valid option.
        :raise AttributeError: Calling method without a fitted model.
        """
        try:
            # Code not really adequate for each Y variable in the multi-Y case - SSy should be changed so
            # that it is calculated for each y and not for the whole bloc
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")

            nvars = self.loadings_p.shape[0]

            SSYcomp = np.sum(self.scores_t ** 2, axis=0) * np.sum(self.scores_u ** 2, axis=0)

            vip = np.sqrt(np.sum(self.weights_w ** 2 * SSYcomp * nvars / SSYcomp.sum(), axis=1))

            return vip

        except AttributeError as atter:
            raise atter
        except ValueError as verr:
            raise verr

    def hotelling_T2(self, comps=[0, 1], alpha=0.05):
        """

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        :param list comps: List of components to calculate the Hotelling T2.
        :param float alpha: Significant level for the F statistic.
        :return: List with the Hotelling T2 ellipse radii
        :rtype: list
        :raise ValueError: If the dimensions request
        """
        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")

            nsamples = self.scores_t.shape[0]

            if comps is None:
                n_components = self.n_components
                ellips = self.scores_t[:, range(self.n_components)] ** 2
            else:
                n_components = len(comps)
                ellips = self.scores_t[:, comps] ** 2

            ellips = 1 / nsamples * (ellips.sum(0))

            # F stat
            a = (nsamples - 1) / nsamples * n_components * (nsamples ** 2 - 1) / (nsamples * (nsamples - n_components))
            a = a * st.f.ppf(1-alpha, n_components, nsamples - n_components)

            hoteling_t2 = list()
            for comp in range(n_components):
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
        s = np.sqrt(resids_ssx/(self.loadings_p.shape[0] - self.n_components))
        dmodx = np.sqrt((s/self.modelParameters['S0X'])**2)
        return dmodx

    def leverages(self, block):
        """
        Calculate the leverages for each observation
        :return:
        :rtype:
        """
        try:
            if block == 'X':
                return np.dot(self.scores_t, np.dot(np.linalg.inv(np.dot(self.scores_t.T, self.scores_t), self.scores_t.T)))
            elif block == 'Y':
                return np.dot(self.scores_u, np.dot(np.linalg.inv(np.dot(self.scores_u.T, self.scores_u), self.scores_u.T)))
            else:
                raise ValueError
        except ValueError as verr:
            raise ValueError('block option must be either X or Y')

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
                comps = range(self.scores_t.shape[1])
            if measure == 'T2':
                scores = self.transform(x)
                t2 = self.hotelling_T2(comps=comps)
                outlier_idx = np.where(((scores[:, comps] ** 2) / t2 ** 2).sum(axis=1) > 1)[0]
            elif measure == 'DmodX':
                dmodx = self.dmodx(x)
                dcrit = st.f.ppf(1 - alpha, x.shape[1] - self.n_components,
                                 (x.shape[0] - self.n_components - 1) * (x.shape[1] - self.n_components))
                outlier_idx = np.where(dmodx > dcrit)[0]
            else:
                print("Select T2 (Hotelling T2) or DmodX as outlier exclusion criteria")
            return outlier_idx
        except Exception as exp:
            raise exp

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

            # Initialize list structures to contain the fit
            cv_loadings_p = np.zeros((ncvrounds, x_nvars, self.n_components))
            cv_loadings_q = np.zeros((ncvrounds, y_nvars, self.n_components))
            cv_weights_w = np.zeros((ncvrounds, x_nvars, self.n_components))
            cv_weights_c = np.zeros((ncvrounds, y_nvars, self.n_components))
            cv_rotations_ws = np.zeros((ncvrounds, x_nvars, self.n_components))
            cv_rotations_cs = np.zeros((ncvrounds, y_nvars, self.n_components))
            cv_betacoefs = np.zeros((ncvrounds, x_nvars))
            cv_vipsw = np.zeros((ncvrounds, x_nvars))

            cv_train_scores_t = list()
            cv_train_scores_u = list()
            cv_test_scores_t = list()
            cv_test_scores_u = list()

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
                cv_rotations_ws[cvround, :, :] = cv_pipeline.rotations_ws
                cv_rotations_cs[cvround, :, :] = cv_pipeline.rotations_cs
                cv_betacoefs[cvround, :] = cv_pipeline.beta_coeffs.T
                cv_vipsw[cvround, :] = cv_pipeline.VIP()

            # Align model parameters to account for sign indeterminacy.
            # The criteria here used is to select the sign that gives a more similar profile (by L1 distance) to the loadings fitted
            # on the model fitted with the whole data. Any other parameter can be used, but since the loadings in X capture
            # the covariance structure in X data block, in theory they should have more pronounced features even in cases of
            # null X-Y association, making the sign flip more resilient.
            for cvround in range(0, ncvrounds):
                for currload in range(0, self.n_components):
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
                        cv_train_scores_t.append([*zip(train, -1 * cv_pipeline.scores_t)])
                        cv_train_scores_u.append([*zip(train, -1 * cv_pipeline.scores_u)])
                        cv_test_scores_t.append([*zip(test, -1 * cv_pipeline.scores_t)])
                        cv_test_scores_u.append([*zip(test, -1 * cv_pipeline.scores_u)])
                    else:
                        cv_train_scores_t.append([*zip(train, cv_pipeline.scores_t)])
                        cv_train_scores_u.append([*zip(train, cv_pipeline.scores_u)])
                        cv_test_scores_t.append([*zip(test, cv_pipeline.scores_t)])
                        cv_test_scores_u.append([*zip(test, cv_pipeline.scores_u)])

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

            # Means and standard deviations...
            # self.cvParameters['Mean_Scores_t'] = cv_scores_t.mean(0)
            # self.cvParameters['Stdev_Scores_t'] = cv_scores_t.std(0)
            # self.cvParameters['Mean_Scores_u'] = cv_scores_u.mean(0)
            # self.cvParameters['Stdev_Scores_u'] = cv_scores_u.std(0)
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

    def permutation_test(self, x, y, nperms=1000, cv_method=KFold(7, shuffle=True), **permtest_kwargs):
        """

        Permutation test for the classifier. Outputs permuted null distributions for model performance metrics (Q2X/Q2Y)
        and most model parameters.

        :param x: Data matrix to fit the PLS model.
        :type x: numpy.ndarray, shape [n_samples, n_features]
        :param y: Data matrix to fit the PLS model.
        :type y: numpy.ndarray, shape [n_samples, n_features]
        :param int nperms: Number of permutations to perform.
        :param cv_method: An instance of a scikit-learn CrossValidator object.
        :type cv_method: BaseCrossValidator or BaseShuffleSplit
        :param kwargs permtest_kwargs: Keyword arguments to be passed to the .fit() method during cross-validation and model fitting.
        :return: Permuted null distributions for model parameters and the permutation p-value for the Q2Y value.
        :rtype: dict
        """
        try:
            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False or self.loadings_p is None:
                self.fit(x, y, **permtest_kwargs)
            if self.cvParameters is None:
                self.cross_validation(x, y, cv_method=cv_method)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            permute_class = deepcopy(self)

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            if y.ndim > 1:
                y_nvars = y.shape[1]
            else:
                y_nvars = 1

            # Initialize data structures for permuted distributions
            perm_loadings_q = np.zeros((nperms, y_nvars, self.n_components))
            perm_loadings_p = np.zeros((nperms, x_nvars, self.n_components))
            perm_weights_c = np.zeros((nperms, y_nvars, self.n_components))
            perm_weights_w = np.zeros((nperms, x_nvars, self.n_components))
            perm_rotations_cs = np.zeros((nperms, y_nvars, self.n_components))
            perm_rotations_ws = np.zeros((nperms, x_nvars, self.n_components))
            perm_beta = np.zeros((nperms, x_nvars, y_nvars))
            perm_vipsw = np.zeros((nperms, x_nvars))

            permuted_R2Y = np.zeros(nperms)
            permuted_R2X = np.zeros(nperms)
            permuted_Q2Y = np.zeros(nperms)
            permuted_Q2X = np.zeros(nperms)
            permuted_R2Y_test = np.zeros(nperms)
            permuted_R2X_test = np.zeros(nperms)

            for permutation in range(0, nperms):
                # Copy original column order, shuffle array in place...
                perm_y = np.random.permutation(y)
                # ... Fit model and replace original data
                permute_class.fit(x, perm_y, **permtest_kwargs)
                permute_class.cross_validation(x, perm_y, cv_method=cv_method, **permtest_kwargs)
                permuted_R2Y[permutation] = permute_class.modelParameters['R2Y']
                permuted_R2X[permutation] = permute_class.modelParameters['R2X']
                permuted_Q2Y[permutation] = permute_class.cvParameters['Q2Y']
                permuted_Q2X[permutation] = permute_class.cvParameters['Q2X']

                # Store the loadings for each permutation component-wise
                perm_loadings_q[permutation, :, :] = permute_class.loadings_q
                perm_loadings_p[permutation, :, :] = permute_class.loadings_p
                perm_weights_c[permutation, :, :] = permute_class.weights_c
                perm_weights_w[permutation, :, :] = permute_class.weights_w
                perm_rotations_cs[permutation, :, :] = permute_class.rotations_cs
                perm_rotations_ws[permutation, :, :] = permute_class.rotations_ws
                perm_beta[permutation, :, :] = permute_class.beta_coeffs
                perm_vipsw[permutation, :] = permute_class.VIP()
            # Align model parameters due to sign indeterminacy.
            # Solution provided is to select the sign that gives a more similar profile to the
            # Loadings calculated with the whole data.
            for perm_round in range(0, nperms):
                for currload in range(0, self.n_components):
                    # evaluate based on loadings _p
                    choice = np.argmin(np.array(
                        [np.sum(np.abs(self.loadings_p[:, currload] - perm_loadings_p[perm_round, :, currload])),
                         np.sum(np.abs(self.loadings_p[:, currload] - perm_loadings_p[perm_round, :, currload] * -1))]))
                    if choice == 1:
                        perm_loadings_p[perm_round, :, currload] = -1 * perm_loadings_p[perm_round, :, currload]
                        perm_loadings_q[perm_round, :, currload] = -1 * perm_loadings_q[perm_round, :, currload]
                        perm_weights_w[perm_round, :, currload] = -1 * perm_weights_w[perm_round, :, currload]
                        perm_weights_c[perm_round, :, currload] = -1 * perm_weights_c[perm_round, :, currload]
                        perm_rotations_ws[perm_round, :, currload] = -1 * perm_rotations_ws[perm_round, :, currload]
                        perm_rotations_cs[perm_round, :, currload] = -1 * perm_rotations_cs[perm_round, :, currload]

            # Pack everything into a nice data structure and return
            # Calculate p-value for Q2Y as well
            permutationTest = dict()
            permutationTest['R2Y'] = permuted_R2Y
            permutationTest['R2X'] = permuted_R2X
            permutationTest['Q2Y'] = permuted_Q2Y
            permutationTest['Q2X'] = permuted_Q2X
            permutationTest['R2Y_Test'] = permuted_R2Y_test
            permutationTest['R2X_Test'] = permuted_R2X_test
            permutationTest['Loadings_p'] = perm_loadings_p
            permutationTest['Loadings_q'] = perm_loadings_q
            permutationTest['Weights_c'] = perm_weights_c
            permutationTest['Weights_w'] = perm_weights_w
            permutationTest['Rotations_ws'] = perm_rotations_ws
            permutationTest['Rotations_cs'] = perm_rotations_cs
            permutationTest['Beta'] = perm_beta
            permutationTest['VIPw'] = perm_vipsw

            obs_q2y = self.cvParameters['Q2Y']
            pvals = dict()
            pvals['Q2Y'] = (len(np.where(permuted_Q2Y >= obs_q2y)) + 1) / (nperms + 1)
            obs_r2y = self.cvParameters['R2Y_Test']
            pvals['R2Y_Test'] = (len(np.where(permuted_R2Y_test >= obs_r2y)) + 1) / (nperms + 1)
            return permutationTest, pvals

        except ValueError as exp:
            raise exp

    def _residual_ssx(self, x):
        """

        :param x: Data matrix [n samples, m variables]
        :return: The residual Sum of Squares per sample
        """
        pred_scores = self.transform(x)

        x_reconstructed = self.x_scaler.transform(self.inverse_transform(pred_scores))
        xscaled = self.x_scaler.transform(x)
        residuals = np.sum(np.square(xscaled - x_reconstructed), axis=1)
        return residuals

    def plot_scores(self, comps=[0, 1], color=None, discrete=False, label_outliers=False, plot_title=None):
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
            outlier_idx = np.where(((self.scores_t[:, comps] ** 2) / t2 ** 2).sum(axis=1) > 1)[0]

            if len(comps) == 1:
                x_coord = np.arange(0, self.scores_t.shape[0])
                y_coord = self.scores_t[:, comps[0]]
            else:
                x_coord = self.scores_t[:, comps[0]]
                y_coord = self.scores_t[:, comps[1]]

            if color is None:
                ax.scatter(x_coord, y_coord)
                #ax.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
                #            marker='x', s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
            else:
                if discrete is False:
                    cmap = cm.jet
                    cnorm = Normalize(vmin=min(color), vmax=max(color))

                    scatter = ax.scatter(x_coord, y_coord, c=color, cmap=cmap, norm=cnorm)
                    #ax.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
                    #            c=color[outlier_idx], cmap=cmap, norm=cnorm, marker='x',
                    #            s=1.5 * mpl.rcParams['lines.markersize'] ** 2)
                    fig.colorbar(scatter)
                else:
                    cmap = cm.Set1
                    subtypes = np.unique(color)
                    for subtype in subtypes:
                        subset_index = np.where(color == subtype)
                        ax.scatter(x_coord[subset_index], y_coord[subset_index],
                                    c=cmap(subtype), label=subtype)
                    ax.legend()
                    #plt.scatter(x_coord[outlier_idx], y_coord[outlier_idx],
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

                # axes = plt.gca()
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
            fig.suptitle("PLS score plot")
        else:
            fig.suptitle(plot_title)

        if len(comps) == 1:
            ax.set_xlabel("T[{0}]".format((comps[0] + 1)))
        else:
            ax.set_xlabel("T[{0}]".format((comps[0] + 1)))
            ax.set_ylabel("T[{0}]".format((comps[1] + 1)))
        plt.show()
        return ax

    def scree_plot(self, x, y, total_comps=5):
        """

        :param x:
        :param y:
        :param total_comps:
        :return:
        """
        fig, ax = plt.subplots()

        models = list()
        for n_components in range(1, total_comps + 1):
            currmodel = deepcopy(self)
            currmodel.n_components = n_components
            currmodel.fit(x, y)
            currmodel.cross_validation(x, y)
            models.append(currmodel)
            q2 = np.array([x.cvParameters['PLS']['Q2Y'] for x in models])
            r2 = np.array([x.modelParameters['PLS']['R2Y'] for x in models])

        ax.bar([x - 0.1 for x in range(1, total_comps + 1)], height=r2, width=0.2)
        ax.bar([x + 0.1 for x in range(1, total_comps + 1)], height=q2, width=0.2)
        ax.legend(['R2', 'Q2'])
        ax.set_xlabel("Number of components")
        ax.set_ylabel("R2/Q2Y")

        # Specific case where n comps = 2
        if q2.size == 2:
            plateau_index = np.where(np.diff(q2) / q2[0] < 0.05)[0]
            if plateau_index.size == 0:
                print("Consider exploring a higher level of components")
            else:
                plateau = np.min(np.where(np.diff(q2)/q2[0] < 0.05)[0])
                ax.vlines(x=(plateau + 1), ymin=0, ymax=1, colors='red', linestyles='dashed')
                print("Q2Y measure stabilizes (increase of less than 5% of previous value or decrease) "
                      "at component {0}".format(plateau + 1))

        else:
            plateau_index = np.where((np.diff(q2) / q2[0:-1]) < 0.05)[0]
            if plateau_index.size == 0:
                print("Consider exploring a higher level of components")
            else:
                plateau = np.min(plateau_index)
                ax.vlines(x=(plateau + 1), ymin=0, ymax=1, colors='red', linestyles='dashed')
                print("Q2Y measure stabilizes (increase of less than 5% of previous value or decrease) "
                      "at component {0}".format(plateau + 1))

        plt.show()
        return ax

    def repeated_cv(self, x, y, total_comps=7, repeats=15, cv_method=KFold(7, shuffle=True)):
        """

        Perform repeated cross-validation and plot Q2Y values and their distribution (violin plot) per component
        number to help select the appropriate number of components.

        :param x: Data matrix [n samples, m variables]
        :param total_comps: Maximum number of components to fit
        :param repeats: Number of CV procedure repeats
        :param cv_method: scikit-learn Base Cross-Validator to use
        :return: Violin plot with Q2Y values and distribution per component number.
        """

        q2y = np.zeros((total_comps, repeats))

        for n_components in range(1, total_comps + 1):
            for rep in range(repeats):
                currmodel = deepcopy(self)
                currmodel.n_components = n_components
                currmodel.fit(x, y)
                currmodel.cross_validation(x, y, cv_method=cv_method, outputdist=False)
                q2y[n_components - 1, rep] = currmodel.cvParameters['Q2Y']

        fig, ax = plt.subplots()
        sns.violinplot(data=q2y.T, palette="Set1", ax=ax)
        sns.swarmplot(data=q2y.T, edgecolor="black", color='black', ax=ax)
        ax.set_xticklabels(range(1, total_comps + 1))
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Q2Y")
        plt.show()

        return q2y, ax

    def plot_permutation_test(self, permt_res, metric='Q2Y'):
        try:
            fig, ax = plt.figure()
            hst = ax.hist(permt_res[0][metric], 100)
            if metric == 'Q2Y':
                ax.vlines(x=self.cvParameters['Q2Y'], ymin=0, ymax=max(hst[0]), linestyle='--')
            return ax

        except KeyError:
            print("Run cross-validation before calling the plotting function")
        except Exception as exp:
            raise exp

    def plot_model_parameters(self, parameter='w', component=1, cross_val=False, sigma=2, plottype='spectra',
                              xaxis=None, yaxis=None,
                              xaxislabel='Retention Time', yaxislabel='Mass to charge ratio (m/z)'):

        choices = {'w': self.weights_w, 'c': self.weights_c, 'p': self.loadings_p, 'q': self.loadings_q,
                   'beta': self.beta_coeffs, 'ws': self.rotations_ws, 'cs': self.rotations_cs,
                   'VIP': self.VIP(), 'bu': self.b_u, 'bt': self.b_u}
        choices_cv = {'w': 'Weights_w', 'c': 'Weights_c', 'cs': 'Rotations_cs', 'ws':'Rotations_ws',
                      'q': 'Loadings_q', 'p': 'Loadings_p', 'beta': 'Beta', 'VIP':'VIP'}

        # decrement component to adjust for python indexing
        component -= 1
        # Beta and VIP don't depend on components so have an exception status here
        if cross_val is True:
            if parameter in ['beta', 'VIP']:
                mean = self.cvParameters['Mean_' + choices_cv[parameter]].squeeze()
                error = sigma * self.cvParameters['Stdev_' + choices_cv[parameter]].squeeze()
            else:
                mean = self.cvParameters['Mean_' + choices_cv[parameter]][:, component]
                error = sigma * self.cvParameters['Stdev_' + choices_cv[parameter]][:, component]
        else:
            error = None
            if parameter in ['beta', 'VIP']:
                mean = choices[parameter].squeeze()
            else:
                mean = choices[parameter][:, component]
        if plottype == 'spectra':
            _lineplots(mean, error=error, xaxis=xaxis)
        # To use with barplots for other types of data
        elif plottype == 'bar':
            _barplots(mean, error=error, xaxis=xaxis)
        elif plottype == 'scatterplot':
            _scatterplots(mean, xaxis=xaxis, yaxis=yaxis, xlabel=xaxislabel,
                          ylabel=yaxislabel, cbarlabel=parameter)
        if plottype in ['spectra', 'bar']:
            plt.xlabel("Variable No")
            if parameter in ['beta', 'VIP']:
                plt.ylabel("{0} for PLS model".format(parameter))
            else:
                plt.ylabel("{0} for PLS component {1}".format(parameter, (component + 1)))
            plt.show()

        return None

    def external_validation_set(self, x):
        """

        Interface to score classification using an external hold-out dataset

        :param x:
        :return:
        """
        y_pred = self.predict(x)
        self.score
        validation_set_results = dict()

        return validation_set_results

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
