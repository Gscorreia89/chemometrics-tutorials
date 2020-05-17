"""

Algorithm for PLS with built in filtering of Y orthogonal variation present in the X block.
Adaptation of the default scikit-learn _pls regression code to implement the algorithm described in
Johan Trygg, Svante Wold, Orthogonal projections to latent structures (O-PLS), J. Chemometrics 2002; 16: 119-128

"""
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import pinv2

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
import six

__author__ = 'gscorreia89'


def _nipals_twoblocks_inner_loop(X, Y, mode="A", max_iter=500, tol=1e-06,
                                 norm_y_weights=False):
    """Inner loop of the iterative NIPALS algorithm.
    Provides an alternative to the svd(X'Y); returns the first left and right
    singular vectors of X'Y.  See PLS for the meaning of the parameters.  It is
    similar to the Power method for determining the eigenvectors and
    eigenvalues of a X'Y.
    """
    y_score = Y[:, [0]]
    x_weights_old = 0
    ite = 1
    X_pinv = Y_pinv = None
    eps = np.finfo(X.dtype).eps
    # Inner loop of the Wold algo.
    while True:
        # 1.1 Update u: the X weights
        if mode == "B":
            if X_pinv is None:
                # We use slower pinv2 (same as np.linalg.pinv) for stability
                # reasons
                X_pinv = pinv2(X, check_finite=False)
            x_weights = np.dot(X_pinv, y_score)
        else:  # mode A
            # Mode A regress each X column on y_score
            x_weights = np.dot(X.T, y_score) / np.dot(y_score.T, y_score)
        # If y_score only has zeros x_weights will only have zeros. In
        # this case add an epsilon to converge to a more acceptable
        # solution
        if np.dot(x_weights.T, x_weights) < eps:
            x_weights += eps
        # 1.2 Normalize u
        x_weights /= np.sqrt(np.dot(x_weights.T, x_weights)) + eps
        # 1.3 Update x_score: the X latent scores
        x_score = np.dot(X, x_weights)
        # 2.1 Update y_weights
        if mode == "B":
            if Y_pinv is None:
                Y_pinv = pinv2(Y, check_finite=False)  # compute once pinv(Y)
            y_weights = np.dot(Y_pinv, x_score)
        else:
            # Mode A regress each Y column on x_score
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)
        # 2.2 Normalize y_weights
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights.T, y_weights)) + eps
        # 2.3 Update y_score: the Y latent scores
        y_score = np.dot(Y, y_weights) / (np.dot(y_weights.T, y_weights) + eps)
        # y_score = np.dot(Y, y_weights) / np.dot(y_score.T, y_score) ## BUG
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff.T, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached')
            break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights, ite


def _center_scale_xy(X, Y, scale=True):
    """

    :param X:
    :param Y:
    :param scale:
    :return:
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


class _orthogonal_pls(six.with_metaclass(ABCMeta), BaseEstimator, TransformerMixin,
           RegressorMixin):
    """

    Partial Least Squares (PLS) with filters for Y orthogonal variation present in X.
    Implementation of the algorithm described in:
    Johan Trygg, Svante Wold, Orthogonal projections to latent structures (O-PLS), J. Chemometrics 2002; 16: 119-128

    """

    @abstractmethod
    def __init__(self, n_components=2, scale=True, deflation_mode="regression",
                 mode="A", algorithm="nipals", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        """

        :param n_components: Number of components.
        :param boolean scale: Scale the data matrices.
        :param str deflation_mode: Type of deflation, either 'regression' or 'canonical'
        :param str mode: 'A' for PLS, 'B' for CanonicalCorrelation
        :param str algorithm: Which algorithm to find the weight vector 'nipals' or 'svd'.
        :param boolean norm_y_weights: Normalise y weights.
        :param int max_iter: Maximum number of iterations for NIPALS loop
        :param float tol: tolerance to define convergence in NIPALS loop
        :param boolean copy: Copy the data matrices.
        """
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.norm_y_weights = norm_y_weights
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def fit(self, X, Y):
        """
        :param X: Data matrix to fit the orthogonal PLS model.
        :type X: numpy.ndarray, shape [n_samples, n_features].
        :param Y: Data matrix to fit the orthogonal PLS model.
        :type Y: numpy.ndarray, shape [n_samples, n_features].
        :return: Fitted object.
        :rtype: pyChemometrics._orthogonal_pls
        """

        # copy since this will contains the residuals (deflated) matrices
        check_consistent_length(X, Y)
        Xk = check_array(X, dtype=np.float64, copy=self.copy,
                        ensure_min_samples=2)
        Yk = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        if self.n_components < 1 or self.n_components > p:
            raise ValueError('Invalid number of components: %d' %
                             self.n_components)
        # Scale (in place)
        Xk, Yk, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _center_scale_xy(Xk, Yk, self.scale))

        # Results matrices

        # Orthogonal PLS components
        self.w_ortho = np.zeros((p, self.n_components - 1))
        self.t_ortho = np.zeros((n, self.n_components - 1))
        self.u_ortho = np.zeros((n, self.n_components - 1))
        self.p_ortho = np.zeros((p, self.n_components - 1))
        self.q_ortho = np.zeros((q, self.n_components - 1))
        self.c_ortho = np.zeros((q, self.n_components - 1))

        self.n_iter_ = []

        # 1) weights estimation (inner loop)
        # -----------------------------------

        x_weights, y_weights, n_iter_ = \
            _nipals_twoblocks_inner_loop(
                X=Xk, Y=Yk, mode=self.mode, max_iter=self.max_iter,
                tol=self.tol, norm_y_weights=self.norm_y_weights)
        # different cases for first and other components, where orthogonal partial least squares diverge from PLS]
        # the regular calculations for PLS will also be done even if not needed for orthogonal PLS, for commodity
        x_weights, y_weights = svd_flip(x_weights, y_weights.T)
        y_weights = y_weights.T

        # NIPALS algo: outer loop, over components
        for k in range(self.n_components - 1):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                # Yk constant
                warnings.warn('Y residual constant at iteration %s' % k)
                break

            # compute scores
            x_scores = np.dot(Xk, x_weights)
            x_loadings = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)

            # - regress Yk's on x_score, then subtract rank-one approx.
            y_loadings = (np.dot(Yk.T, x_scores)
                            / np.dot(x_scores.T, x_scores))

            # After calculating the Y associated W component, we have the orthogonal components.
            w_ortho = x_loadings - np.dot((np.dot(x_weights, x_loadings.T) / np.dot(x_weights.T, x_weights)),
                                            x_weights)

            w_ortho /= np.sqrt(np.dot(w_ortho.T, w_ortho))

            t_ortho = np.dot(Xk, w_ortho) / np.dot(w_ortho.T, w_ortho)
            p_ortho = np.dot(Xk.T, t_ortho) / np.dot(t_ortho.T, t_ortho)

            c_ortho = np.dot(Yk.T, t_ortho) / np.dot(t_ortho.T, t_ortho)

            # - regress Yk's on x_score, then subtract rank-one approx.
            q_ortho = (np.dot(Yk.T, t_ortho) / np.dot(t_ortho.T, t_ortho))

            Xk -= np.dot(t_ortho, p_ortho.T)

            if self.norm_y_weights:
                y_ss = 1
                c_ss = 1
            else:
                y_ss = np.dot(y_weights.T, y_weights)
                c_ss = np.dot(c_ortho.T, c_ortho)

            y_scores = np.dot(Yk, y_weights) / y_ss
            u_ortho = np.dot(Yk, c_ortho) / c_ss

            # test for null variance
            if np.dot(x_scores.T, x_scores) < np.finfo(np.double).eps or np.dot(t_ortho.T, t_ortho) < np.finfo(np.double).eps:
                warnings.warn('X scores are null at iteration %s' % k)
                break

            #Yk -= np.dot(x_scores, y_loadings.T)

            # 3) Store weights, scores and loadings # Notation:
            self.t_ortho[:, k] = t_ortho.ravel()
            self.w_ortho[:, k] = w_ortho.ravel()
            self.p_ortho[:, k] = p_ortho.ravel()
            self.q_ortho[:, k] = q_ortho.ravel()
            self.c_ortho[:, k] = c_ortho.ravel()
            self.u_ortho[:, k] = u_ortho.ravel()

        # Refit PLS component
        #Xk = check_array(X, dtype=np.float64, copy=self.copy,
        #                 ensure_min_samples=2)
        #Yk = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)

        #if Y.ndim == 1:
        #    Yk = Y.reshape(-1, 1)

        #Xk, Yk, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
        #    _center_scale_xy(Xk, Yk, self.scale))
        #Xk -= np.dot(self.t_ortho, self.p_ortho.T)
        #Yk -= np.dot(self.t_ortho, self.q_ortho.T)
        # fit filtered predictive component for visualization

        x_weights, y_weights, n_iter_ = \
            _nipals_twoblocks_inner_loop(
                X=Xk, Y=Yk, mode=self.mode, max_iter=self.max_iter,
                tol=self.tol, norm_y_weights=self.norm_y_weights)

        # different cases whether its the first component or not, where orthogonal partial least squares
        # diverge from PLS regular calculations for PLS will also be
        # done even if not needed for orthogonal PLS, for commodity

        x_weights, y_weights = svd_flip(x_weights, y_weights.T)
        y_weights = y_weights.T

        # compute scores
        # final PLS/ "Predictive" component
        # - regress Yk's on x_score, then subtract rank-one approx.
        # compute scores

        x_scores = np.dot(Xk, x_weights)

        self.predictive_w = x_weights
        self.predictive_t = np.dot(Xk, x_weights)
        self.predictive_c = y_weights.T
        self.predictive_p = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)
        self.predictive_q = np.dot(Yk.T, x_scores) / np.dot(x_scores.T, x_scores)

        if self.norm_y_weights:
            y_ss = 1
        else:
            y_ss = np.dot(self.predictive_c.T, self.predictive_c)

        self.predictive_u = np.dot(Yk, self.predictive_c) / y_ss

        # stack the matrices for the orthogonal pls coefficient calculation
        w = np.c_[self.w_ortho, self.predictive_w]
        p = np.c_[self.p_ortho, self.predictive_p]
        q = np.c_[self.q_ortho, self.predictive_q]
        c = np.c_[self.c_ortho, self.predictive_c]
        t = np.c_[self.t_ortho, self.predictive_t]
        u = np.c_[self.u_ortho, self.predictive_u]
        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)

        self.x_rotations_ = np.dot(w,
            pinv2(np.dot(p.T, w),
                  check_finite=False))

        #if Y.shape[1] > 1:
        self.y_rotations_ = np.dot(c, pinv2(np.dot(q.T, c), check_finite=False))
        #else:
        #    self.y_rotations_ = np.ones(1)

        self.coef_ = np.dot(np.dot(w, np.linalg.pinv(np.dot(p.T, w))), q.T)
        self.coef_ *= self.y_std_

        self.b_u = np.dot(np.dot(np.linalg.pinv(np.dot(u.T, u)), u.T),
                          t)
        self.b_t = np.dot(np.dot(np.linalg.pinv(np.dot(t.T, t)), t.T),
                          u)

        return self

    def transform(self, X, Y=None, copy=True):
        """

        Calculate the scores for a data block from the original data.

        :param X: Data matrix to be projected onto the score space (T)
        :type x: numpy.ndarray, shape [n_samples, n_features] or None
        :param y: Data matrix to be projected onto the score space (U)
        :type y: numpy.ndarray, shape [n_samples, n_features] or None
        :param boolean copy: Copy the data matrix
        :return: Either the Latent Variable scores T and U (if Y is not None) or T only.
        :rtype: tuple with 2 numpy.ndarray, shape [n_samples, n_comps], or numpy.ndarray, shape [n_samples, n_comps]
        """

        check_is_fitted(self, 'x_mean_')
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        # Apply rotation
        x_scores = np.dot(X, self.x_rotations_)
        if Y is not None:
            Y = check_array(Y, ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self.y_mean_
            Y /= self.y_std_
            y_scores = np.dot(Y, self.y_rotations_)
            return x_scores, y_scores

        return x_scores

    def predict(self, X, copy=True):
        """Apply the dimension reduction learned on the train data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        check_is_fitted(self, 'x_mean_')
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        Ypred = np.dot(X, self.coef_)
        return Ypred + self.y_mean_

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)


class OrthogonalPLSRegression(_orthogonal_pls):
    """

    Orthogonal PLS regression

    """

    def __init__(self, n_components=2, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super(OrthogonalPLSRegression, self).__init__(
            n_components=n_components, scale=scale,
            deflation_mode="regression", mode="A",
            norm_y_weights=False, max_iter=max_iter, tol=tol,
            copy=copy)
