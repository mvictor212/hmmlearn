# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
# and Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux

"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import string

import numpy as np
from numpy.random import multivariate_normal, normal

from sklearn.utils import check_random_state
from sklearn.utils.extmath import logsumexp
from sklearn.base import BaseEstimator
from sklearn.mixture import (
    GMM, sample_gaussian,
    distribute_covar_matrix_to_match_covariance_type, _validate_covars)
from sklearn import cluster
from scipy.stats import (poisson, expon)
from copy import deepcopy

from .utils.fixes import (log_multivariate_normal_density,
                          log_poisson_pmf, log_exponential_density)

from . import _hmmc

__all__ = ['GMMHMM',
           'GaussianHMM',
           'MultinomialHMM',
           'decoder_algorithms',
           'normalize']

ZEROLOGPROB = -1e200
EPS = np.finfo(float).eps
NEGINF = -np.inf
decoder_algorithms = ("viterbi", "map")


def identity(x):
    return x


def _do_estep(seq, modelBroadcast):
    model = modelBroadcast.value
    stats = model._initialize_sufficient_statistics()
    framelogprob = model._compute_log_likelihood(seq)
    lpr, fwdlattice = model._do_forward_pass(framelogprob)
    bwdlattice = model._do_backward_pass(framelogprob)
    gamma = fwdlattice + bwdlattice
    posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
    model._accumulate_sufficient_statistics(stats,
                                            seq,
                                            framelogprob,
                                            posteriors,
                                            fwdlattice,
                                            bwdlattice,
                                            model.params)
    return stats, lpr


def _score(seq, modelBroadcast):
    model = modelBroadcast.value
    seq = np.asarray(seq)
    framelogprob = model._compute_log_likelihood(seq)
    lpr, _ = model._do_forward_pass(framelogprob)
    return lpr


def batches(l, n):
    """ Yield successive n-sized batches from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:(i + n)]


def merge_sum(x, y):
    D = {}
    for k in x.keys():
        if isinstance(x[k], list):
            z = []
            for i in xrange(len(x[k])):
                z.append(x[k][i] + y[k][i])
            D[k] = z
        else:
            D[k] = x[k] + y[k]
    return D


def log_normalize(A, axis=None):
    arr = np.rollaxis(A, axis)
    vmax = arr.max(axis=axis)
    return normalize(np.exp((arr.T - vmax).T))


def normalize(A, axis=None):
    """ Normalize the input array so that it sums to 1.

    WARNING: The HMM module and its functions will be removed in 0.17
    as it no longer falls within the project's scope and API.

    Parameters
    ----------
    A: array, shape (n_samples, n_features)
       Non-normalized input data
    axis: int
          dimension along which normalization is performed

    Returns
    -------
    normalized_A: array, shape (n_samples, n_features)
        A with values normalized (summing to 1) along the prescribed axis

    WARNING: Modifies inplace the array
    """
    A += EPS
    Asum = A.sum(axis)
    if axis and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    return A / Asum


def randomize(A, axis=None):
    randomizer = np.random.rand(*A.shape) / 10.
    Arand = A + randomizer
    return normalize(Arand, axis)


class VerboseReporter(object):
    """Reports verbose output to stdout.

    If ``verbose==1`` output is printed once in a while (when iteration mod
    verbose_mod is zero).; if larger than 1 then output is printed for
    each update.
    """

    def __init__(self, verbose):
        self.verbose = verbose
        self.verbose_fmt = '{iter:>10d} {lpr:>16.4f} {improvement:>16.4f}'
        self.verbose_mod = 1

    def init(self):
        header_fields = ['Iter', 'Log Likelihood', 'Log Improvement']
        print(('%10s ' + '%16s ' *
               (len(header_fields) - 1)) % tuple(header_fields))

    def update(self, i, lpr, improvement):
        """Update reporter with new iteration. """
        # we need to take into account if we fit additional estimators.
        if (i + 1) % self.verbose_mod == 0:
            print(self.verbose_fmt.format(iter=i + 1,
                                          lpr=lpr,
                                          improvement=improvement))
            if self.verbose == 1 and ((i + 1) // (self.verbose_mod * 10) > 0):
                # adjust verbose frequency (powers of 10)
                self.verbose_mod *= 10


class _BaseHMM(BaseEstimator):
    """Hidden Markov Model base class.

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    Attributes
    ----------
    n_states : int
        Number of states in the model.

    transmat : array, shape (`n_states`, `n_states`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_states`,)
        Initial state occupation distribution.

    transmat_prior : array, shape (`n_states`, `n_states`)
        Matrix of prior transition probabilities between states.

    startprob_prior : array, shape ('n_states`,)
        Initial state occupation prior distribution.

    algorithm : string, one of the decoder_algorithms
        decoder algorithm

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, and other characters for subclass-specific
        emmission parameters. Defaults to all parameters.


    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emmission parameters. Defaults to all
        parameters.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more iterations the lower the frequency). If
        greater than 1 then it prints progress and performance for every
        iteration.


    See Also
    --------
    GMM : Gaussian mixture model
    """

    # This class implements the public interface to all HMMs that
    # derive from it, including all of the machinery for the
    # forward-backward and Viterbi algorithms.  Subclasses need only
    # implement _generate_sample_from_state(), _compute_log_likelihood(),
    # _init(), _initialize_sufficient_statistics(),
    # _accumulate_sufficient_statistics(), and _do_mstep(), all of
    # which depend on the specific emission distribution.
    #
    # Subclasses will probably also want to implement properties for
    # the emission distribution parameters to expose them publicly.

    def __init__(self, n_states=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters, verbose=0,
                 batch_size=1):

        self.n_states = n_states
        self.n_iter = n_iter
        self.thresh = thresh
        self.params = params
        self.init_params = init_params
        self.startprob_ = startprob
        if startprob_prior is None:
            startprob_prior = np.ones(n_states)
        self.startprob_prior = startprob_prior
        self.transmat_ = transmat
        if transmat_prior is None:
            transmat_prior = np.ones((n_states,
                                      n_states))
        self.transmat_prior = transmat_prior
        self._algorithm = algorithm
        self.random_state = random_state
        self.verbose = verbose
        self.batch_size = batch_size

    def eval(self, X):
        return self.score_samples(X)

    def score_samples(self, obs):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        logprob : float
            Log likelihood of the sequence ``obs``.

        posteriors : list of array_like, shape (n, n_states)
            Posterior probabilities of each state for each
            observation

        See Also
        --------
        score : Compute the log probability under the model
        decode : Find most likely state sequence corresponding to a `obs`
        """
        logprob = 0
        posteriors = []
        for seq in obs:
            seq = np.asarray(seq)
            framelogprob = self._compute_log_likelihood(seq)
            lpr, fwdlattice = self._do_forward_pass(framelogprob)
            bwdlattice = self._do_backward_pass(framelogprob)
            gamma = fwdlattice + bwdlattice
            # gamma is guaranteed to be correctly normalized by logprob at
            # all frames, unless we do approximate inference using pruning.
            # So, we will normalize each frame explicitly in case we
            # pruned too aggressively.
            posteriors.append(np.exp(gamma.T - logsumexp(gamma, axis=1)).T)
            posteriors[-1] += np.finfo(np.float32).eps
            posteriors[-1] /= np.sum(posteriors, axis=1).reshape((-1, 1))
            logprob += lpr
        return logprob, posteriors

    def score(self, sc, data):
        """Compute the log probability under the model.

        Parameters
        ----------
        obs : list of array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : float
            Log likelihood of the ``obs``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors

        decode : Find most likely state sequence corresponding to a `obs`
        """
        modelBroadcast = sc.broadcast(self)
        logprob = data.map(lambda seq: _score(seq, modelBroadcast)).reduce(lambda a, b: a + b)
        return logprob

    def aic(self, sc, obs):
        """Computes the Aikaike Information Criterion of the model and
        set of observations.

        Parameters
        ----------
        obs : list of arrays
            List of observation sequences.

        Returns
        -------
        aic_score : float
            The Aikaike Information Criterion.
        """
        logprob = self.score(sc, obs)
        n_pars = self._n_free_parameters()
        aic_score = 2 * n_pars - 2 * logprob
        return aic_score

    def bic(self, sc, obs):
        """Computes the Aikaike Information Criterion of the model and
        set of observations.

        Parameters
        ----------
        obs : list of arrays
            List of observation sequences.

        Returns
        -------
        bic_score : float
            The Aikaike Information Criterion.
        """
        logprob = self.score(sc, obs)
        n_pars = self._n_free_parameters()
        n_data = sum([len(seq) for seq in obs])
        bic_score = n_pars * (np.log(n_data) - np.log(2 * np.pi)) - 2 * logprob
        return bic_score

    def _decode_viterbi(self, obs):
        """Find most likely state sequence corresponding to ``obs``.

        Uses the Viterbi algorithm.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        viterbi_logprobs : array_like, shape (n,)
            Log probability of the maximum likelihood path through the HMM.

        state_sequences : list of array_like, shape (n,)
            Index of the most likely states for each observation.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model
        """
        viterbi_logprobs = np.zeros(len(obs))
        state_sequences = []
        for n, seq in enumerate(obs):
            seq = np.asarray(seq)
            framelogprob = self._compute_log_likelihood(seq)
            viterbi_logprobs[n], state_sequence = self._do_viterbi_pass(
                framelogprob)
            state_sequences.append(state_sequence)
        return viterbi_logprobs, state_sequences

    def _decode_map(self, obs):
        """Find most likely state sequence corresponding to `obs`.

        Uses the maximum a posteriori estimation.

        Parameters
        ----------
        obs : list of array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        map_logprobs : array_like, shape (n,)
            Log probability of the maximum likelihood path through the HMM
        state_sequences : list of array_like, shape (n,)
            Index of the most likely states for each observation

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        map_logprobs = np.zeros(len(obs))
        state_sequences = []
        _, posteriors = self.score_samples(obs)
        for n, post in enumerate(posteriors):
            state_sequences.append(np.argmax(post, axis=1))
            map_logprobs[n] = np.max(post, axis=1).sum()
        return map_logprobs, state_sequences

    def decode(self, obs, algorithm="viterbi"):
        """Find most likely state sequence corresponding to ``obs``.
        Uses the selected algorithm for decoding.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        algorithm : string, one of the `decoder_algorithms`
            decoder algorithm to be used

        Returns
        -------
        logprobs : array_like, shape (n,)
            Log probability of the maximum likelihood path through the HMM

        state_sequences : list of array_like, shape (n,)
            Index of the most likely states for each observation

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model.
        """
        if self._algorithm in decoder_algorithms:
            algorithm = self._algorithm
        elif algorithm in decoder_algorithms:
            algorithm = algorithm
        decoder = {"viterbi": self._decode_viterbi,
                   "map": self._decode_map}
        logprobs, state_sequences = decoder[algorithm](obs)
        return logprobs, state_sequences

    def predict(self, obs, algorithm="viterbi"):
        """Find most likely state sequence corresponding to `obs`.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        state_sequences : list of array_like, shape (n,)
            Index of the most likely states for each observation
        """
        _, state_sequences = self.decode(obs, algorithm)
        return state_sequences

    def predict_proba(self, obs):
        """Compute the posterior probability for each state in the model

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        posteriors : list of array-like, shape (n, n_states)
            Returns the probability of the sample for each state in the model.
        """
        _, posteriors = self.score_samples(obs)
        return posteriors

    def sample(self, n_seq=1, n_min=10, n_max=20, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_seq : int
            Number of observation sequences to generate.

        n_min : int
            Minimum number of observations for a sequence.

        n_max : int
            Maximum number of observations for a sequence.

        random_state: RandomState or an int seed (0 by default)
            A random number generator instance. If None is given, the
            object's random_state is used

        Returns
        -------
        (obs, hidden_states)
        obs : list of array_like, length `n_seq` List of samples
        states : list of array_like, length `n_seq` List of hidden states
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_pdf = self.startprob_
        startprob_cdf = np.cumsum(startprob_pdf)
        transmat_pdf = self.transmat_
        transmat_cdf = np.cumsum(transmat_pdf, 1)

        obs = []
        states = []

        for _ in range(n_seq):
            n = np.random.randint(n_min, n_max, size=1)
            # Initial state.
            rand = random_state.rand()
            currstate = (startprob_cdf > rand).argmax()
            state_seq = [currstate]
            obs_seq = [self._generate_sample_from_state(
                currstate, random_state=random_state)]

            for _ in range(n - 1):
                rand = random_state.rand()
                currstate = (transmat_cdf[currstate] > rand).argmax()
                state_seq.append(currstate)
                obs_seq.append(self._generate_sample_from_state(
                    currstate, random_state=random_state))

            obs.append(deepcopy(np.array(obs_seq)))
            states.append(deepcopy(np.array(state_seq, dtype=int)))

        return obs, states

    def fit(self, sc, data):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation. Alternatively, a list of strings,
            each of which is a filepath to a pickled object, being
            a list of array-like observation sequences.

        Notes
        -----
        In general, `logprob` should be non-decreasing unless
        aggressive pruning is used.  Decreasing `logprob` is generally
        a sign of overfitting (e.g. a covariance parameter getting too
        small).  You can fix this by getting more training data,
        or strengthening the appropriate subclass-specific regularization
        parameter.
        """

        if self.algorithm not in decoder_algorithms:
            self._algorithm = "viterbi"

        self._init(data, self.init_params)

        if self.verbose:
            verbose_reporter = VerboseReporter(self.verbose)
            verbose_reporter.init()

        data.cache()

        logprob = []
        for i in range(self.n_iter):
            # Expectation step
            modelBroadcast = sc.broadcast(self)
            results = data.map(lambda seq: _do_estep(seq, modelBroadcast)).cache()
            stats = results.keys().reduce(merge_sum)
            curr_logprob = results.values().reduce(lambda a, b:
                                                   a + b)
            logprob.append(curr_logprob)
            if i > 0:
                improvement = logprob[-1] - logprob[-2]
            else:
                improvement = np.inf
            if self.verbose:
                verbose_reporter.update(i, curr_logprob, improvement)

            # Check for convergence.
            if i > 0 and abs(logprob[-1] - logprob[-2]) < self.thresh:
                break

            # Maximization step
            self._do_mstep(stats, self.params)

        return self

    def _get_algorithm(self):
        "decoder algorithm"
        return self._algorithm

    def _set_algorithm(self, algorithm):
        if algorithm not in decoder_algorithms:
            raise ValueError("algorithm must be one of the decoder_algorithms")
        self._algorithm = algorithm

    algorithm = property(_get_algorithm, _set_algorithm)

    def _get_startprob(self):
        """Mixing startprob for each state."""
        return np.exp(self._log_startprob)

    def _set_startprob(self, startprob):
        if startprob is None:
            startprob = np.tile(1.0 / self.n_states, self.n_states)
        else:
            startprob = np.asarray(startprob, dtype=np.float)

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(startprob):
            normalize(startprob)

        if len(startprob) != self.n_states:
            raise ValueError('startprob must have length n_states')
        if not np.allclose(np.sum(startprob), 1.0):
            raise ValueError('startprob must sum to 1.0')

        self._log_startprob = np.log(np.asarray(startprob).copy())

    startprob_ = property(_get_startprob, _set_startprob)

    def _get_transmat(self):
        """Matrix of transition probabilities."""
        return np.exp(self._log_transmat)

    def _set_transmat(self, transmat):
        if transmat is None:
            transmat = np.tile(1.0 / self.n_states,
                               (self.n_states, self.n_states))

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(transmat):
            normalize(transmat, axis=1)

        if (np.asarray(transmat).shape
                != (self.n_states, self.n_states)):
            raise ValueError('transmat must have shape '
                             '(n_states, n_states)')
        if not np.all(np.allclose(np.sum(transmat, axis=1), 1.0)):
            raise ValueError('Rows of transmat must sum to 1.0')

        self._log_transmat = np.log(np.asarray(transmat).copy())
        underflow_idx = np.isnan(self._log_transmat)
        self._log_transmat[underflow_idx] = NEGINF

    transmat_ = property(_get_transmat, _set_transmat)

    def _do_viterbi_pass(self, framelogprob):
        n_observations, n_states = framelogprob.shape
        state_sequence, logprob = _hmmc._viterbi(
            n_observations, n_states, self._log_startprob,
            self._log_transmat, framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob):

        n_observations, n_states = framelogprob.shape
        fwdlattice = np.zeros((n_observations, n_states))
        _hmmc._forward(n_observations, n_states, self._log_startprob,
                       self._log_transmat, framelogprob, fwdlattice)
        fwdlattice[fwdlattice <= ZEROLOGPROB] = NEGINF
        return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        n_observations, n_states = framelogprob.shape
        bwdlattice = np.zeros((n_observations, n_states))
        _hmmc._backward(n_observations, n_states, self._log_startprob,
                        self._log_transmat, framelogprob, bwdlattice)

        bwdlattice[bwdlattice <= ZEROLOGPROB] = NEGINF

        return bwdlattice

    def _compute_log_likelihood(self, obs):
        pass

    def _generate_sample_from_state(self, state, random_state=None):
        pass

    def _init(self, obs, params):
        if 's' in params:
            self.startprob_ = np.random.dirichlet(self.startprob_prior)
        if 't' in params:
            self.transmat_ = np.vstack([np.random.dirichlet(
                self.transmat_prior[i])
                for i in xrange(self.n_states)])
    # Methods used by self.fit()

    def _initialize_sufficient_statistics(self):
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_states),
                 'trans': np.zeros((self.n_states, self.n_states))}
        return stats

    def _accumulate_sufficient_statistics(self, stats, seq, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        stats['nobs'] += 1
        if 's' in params:
            stats['start'] += posteriors[0]
        if 't' in params:
            n_observations, n_states = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_observations > 1:
                lneta = np.zeros((n_observations - 1,
                                  n_states,
                                  n_states))
                lnP = logsumexp(fwdlattice[-1])
                _hmmc._compute_lneta(n_observations, n_states, fwdlattice,
                                     self._log_transmat, bwdlattice,
                                     framelogprob, lnP, lneta)
                stats['trans'] += np.exp(np.minimum(logsumexp(lneta, 0), 700))

    def _do_mstep(self, stats, params):
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445

        if 's' in params:
            self.startprob_ = normalize(np.maximum(stats['start'], 1e-20))
        if 't' in params:
            self.transmat_ = normalize(np.maximum(stats['trans'], 1e-20), 1)

    def _n_free_parameters(self):
        pass


class GaussianHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian emissions

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    Parameters
    ----------
    n_states : int
        Number of states.

    ``_covariance_type`` : string
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    Attributes
    ----------
    ``_covariance_type`` : string
        String describing the type of covariance parameters used by
        the model.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    n_features : int
        Dimensionality of the Gaussian emissions.

    n_states : int
        Number of states in the model.

    transmat : array, shape (`n_states`, `n_states`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_states`,)
        Initial state occupation distribution.

    means : array, shape (`n_states`, `n_features`)
        Mean parameters for each state.

    covars : array
        Covariance parameters for each state.  The shape depends on
        ``_covariance_type``::

            (`n_states`,)                   if 'spherical',
            (`n_features`, `n_features`)              if 'tied',
            (`n_states`, `n_features`)           if 'diag',
            (`n_states`, `n_features`, `n_features`)  if 'full'

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means, and 'c' for covars.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means, and 'c' for
        covars.  Defaults to all parameters.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more iterations the lower the frequency). If
        greater than 1 then it prints progress and performance for every
        iteration.

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_states=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GaussianHMM(algorithm='viterbi',...


    See Also
    --------
    GMM : Gaussian mixture model
    """

    def __init__(self, n_states=1, covariance_type='diag', startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", means_var=1.0,
                 covars_prior=1e-2, covars_weight=1,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 verbose=0,
                 batch_size=1):
        _BaseHMM.__init__(self, n_states, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          thresh=thresh, params=params,
                          init_params=init_params, verbose=verbose,
                          batch_size=batch_size)

        self._covariance_type = covariance_type
        if not covariance_type in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('bad covariance_type')

        self.means_var = means_var

        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    @property
    def covariance_type(self):
        """Covariance type of the model.

        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._covariance_type

    def _get_means(self):
        """Mean parameters for each state."""
        return self._means_

    def _set_means(self, means):
        means = np.asarray(means)
        if (hasattr(self, 'n_features')
                and means.shape != (self.n_states, self.n_features)):
            raise ValueError('means must have shape '
                             '(n_states, n_features)')
        self._means_ = means.copy()
        self.n_features = self._means_.shape[1]

    means_ = property(_get_means, _set_means)

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self._covariance_type == 'full':
            return self._covars_
        elif self._covariance_type == 'diag':
            return [np.diag(cov) for cov in self._covars_]
        elif self._covariance_type == 'tied':
            return [self._covars_] * self.n_states
        elif self._covariance_type == 'spherical':
            return [np.eye(self.n_features) * f for f in self._covars_]

    def _set_covars(self, covars):
        covars = np.asarray(covars)
        _validate_covars(covars, self._covariance_type, self.n_states)
        self._covars_ = covars.copy()

    covars_ = property(_get_covars, _set_covars)

    def _compute_log_likelihood(self, obs):
        return log_multivariate_normal_density(
            obs, self._means_, self._covars_, self._covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        if self._covariance_type == 'tied':
            cv = self._covars_
        else:
            cv = self._covars_[state]
        return sample_gaussian(self._means_[state], cv, self._covariance_type,
                               random_state=random_state)

    def _init(self, obs, params='stmc'):
        super(GaussianHMM, self)._init(obs, params=params)

        concat_obs = np.vstack(obs
                               .flatMap(identity)
                               .sample(False, 0.01)
                               .map(np.atleast_2d)
                               .collect())
        if (hasattr(self, 'n_features')
                and self.n_features != concat_obs.shape[1]):
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (concat_obs.shape[1],
                                              self.n_features))

        self.n_features = concat_obs.shape[1]

        if 'm' in params:
            clu = cluster.KMeans(n_clusters=self.n_states).fit(
                concat_obs)
            self._means_ = np.array([multivariate_normal(
                mean,
                np.eye(self.n_features) * self.means_var)
                for mean in clu.cluster_centers_])
        if 'c' in params:
            cv = np.cov(concat_obs.T)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self._covariance_type, self.n_states)
            self._covars_[self._covars_ == 0] = 1e-5

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_states)
        stats['obs'] = np.zeros((self.n_states, self.n_features))
        stats['obs**2'] = np.zeros((self.n_states, self.n_features))
        if self._covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_states, self.n_features,
                                          self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'm' in params or 'c' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in params:
            if self._covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self._covariance_type in ('tied', 'full'):
                for t, o in enumerate(obs):
                    obsobsT = np.outer(o, o)
                    for c in range(self.n_states):
                        stats['obs*obs.T'][c] += posteriors[t, c] * obsobsT

    def _do_mstep(self, stats, params):
        super(GaussianHMM, self)._do_mstep(stats, params)

        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]
        if 'm' in params:
            self._means_ = stats['obs'] / stats['post'][:, np.newaxis]

        if 'c' in params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            if covars_prior is None:
                covars_weight = 0
                covars_prior = 0

            if self._covariance_type in ('spherical', 'diag'):
                cv_num = ((self._means_) ** 2
                          + stats['obs**2']
                          - 2 * self._means_ * stats['obs']
                          + self._means_ ** 2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = (covars_prior + cv_num) / np.maximum(cv_den,
                                                                     1e-5)
                if self._covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self._covariance_type in ('tied', 'full'):
                cvnum = np.empty((self.n_states, self.n_features,
                                  self.n_features))
                for c in range(self.n_states):
                    obsmean = np.outer(stats['obs'][c], self._means_[c])

                    cvnum[c] = (np.outer(self._means_[c],
                                         self._means_[c])
                                + stats['obs*obs.T'][c]
                                - obsmean - obsmean.T
                                + np.outer(self._means_[c], self._means_[c])
                                * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self._covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cvnum.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self._covariance_type == 'full':
                    self._covars_ = ((covars_prior + cvnum) /
                                     (cvweight + stats['post'][:, None, None]))

    def _n_free_parameters(self):
        n_pars = (self.n_states - 1) * (self.n_states + 1)
        n_pars += self.n_states * self.n_features
        if self._covariance_type == 'spherical':
            n_pars += self.n_states
        elif self._covariance_type == 'tied':
            n_pars += ((self.n_features + 1) * self.n_features) / 2
        elif self._covariance_type == 'diag':
            n_pars += self.n_states * self.n_features
        elif self._covariance_type == 'full':
            n_pars += self.n_states * ((self.n_features + 1)
                                       * self.n_features) / 2
        return n_pars

    def fit(self, sc, obs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation. Alternatively, a list of strings,
            each of which is a filepath to a pickled object, being
            a list of array-like observation sequences.

        Notes
        -----
        In general, `logprob` should be non-decreasing unless
        aggressive pruning is used.  Decreasing `logprob` is generally
        a sign of overfitting (e.g. the covariance parameter on one or
        more components becomminging too small).  You can fix this by getting
        more training data, or increasing covars_prior.
        """
        return super(GaussianHMM, self).fit(sc, obs)


class MultinomialHMM(_BaseHMM):
    """Hidden Markov Model with multinomial (discrete) emissions

    Attributes
    ----------
    n_states : int
        Number of states in the model.

    n_symbols : int
        Number of possible symbols emitted by the model (in the observations).

    transmat : array, shape (`n_states`, `n_states`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_states`,)
        Initial state occupation distribution.

    emissionprob : array, shape ('n_states`, 'n_symbols`)
        Probability of emitting a given symbol when in each state.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more iterations the lower the frequency). If
        greater than 1 then it prints progress and performance for every
        iteration.

    Examples
    --------
    >>> from hmmlearn.hmm import MultinomialHMM
    >>> MultinomialHMM(n_states=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultinomialHMM(algorithm='viterbi',...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_states=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 emissionprob_prior=None, algorithm="viterbi",
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters, init_params=string.ascii_letters,
                 verbose=0, batch_size=1):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_states : int
            Number of states.
        """
        _BaseHMM.__init__(self, n_states, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params,
                          verbose=verbose,
                          batch_size=batch_size)

        self.emissionprob_prior = emissionprob_prior

    def _get_emissionprob(self):
        """Emission probability distribution for each state."""
        return np.exp(self._log_emissionprob)

    def _set_emissionprob(self, emissionprob):
        emissionprob = np.asarray(emissionprob)
        if hasattr(self, 'n_symbols') and \
                emissionprob.shape != (self.n_states, self.n_symbols):
            raise ValueError('emissionprob must have shape '
                             '(n_states, n_symbols)')

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(emissionprob):
            normalize(emissionprob)

        self._log_emissionprob = np.log(emissionprob)
        underflow_idx = np.isnan(self._log_emissionprob)
        self._log_emissionprob[underflow_idx] = NEGINF
        self.n_symbols = self._log_emissionprob.shape[1]

    emissionprob_ = property(_get_emissionprob, _set_emissionprob)

    def _compute_log_likelihood(self, obs):
        return self._log_emissionprob[:, obs].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        rand = random_state.rand()
        symbol = (cdf > rand).argmax()
        return symbol

    def _init(self, obs, params='ste'):
        super(MultinomialHMM, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

        if 'e' in params:
            if not hasattr(self, 'n_symbols'):
                symbols = set(obs.flatMap(identity).distinct().collect())
                self.n_symbols = len(symbols)
            if self.emissionprob_prior is None:
                self.emissionprob_prior = np.ones((self.n_states,
                                                   self.n_symbols))
            emissionprob = np.vstack([np.random.dirichlet(
                self.emissionprob_prior[i])
                for i in xrange(self.n_states)])
            self.emissionprob_ = emissionprob

    def _initialize_sufficient_statistics(self):
        stats = super(MultinomialHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_states, self.n_symbols))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(MultinomialHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        if 'e' in params:
            for t, symbol in enumerate(obs):
                stats['obs'][:, symbol] += posteriors[t]

    def _do_mstep(self, stats, params):
        super(MultinomialHMM, self)._do_mstep(stats, params)
        if 'e' in params:
            self.emissionprob_ = (stats['obs']
                                  / stats['obs'].sum(1)[:, np.newaxis])

    def _check_input_symbols(self, obs):
        """check if input can be used for Multinomial.fit input must be both
        positive integer array and every element must be continuous.
        e.g. x = [0, 0, 2, 1, 3, 1, 1] is OK and y = [0, 0, 3, 5, 10] not
        """

        symbols = obs.copy()

        if symbols.dtype.kind != 'i':
            # input symbols must be integer
            return False

        if len(symbols) == 1:
            # input too short
            return False

        if np.any(symbols < 0):
            # input contains negative intiger
            return False

        return True

    def _n_free_parameters(self):
        n_pars = (self.n_states - 1) * (self.n_states + 1)
        n_pars += self.n_states * (self.n_symbols - 1)
        return n_pars

    def fit(self, sc, data, **kwargs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation. Alternatively, a list of strings,
            each of which is a filepath to a pickled object, being
            a list of array-like observation sequences.
        """
        err_msg = ("Input must be a list of non-negative integer arrays where "
                   "in all, every element must be continuous, but %s was "
                   "given.")

        modelBroadcast = sc.broadcast(self)

        if not data.map(lambda x: modelBroadcast.value._check_input_symbols(x)).min():
            raise ValueError(err_msg % data.take(5))
        elif np.any(np.diff(data.flatMap(identity).distinct().sortBy(identity).collect()) > 1):
            raise ValueError(err_msg % data.take(5))

        return _BaseHMM.fit(self, sc, data, **kwargs)


class PoissonHMM(_BaseHMM):
    """Hidden Markov Model with Poisson (discrete) emissions

    Attributes
    ----------
    n_states : int
        Number of states in the model.

    transmat : array, shape (`n_states`, `n_states`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_states`,)
        Initial state occupation distribution.

    rates : array, shape ('n_states`,)
        Poisson rate parameters for each state.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more iterations the lower the frequency). If
        greater than 1 then it prints progress and performance for every
        iteration.

    Examples
    --------
    >>> from hmmlearn.hmm import PoissonHMM
    >>> PoissonHMM(n_states=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    PoissonHMM(algorithm='viterbi',...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_states=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 rates_var=1.0, algorithm="viterbi",
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters, verbose=0,
                 batch_size=1):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_states : int
            Number of states.
        """
        _BaseHMM.__init__(self, n_states, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params,
                          verbose=verbose,
                          batch_size=batch_size)
        self.rates_var = rates_var

    def _get_rates(self):
        """Emission rate for each state."""
        return self._rates

    def _set_rates(self, rates):
        rates = np.asarray(rates)
        self._rates = rates.copy()

    rates_ = property(_get_rates, _set_rates)

    def _compute_log_likelihood(self, obs):
        return log_poisson_pmf(obs, self._rates)

    def _generate_sample_from_state(self, state, random_state=None):
        return poisson.rvs(self._rates[state])

    def _init(self, obs, params='str'):
        super(PoissonHMM, self)._init(obs, params=params)

        concat_obs = np.array(obs.flatMap(identity).sample(False, 0.01).collect())
        if 'r' in params:
            clu = cluster.KMeans(n_clusters=self.n_states).fit(
                np.atleast_2d(concat_obs).T)
            rates = normal(0, self.rates_var, self.n_states) + \
                clu.cluster_centers_.T[0]
            self._rates = np.maximum(0.1, rates)

    def _initialize_sufficient_statistics(self):
        stats = super(PoissonHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_states)
        stats['obs'] = np.zeros((self.n_states,))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'r' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

    def _do_mstep(self, stats, params):
        super(PoissonHMM, self)._do_mstep(stats, params)

        if 'r' in params:
            self._rates = stats['obs'] / stats['post']

    def _check_input_symbols(self, obs):
        """check if input can be used for PoissonHMM. Input must be a list
        of non-negative integers.
        e.g. x = [0, 0, 2, 1, 3, 1, 1] is OK and y = [0, -1, 3, 5, 10] not
        """
        symbols = obs.copy()
        if symbols.dtype.kind != 'i':
            # input symbols must be integer
            return False

        if len(symbols) == 1:
            # input too short
            return False

        if np.any(symbols < 0):
            # input contains negative intiger
            return False

        return True

    def _n_free_parameters(self):
        n_pars = (self.n_states - 1) * (self.n_states + 1)
        n_pars += self.n_states
        return n_pars

    def fit(self, sc, data):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation. Alternatively, a list of strings,
            each of which is a filepath to a pickled object, being
            a list of array-like observation sequences.

        Notes
        -----
        In general, `logprob` should be non-decreasing unless
        aggressive pruning is used.  Decreasing `logprob` is generally
        a sign of overfitting (e.g. the covariance parameter on one or
        more components becomminging too small).  You can fix this by getting
        more training data, or increasing covars_prior.
        """
        err_msg = ("Input must be a list of non-negative integer arrays, \
                   but %s was given.")

        modelBroadcast = sc.broadcast(self)

        if not data.map(lambda x: modelBroadcast.value._check_input_symbols(x)).min():
            raise ValueError(err_msg % data.take(5))

        return super(PoissonHMM, self).fit(sc, data)


class ExponentialHMM(_BaseHMM):
    """Hidden Markov Model with Exponential (continuous) emissions

    Attributes
    ----------
    n_states : int
        Number of states in the model.

    transmat : array, shape (`n_states`, `n_states`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_states`,)
        Initial state occupation distribution.

    rates : array, shape ('n_states`,)
        Exponential rate parameters for each state.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more iterations the lower the frequency). If
        greater than 1 then it prints progress and performance for every
        iteration.

    Examples
    --------
    >>> from hmmlearn.hmm import ExponentialHMM
    >>> ExponentialHMM(n_states=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ExponentialHMM(algorithm='viterbi',...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_states=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 rates_var=1.0, algorithm="viterbi",
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters, verbose=0,
                 batch_size=1):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_states : int
            Number of states.
        """
        _BaseHMM.__init__(self, n_states, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params,
                          verbose=verbose,
                          batch_size=batch_size)
        self.rates_var = rates_var

    def _get_rates(self):
        """Emission rate for each state."""
        return self._rates

    def _set_rates(self, rates):
        rates = np.asarray(rates)
        self._rates = rates.copy()

    rates_ = property(_get_rates, _set_rates)

    def _compute_log_likelihood(self, obs):
        return log_exponential_density(obs, self._rates)

    def _generate_sample_from_state(self, state, random_state=None):
        return expon.rvs(scale=1. / self._rates[state])

    def _init(self, obs, params='str'):
        super(ExponentialHMM, self)._init(obs, params=params)

        concat_obs = np.array(obs
                              .flatMap(identity)
                              .sample(False, 0.01)
                              .collect())
        if 'r' in params:
            clu = cluster.KMeans(n_clusters=self.n_states).fit(
                np.atleast_2d(concat_obs).T)
            rates = normal(0, self.rates_var, self.n_states) + \
                1. / clu.cluster_centers_.T[0]
            self._rates = np.maximum(0.1, rates)

    def _initialize_sufficient_statistics(self):
        stats = super(ExponentialHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_states)
        stats['obs'] = np.zeros((self.n_states,))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(ExponentialHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'r' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

    def _do_mstep(self, stats, params):
        super(ExponentialHMM, self)._do_mstep(stats, params)

        if 'r' in params:
            self._rates = stats['post'] / stats['obs']

    def _check_input_symbols(self, obs):
        """check if input can be used for ExponentialHMM. Input must be a list
        of non-negative reals.
        e.g. x = [0., 0.5, 2.3] is OK and y = [0.0, -1.0, 3.3, 5.4, 10.9] not
        """

        symbols = obs.copy()
        if symbols.dtype.kind not in ('f', 'i'):
            # input symbols must be integer
            return False

        if len(symbols) == 1:
            # input too short
            return False

        if np.any(symbols < 0):
            # input contains negative intiger
            return False

        return True

    def _n_free_parameters(self):
        n_pars = (self.n_states - 1) * (self.n_states + 1)
        n_pars += self.n_states
        return n_pars

    def fit(self, sc, data):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation. Alternatively, a list of strings,
            each of which is a filepath to a pickled object, being
            a list of array-like observation sequences.

        Notes
        -----
        In general, `logprob` should be non-decreasing unless
        aggressive pruning is used.  Decreasing `logprob` is generally
        a sign of overfitting (e.g. the covariance parameter on one or
        more components becomminging too small).  You can fix this by getting
        more training data, or increasing covars_prior.
        """
        err_msg = ("Input must be a list of non-negative real arrays, \
                   but %s was given.")

        modelBroadcast = sc.broadcast(self)

        if not data.map(lambda x: modelBroadcast.value._check_input_symbols(x)).min():
            raise ValueError(err_msg % data.take(5))

        return super(ExponentialHMM, self).fit(sc, data)


class MultinomialExponentialHMM(_BaseHMM):
    """Hidden Markov Model with joint multinomial and exponential emissions

    Attributes
    ----------
    n_states : int
        Number of states in the model.

    n_symbols : int
        Number of possible symbols emitted by the model (in the observations).

    transmat : array, shape (`n_states`, `n_states`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_states`,)
        Initial state occupation distribution.

    emissionprob : array, shape ('n_states`, 'n_symbols`)
        Probability of emitting a given symbol when in each state.

    rates : array, shape ('n_states`,)
        Exponential rate parameters for each state.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more iterations the lower the frequency). If
        greater than 1 then it prints progress and performance for every
        iteration.

    Examples
    --------
    >>> from hmmlearn.hmm import MultinomialHMM
    >>> MultinomialHMM(n_states=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultinomialHMM(algorithm='viterbi',...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_states=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 emissionprob_prior=None, rates_var=1.0, algorithm="viterbi",
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters, init_params=string.ascii_letters,
                 verbose=0, batch_size=1):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_states : int
            Number of states.
        """
        _BaseHMM.__init__(self, n_states, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params,
                          verbose=verbose,
                          batch_size=batch_size)

        self.emissionprob_prior = emissionprob_prior
        self.rates_var = rates_var

    def _get_emissionprob(self):
        """Emission probability distribution for each state."""
        return np.exp(self._log_emissionprob)

    def _set_emissionprob(self, emissionprob):
        emissionprob = np.asarray(emissionprob)
        if hasattr(self, 'n_symbols') and \
                emissionprob.shape != (self.n_states, self.n_symbols):
            raise ValueError('emissionprob must have shape '
                             '(n_states, n_symbols)')

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(emissionprob):
            normalize(emissionprob)

        self._log_emissionprob = np.log(emissionprob)
        underflow_idx = np.isnan(self._log_emissionprob)
        self._log_emissionprob[underflow_idx] = NEGINF
        self.n_symbols = self._log_emissionprob.shape[1]

    emissionprob_ = property(_get_emissionprob, _set_emissionprob)

    def _get_rates(self):
        """Emission rate for each state."""
        return self._rates

    def _set_rates(self, rates):
        rates = np.asarray(rates)
        self._rates = rates.copy()

    rates_ = property(_get_rates, _set_rates)

    def _compute_log_likelihood(self, obs):
        return self._log_emissionprob[:, map(int, obs[:, 0])].T + \
            log_exponential_density(obs[:, 1], self._rates)

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        rand = random_state.rand()
        symbol = (cdf > rand).argmax()
        expon_obs = expon.rvs(scale=1. / self._rates[state])
        return symbol, expon_obs

    def _init(self, obs, params='ster'):
        super(MultinomialExponentialHMM, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

        if 'e' in params:
            if not hasattr(self, 'n_symbols'):
                symbols = set()
                for o in obs:
                    symbols = symbols.union(set(o[:, 0]))
                self.n_symbols = len(symbols)
            if self.emissionprob_prior is None:
                self.emissionprob_prior = np.ones((self.n_states,
                                                   self.n_symbols))
            emissionprob = np.vstack([np.random.dirichlet(
                self.emissionprob_prior[i])
                for i in xrange(self.n_states)])
            self.emissionprob_ = emissionprob

        concat_obs = np.concatenate(obs)[:, 1]
        if 'r' in params:
            clu = cluster.KMeans(n_clusters=self.n_states).fit(
                np.atleast_2d(concat_obs).T)
            rates = normal(0, self.rates_var, self.n_states) + \
                1. / clu.cluster_centers_.T[0]
            self._rates = np.maximum(0.1, rates)

    def _initialize_sufficient_statistics(self):
        stats = super(MultinomialExponentialHMM,
                      self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_states, self.n_symbols))
        stats['post'] = np.zeros(self.n_states)
        stats['expon_obs'] = np.zeros((self.n_states,))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(MultinomialExponentialHMM,
              self)._accumulate_sufficient_statistics(stats,
                                                      obs,
                                                      framelogprob,
                                                      posteriors,
                                                      fwdlattice,
                                                      bwdlattice,
                                                      params)
        if 'e' in params:
            for t, symbol in enumerate(obs[:, 0]):
                stats['obs'][:, int(symbol)] += posteriors[t]
        if 'r' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['expon_obs'] += np.dot(posteriors.T, obs[:, 1])

    def _do_mstep(self, stats, params):
        super(MultinomialExponentialHMM, self)._do_mstep(stats, params)
        if 'e' in params:
            self.emissionprob_ = (stats['obs']
                                  / stats['obs'].sum(1)[:, np.newaxis])
        if 'r' in params:
            self._rates = stats['post'] / stats['expon_obs']

    def _check_input_symbols(self, obs):
        """check if input can be used for Multinomial.fit input must be both
        positive integer array and every element must be continuous.
        e.g. x = [0, 0, 2, 1, 3, 1, 1] is OK and y = [0, 0, 3, 5, 10] not
        """

        symbols = np.concatenate(obs)[:, 0]

        if symbols.dtype.kind not in ('i', 'f'):
            # input symbols must be integer
            return False

        if len(symbols) == 1:
            # input too short
            return False

        if np.any(symbols < 0):
            # input contains negative intiger
            return False

        symbols.sort()
        if np.any(np.diff(symbols) > 1):
            # input is discontinous
            return False

        symbols = np.concatenate(obs)[:, 1]
        if symbols.dtype.kind not in ('f', 'i'):
            # input symbols must be integer
            return False

        if len(symbols) == 1:
            # input too short
            return False

        if np.any(symbols < 0):
            # input contains negative intiger
            return False

        return True

    def _n_free_parameters(self):
        n_pars = (self.n_states - 1) * (self.n_states + 1)
        n_pars += self.n_states * (self.n_symbols - 1)
        n_pars += self.n_states
        return n_pars

    def fit(self, sc, obs, **kwargs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation. Alternatively, a list of strings,
            each of which is a filepath to a pickled object, being
            a list of array-like observation sequences.
        """
        err_msg = ("Input must be a list of non-negative integer arrays where "
                   "in all, every element must be continuous, but %s was "
                   "given.")

        cleaned_obs = [np.array(seq) for seq in obs]

        if not self._check_input_symbols(cleaned_obs):
            raise ValueError(err_msg % obs)

        return _BaseHMM.fit(self, sc, cleaned_obs, **kwargs)


class GMMHMM(_BaseHMM):
    """Hidden Markov Model with Gaussin mixture emissions

    Attributes
    ----------
    init_params : string, optional
        Controls which parameters are initialized prior to training. Can
        contain any combination of 's' for startprob, 't' for transmat, 'm'
        for means, 'c' for covars, and 'w' for GMM mixing weights.
        Defaults to all parameters.

    params : string, optional
        Controls which parameters are updated in the training process.  Can
        contain any combination of 's' for startprob, 't' for transmat, 'm' for
        means, and 'c' for covars, and 'w' for GMM mixing weights.
        Defaults to all parameters.

    n_states : int
        Number of states in the model.

    transmat : array, shape (`n_states`, `n_states`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_states`,)
        Initial state occupation distribution.

    gmms : array of GMM objects, length `n_states`
        GMM emission distributions for each state.

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more iterations the lower the frequency). If
        greater than 1 then it prints progress and performance for every
        iteration.

    var : float, default: 1.0
        Variance parameter to randomize the initialization of the GMM objects.
        The larger var, the greater the randomization.

    Examples
    --------
    >>> from hmmlearn.hmm import GMMHMM
    >>> GMMHMM(n_states=2, n_mix=10, covariance_type='diag')
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    GMMHMM(algorithm='viterbi', covariance_type='diag',...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_states=1, n_mix=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", gmms=None, covariance_type='diag',
                 covars_prior=1e-2, random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 verbose=0, means_var=1.0, batch_size=1):
        """Create a hidden Markov model with GMM emissions.

        Parameters
        ----------
        n_states : int
            Number of states.
        """
        _BaseHMM.__init__(self, n_states, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params,
                          verbose=verbose,
                          batch_size=batch_size)

        # XXX: Hotfit for n_mix that is incompatible with the scikit's
        # BaseEstimator API
        self.n_mix = n_mix
        self._covariance_type = covariance_type
        self.covars_prior = covars_prior
        self.gmms = gmms
        if gmms is None:
            gmms = []
            for x in range(self.n_states):
                if covariance_type is None:
                    g = GMM(n_mix)
                else:
                    g = GMM(n_mix, covariance_type=covariance_type)
                gmms.append(g)
        self.gmms_ = gmms
        self.means_var = means_var

    # Read-only properties.
    @property
    def covariance_type(self):
        """Covariance type of the model.

        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._covariance_type

    def _compute_log_likelihood(self, obs):
        return np.array([g.score(obs) for g in self.gmms_]).T

    def _generate_sample_from_state(self, state, random_state=None):
        return self.gmms_[state].sample(1, random_state=random_state).flatten()

    def _init(self, obs, params='stwmc'):
        super(GMMHMM, self)._init(obs, params=params)

        concat_obs = np.vstack(obs
                               .flatMap(lambda x: x)
                               .sample(False, 0.01)
                               .map(np.atleast_2d)
                               .collect())
        n_features = concat_obs.shape[1]

        for g in self.gmms_:
            g.set_params(init_params=params, n_iter=0)
            g.fit(concat_obs)
            means = np.array([multivariate_normal(
                mean,
                np.eye(n_features) * self.means_var)
                for mean in g.means_])
            g.means_ = means

    def _initialize_sufficient_statistics(self):
        stats = super(GMMHMM, self)._initialize_sufficient_statistics()
        stats['norm'] = [np.zeros(g.weights_.shape) for g in self.gmms_]
        stats['means'] = [np.zeros(np.shape(g.means_)) for g in self.gmms_]
        stats['covars'] = [np.zeros(np.shape(g.covars_)) for g in self.gmms_]
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(GMMHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        for state, g in enumerate(self.gmms_):
            _, tmp_gmm_posteriors = g.score_samples(obs)
            lgmm_posteriors = np.log(tmp_gmm_posteriors
                                     + np.finfo(np.float).eps) + \
                np.log(posteriors[:, state][:, np.newaxis]
                       + np.finfo(np.float).eps)
            gmm_posteriors = np.exp(lgmm_posteriors)
            tmp_gmm = GMM(g.n_components, covariance_type=g.covariance_type)
            n_features = g.means_.shape[1]
            tmp_gmm._set_covars(
                distribute_covar_matrix_to_match_covariance_type(
                    np.eye(n_features), g.covariance_type,
                    g.n_components))
            norm = tmp_gmm._do_mstep(obs, gmm_posteriors, params)

            if np.any(np.isnan(tmp_gmm.covars_)):
                raise ValueError

            stats['norm'][state] += norm
            if 'm' in params:
                stats['means'][state] += tmp_gmm.means_ * norm[:, np.newaxis]
            if 'c' in params:
                if tmp_gmm.covariance_type == 'tied':
                    stats['covars'][state] += tmp_gmm.covars_ * norm.sum()
                else:
                    cvnorm = np.copy(norm)
                    shape = np.ones(tmp_gmm.covars_.ndim)
                    shape[0] = np.shape(tmp_gmm.covars_)[0]
                    cvnorm.shape = shape
                    stats['covars'][state] += tmp_gmm.covars_ * cvnorm

    def _do_mstep(self, stats, params):
        super(GMMHMM, self)._do_mstep(stats, params)
        # All that is left to do is to apply covars_prior to the
        # parameters updated in _accumulate_sufficient_statistics.
        for state, g in enumerate(self.gmms_):
            n_features = g.means_.shape[1]
            norm = stats['norm'][state]
            if 'w' in params:
                g.weights_ = normalize(norm)
            if 'm' in params:
                g.means_ = stats['means'][state] / norm[:, np.newaxis]
            if 'c' in params:
                if g.covariance_type == 'tied':
                    g.covars_ = ((stats['covars'][state]
                                 + self.covars_prior * np.eye(n_features))
                                 / norm.sum())
                else:
                    cvnorm = np.copy(norm)
                    shape = np.ones(g.covars_.ndim)
                    shape[0] = np.shape(g.covars_)[0]
                    cvnorm.shape = shape
                    if (g.covariance_type in ['spherical', 'diag']):
                        g.covars_ = (stats['covars'][state] +
                                     self.covars_prior) / cvnorm
                    elif g.covariance_type == 'full':
                        eye = np.eye(n_features)
                        g.covars_ = ((stats['covars'][state]
                                     + self.covars_prior * eye[np.newaxis])
                                     / cvnorm)

    def _n_free_parameters(self):
        n_pars = (self.n_states - 1) * (self.n_states + 1)
        for g in self.gmms_:
            n_components = g.means_.shape[0]
            n_features = g.means_.shape[1]
            n_pars += n_components - 1
            n_pars += n_components * n_features
            if g.covariance_type == 'spherical':
                n_pars += n_components
            elif g.covariance_type == 'tied':
                n_pars += ((n_features + 1) * n_features) / 2
            elif g.covariance_type == 'diag':
                n_pars += n_components * n_features
            elif g.covariance_type == 'full':
                n_pars += n_components * ((n_features + 1) * n_features) / 2
        return n_pars
