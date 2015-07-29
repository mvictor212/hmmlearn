# Mixtures of Hidden Markov Models
#
# Author: Mason Victors <mason.victors@gmail.com>

"""
The :mod:`hmmlearn.mixhmm` module implements mixtures of
hidden Markov models.
"""

import string

import numpy as np

from copy import deepcopy
from sklearn.utils import check_random_state
from sklearn.utils.extmath import logsumexp
from sklearn.base import BaseEstimator

from .hmmspark import (GaussianHMM, MultinomialHMM,
                       PoissonHMM, ExponentialHMM,
                       MultinomialExponentialHMM, VerboseReporter,
                       randomize, normalize, log_normalize)

from . import _hmmc

__all__ = ['MultinomialMixHMM']

ZEROLOGPROB = -1e200
EPS = np.finfo(float).eps
NEGINF = -np.inf
decoder_algorithms = ("viterbi", "map")


def identity(x):
    return x


def _do_estep(seq, modelBroadcast):
    model = modelBroadcast.value
    stats = model._initialize_sufficient_statistics()

    logprobs = np.zeros(model.n_components)
    inner_stats = model._initialize_inner_sufficient_statistics()
    framelogprob = model._compute_log_likelihood(seq)
    for k, hmm in enumerate(model.hmms):
        lpr, fwdlattice = hmm._do_forward_pass(
            framelogprob[:, :, k])
        bwdlattice = hmm._do_backward_pass(framelogprob[:, :, k])
        gamma = fwdlattice + bwdlattice
        posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
        logprobs[k] = lpr + model._log_component_weights[k]
        model._accumulate_inner_sufficient_statistics(
            inner_stats, seq, framelogprob[:, :, k], posteriors,
            fwdlattice, bwdlattice, model.params, k,
            logprobs[k])
    model._accumulate_sufficient_statistics(stats, inner_stats,
                                            model.params)
    logprob = logsumexp(logprobs)
    return stats, logprob


def _score(seq, modelBroadcast):
    model = modelBroadcast.value
    seq = np.asarray(seq)
    framelogprob = model._compute_log_likelihood(seq)
    lpr = np.array([model.hmms[k]._do_forward_pass(
        framelogprob[:, :, k])[0]
        for k in range(model.n_components)])
    lpr += model._log_component_weights
    logprob = logsumexp(lpr)
    return logprob


def _merge_sum(params, n_components, obs, post, expon_obs, obsSquared, obsObsT, stats, additional_stats):
    if 'p' in params:
        stats['component_weights'] += additional_stats['component_weights']
    if 'h' in params:
        obs = ('obs' in stats['hmm_stats'][0])
        post = ('post' in stats['hmm_stats'][0])
        expon_obs = ('expon_obs' in stats['hmm_stats'][0])
        obsSquared = ('obs**2' in stats['hmm_stats'][0])
        obsObsT = ('obs*obs.T' in stats['hmm_stats'][0])
        for k in range(n_components):
            stats['hmm_stats'][k]['start'] += \
                additional_stats['hmm_stats'][k]['start']
            stats['hmm_stats'][k]['trans'] += \
                additional_stats['hmm_stats'][k]['trans']
            if obs:
                stats['hmm_stats'][k]['obs'] += \
                    additional_stats['hmm_stats'][k]['obs']
            if post:
                stats['hmm_stats'][k]['post'] += \
                    additional_stats['hmm_stats'][k]['post']
            if expon_obs:
                stats['hmm_stats'][k]['expon_obs'] += \
                    additional_stats['hmm_stats'][k]['expon_obs']
            if obsSquared:
                stats['hmm_stats'][k]['obs**2'] += \
                    additional_stats['hmm_stats'][k]['obs**2']
            if obsObsT:
                stats['hmm_stats'][k]['obs*obs.T'] += \
                    additional_stats['hmm_stats'][k]['obs*obs.T']
    return stats


class _BaseMixHMM(BaseEstimator):
    """Hidden Markov Model base class.

    Representation of a mixture of hidden Markov models.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a MixHMM.

    See the instance documentation for details specific to a
    particular object.

    Attributes
    ----------
    n_components : int
        Number of mixture components in the model.

    n_states : int
        Number of states in the model.

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    transmat_prior : array, shape (`n_components`, `n_components`)
        Matrix of prior transition probabilities between states.

    startprob_prior : array, shape ('n_components`,)
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

    def __init__(self, n_components=1, n_states=1, hmms=None,
                 component_weights=None, component_weights_prior=None,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters, verbose=0,
                 tied=True):

        self.n_components = n_components
        self.n_states = n_states
        self.hmms = hmms
        self.n_iter = n_iter
        self.thresh = thresh
        self.params = params
        self.init_params = init_params
        self.component_weights_ = component_weights
        if component_weights_prior is None:
            component_weights_prior = np.ones(n_components)
        self.component_weights_prior = component_weights_prior
        self.random_state = random_state
        self.verbose = verbose
        self.tied = tied

    def score_samples(self, obs):
        """Return the per-sequence likelihood of the data under the model.

        Compute the log probability of obs under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each sequence of obs.

        Parameters
        ----------
        obs: list (n_sequences)
            List of sequences. Each sequence represents a hidden state
            sequence from one of the mixture's HMM components.

        Returns
        -------
        logprob : array_like, shape (n_sequences,)
            Log probabilities of each sequence in obs.

        responsibilities : array_like, shape (n_sequences, n_components)
            Posterior probabilities of each mixture component for each
            observation.
        """
        n_sequences = len(obs)
        logprob = np.zeros(n_sequences)
        responsibilities = np.zeros((n_sequences, self.n_components))
        for i, seq in enumerate(obs):
            framelogprob = self._compute_log_likelihood(seq)
            posteriors = np.array([self.hmms[k]._do_forward_pass(
                framelogprob[:, :, k])[0]
                for k in range(self.n_components)])
            posteriors += self._log_component_weights
            logprob[i] = logsumexp(posteriors)
            responsibilities[i, :] = log_normalize(posteriors, 0)
        return sum(logprob), responsibilities

    def score(self, sc, data):
        """Compute the log probability under the model.

        Parameters
        ----------
        obs : list (n_sequences)
            List of sequences. Each sequence represents a hidden state
            sequence from one of the mixture's HMM components.

        Returns
        -------
        logprob : float
            Log likelihood of the ``obs``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors
        """
        modelBroadcast = sc.broadcast(self)
        logprob = data.map(lambda seq: _score(seq, modelBroadcast)).sum()
        return logprob

    def aic(self, sc, data):
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
        logprob = self.score(sc, data)
        n_pars = self._n_free_parameters()
        aic_score = 2 * n_pars - 2 * logprob
        return aic_score

    def bic(self, sc, data):
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
        logprob = self.score(sc, data)
        n_pars = self._n_free_parameters()
        n_data = data.map(len).sum()
        bic_score = n_pars * (np.log(n_data) - np.log(2 * np.pi)) - 2 * logprob
        return bic_score

    def predict(self, obs):
        """Predict component label for observation sequences.

        Parameters
        ----------
        obs : list (n_sequences)
            List of sequences. Each sequence represents a hidden state
            sequence from one of the mixture's HMM components.

        Returns
        -------
        components : array, shape = (n_sequences,)
        """
        logprob, responsibilities = self.score_samples(obs)
        components = responsibilities.argmax(axis=1)
        return components

    def predict_proba(self, obs):
        """Predict posterior probability of sequences under each HMM
        in the model.

        Parameters
        ----------
        obs : list (n_sequences)
            List of sequences. Each sequence represents a hidden state
            sequence from one of the mixture's HMM components.

        Returns
        -------
        responsibilities : array_like, shape (n_sequences, n_components)
            Posterior probabilities of each mixture component for each
            observation.
        """
        logprob, responsibilities = self.score_samples(obs)
        return responsibilities

    def sample(self, n_seq=1, n_min=10, n_max=20, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_seq : int
            Number of sequences to generate.

        n : int
            Number of samples to generate.

        random_state: RandomState or an int seed (0 by default)
            A random number generator instance. If None is given, the
            object's random_state is used

        Returns
        -------
        (obs, states)
        obs : array_like, length `n_seq` List of observations sequences
        states : array_like, length `n_seq` List of state sequences
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        component_weights_pdf = self.component_weights_
        component_weights_cdf = np.cumsum(component_weights_pdf)

        components = []
        obs = []
        states = []
        for _ in range(n_seq):
            rand = random_state.rand()
            currcomponent = (component_weights_cdf > rand).argmax()
            components.append(currcomponent)

            n = np.random.randint(n_min, n_max)
            obs_seq, state_seq = self.hmms[currcomponent].sample(
                n_seq=1,
                n_min=n,
                n_max=(n + 1),
                random_state=random_state)

            obs.append(deepcopy(obs_seq[0]))
            states.append(deepcopy(state_seq[0]))

        return np.array(components), obs, states

    def fit(self, sc, data, warm_start=False):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.

        Notes
        -----
        In general, `logprob` should be non-decreasing unless
        aggressive pruning is used.  Decreasing `logprob` is generally
        a sign of overfitting (e.g. a covariance parameter getting too
        small).  You can fix this by getting more training data,
        or strengthening the appropriate subclass-specific regularization
        parameter.
        """

        if not warm_start:
            self._init(data, self.init_params)

        if self.verbose:
            verbose_reporter = VerboseReporter(self.verbose)
            verbose_reporter.init()

        data.cache()

        tmp_stats = self._initialize_sufficient_statistics()
        obs = ('obs' in tmp_stats['hmm_stats'][0])
        post = ('post' in tmp_stats['hmm_stats'][0])
        expon_obs = ('expon_obs' in tmp_stats['hmm_stats'][0])
        obsSquared = ('obs**2' in tmp_stats['hmm_stats'][0])
        obsObsT = ('obs*obs.T' in tmp_stats['hmm_stats'][0])

        params = self.params

        def merge_sum(stats, additional_stats):
            return _merge_sum(params, obs, post, expon_obs, obsSquared,
                              obsObsT, stats, additional_stats)

        logprob = []
        for i in range(self.n_iter):
            # Expectation step
            modelBroadcast = sc.broadcast(self)
            results = data.map(lambda seq: _do_estep(seq,
                                                     modelBroadcast)).cache()
            stats = results.keys().reduce(merge_sum)
            curr_logprob = results.values().sum()

            logprob.append(curr_logprob)

            if i > 0:
                improvement = logprob[-1] - logprob[-2]
            else:
                improvement = np.inf
            if self.verbose:
                verbose_reporter.update(i, logprob[-1], improvement)

            # Check for convergence.
            if i > 0 and abs(logprob[-1] - logprob[-2]) < self.thresh:
                break

            # Maximization step
            self._do_mstep(stats, self.params)

        return self

    def _get_component_weights(self):
        """Component weights for each component."""
        return np.exp(self._log_component_weights)

    def _set_component_weights(self, component_weights):
        if component_weights is None:
            component_weights = randomize(np.tile(1.0 / self.n_components,
                                                  self.n_components))
        else:
            component_weights = np.asarray(component_weights, dtype=np.float)

        if not np.alltrue(component_weights):
            normalize(component_weights)

        if len(component_weights) != self.n_components:
            raise ValueError('component_weights must have length \
                             `n_components`')
        if not np.allclose(np.sum(component_weights), 1.0):
            raise ValueError('component_weights must sum to 1.0')

        self._log_component_weights = np.log(np.asarray(
            component_weights).copy())

    component_weights_ = property(_get_component_weights,
                                  _set_component_weights)

    def _compute_log_likelihood(self, obs):
        return np.array([self.hmms[i]._compute_log_likelihood(obs).T
                         for i in xrange(self.n_components)]).T

    def _generate_sample_from_state(self, component, state, random_state=None):
        pass

    def _init(self, obs, params):
        if 'p' in params:
            self.component_weights_ = np.random.dirichlet(
                self.component_weights_prior)

    # Methods used by self.fit()

    def _initialize_sufficient_statistics(self):
        stats = {'component_weights': np.zeros(self.n_components),
                 'hmm_stats': [hmm._initialize_sufficient_statistics()
                               for hmm in self.hmms]}
        return stats

    def _initialize_inner_sufficient_statistics(self):
        stats = {'component_weights': np.zeros(self.n_components),
                 'start': [np.zeros(hmm.n_states) for hmm in self.hmms],
                 'trans': [np.zeros((hmm.n_states, hmm.n_states))
                           for hmm in self.hmms]}
        return stats

    def _accumulate_inner_sufficient_statistics(self, stats, seq,
                                                framelogprob, posteriors,
                                                fwdlattice, bwdlattice,
                                                params, k, curr_logprob):
        stats['component_weights'][k] += curr_logprob
        if 'h' in params:
            stats['start'][k] += posteriors[0]
            n_observations, n_states = framelogprob.shape
            if n_observations > 1:
                lneta = np.zeros((n_observations - 1, n_states, n_states))
                lnP = logsumexp(fwdlattice[-1])
                _hmmc._compute_lneta(n_observations, n_states, fwdlattice,
                                     self.hmms[k]._log_transmat, bwdlattice,
                                     framelogprob, lnP, lneta)
                stats['trans'][k] += np.exp(np.minimum(logsumexp(lneta, 0),
                                                       700))

    def _accumulate_sufficient_statistics(self, stats, inner_stats, params):
        component_weights = log_normalize(inner_stats['component_weights'], 0)
        if 'p' in params:
            stats['component_weights'] += component_weights
        if 'h' in params:
            for k in range(self.n_components):
                stats['hmm_stats'][k]['start'] += component_weights[k] * \
                    inner_stats['start'][k]
                stats['hmm_stats'][k]['trans'] += component_weights[k] * \
                    inner_stats['trans'][k]

    def _do_mstep(self, stats, params):
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        if self.component_weights_prior is None:
            self.component_weights_prior = 1.0

        if 'p' in params:
            self.component_weights_ = normalize(
                np.maximum(self.component_weights_prior - 1.0 +
                           stats['component_weights'], 1e-20))
        if 'h' in params:
            for k, hmm in enumerate(self.hmms):
                hmm._do_mstep(stats['hmm_stats'][k], hmm.params)

    def _n_free_parameters(self):
        pass


class MultinomialMixHMM(_BaseMixHMM):
    """Mixture of Hidden Markov Models with multinomial (discrete) emissions

    Attributes
    ----------
    n_components : int
        Number of HMMs in the model.

    n_states : int
        Number of states in each HMM

    n_symbols : int
        Number of possible symbols emitted by the model (in the observations).

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    emissionprob : array, shape ('n_components`, 'n_symbols`)
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
    >>> from hmmlearn.mixhmm import MultinomialMixHMM
    >>> MultinomialMixHMM(n_components=2, n_states=3)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultinomialMixHMM(...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_components=1, n_states=1,
                 hmms=None, component_weights=None,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 verbose=0, emissionprob_prior=None, tied=True):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_components : int
            Number of HMM components.
        """
        _BaseMixHMM.__init__(self, n_components, n_states,
                             hmms=hmms,
                             component_weights=component_weights,
                             random_state=random_state,
                             n_iter=n_iter,
                             thresh=thresh,
                             params=params,
                             init_params=init_params,
                             verbose=verbose,
                             tied=tied)
        self.emissionprob_prior = emissionprob_prior

    def _init(self, obs, params='ph'):
        super(MultinomialMixHMM, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

        if ('h' in params) and (self.hmms is None):
            self.hmms = [MultinomialHMM(
                self.n_states,
                emissionprob_prior=self.emissionprob_prior)
                for _ in range(self.n_components)]
            self.hmms[0]._init(obs)
            emissionprob = self.hmms[0].emissionprob_
            for hmm in self.hmms[1:]:
                if self.tied:
                    hmm.emissionprob_ = emissionprob
                else:
                    hmm._init(obs)

    def _initialize_inner_sufficient_statistics(self):
        stats = super(MultinomialMixHMM,
                      self)._initialize_inner_sufficient_statistics()
        stats['obs'] = [np.zeros((hmm.n_states, hmm.n_symbols))
                        for hmm in self.hmms]
        return stats

    def _accumulate_inner_sufficient_statistics(self, stats, obs, framelogprob,
                                                posteriors, fwdlattice,
                                                bwdlattice, params,
                                                k, currlogprob):
        super(MultinomialMixHMM, self)._accumulate_inner_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params, k, currlogprob)
        if 'h' in params:
            for t, symbol in enumerate(obs):
                stats['obs'][k][:, symbol] += posteriors[t]

    def _accumulate_sufficient_statistics(self, stats, inner_stats, params):
        super(MultinomialMixHMM, self)._accumulate_sufficient_statistics(
            stats, inner_stats, params)
        component_weights = log_normalize(inner_stats['component_weights'], 0)
        if 'h' in params:
            if self.tied:
                tmp_obs_stats = np.zeros(np.shape(inner_stats['obs'][0]))
                for k in range(self.n_components):
                    tmp_obs_stats += component_weights[k] * \
                        inner_stats['obs'][k]
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['obs'] += tmp_obs_stats
            else:
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['obs'] += component_weights[k] * \
                        inner_stats['obs'][k]

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
        n_pars = self.n_components - 1
        for hmm in self.hmms:
            n_pars += hmm._n_free_parameters()
        if self.tied:
            n_pars -= (self.n_components - 1) * \
                (self.hmms[0].n_states * (self.hmms[0].n_symbols - 1))
        return n_pars

    def fit(self, sc, data, warm_start=False, **kwargs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.
        """
        err_msg = ("Input must be a list of non-negative integer arrays where "
                   "in all, every element must be continuous, but %s was "
                   "given.")

        modelBroadcast = sc.broadcast(self)

        if not data.map(lambda seq: modelBroadcast.value._check_input_symbols(seq)).min():
            raise ValueError(err_msg % data.take(5))
        elif np.any(np.diff(data.flatMap(identity).distinct().sortBy(identity).collect()) > 1):
            raise ValueError(err_msg % data.take(5))

        return super(MultinomialMixHMM, self).fit(self, sc, data, warm_start, **kwargs)


class MultinomialExponentialMixHMM(_BaseMixHMM):
    """Mixture of Hidden Markov Models with multinomial and exponential emissions

    Attributes
    ----------
    n_components : int
        Number of HMMs in the model.

    n_states : int
        Number of states in each HMM

    n_symbols : int
        Number of possible symbols emitted by the model (in the observations).

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    emissionprob : array, shape ('n_components`, 'n_symbols`)
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
    >>> from hmmlearn.mixhmm import MultinomialMixHMM
    >>> MultinomialMixHMM(n_components=2, n_states=3)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultinomialMixHMM(...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_components=1, n_states=1,
                 hmms=None, component_weights=None,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 verbose=0, emissionprob_prior=None,
                 rates_var=1.0, tied=True):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_components : int
            Number of HMM components.
        """
        _BaseMixHMM.__init__(self, n_components, n_states,
                             hmms=hmms,
                             component_weights=component_weights,
                             random_state=random_state,
                             n_iter=n_iter,
                             thresh=thresh,
                             params=params,
                             init_params=init_params,
                             verbose=verbose,
                             tied=tied)
        self.emissionprob_prior = emissionprob_prior
        self.rates_var = rates_var

    def _init(self, obs, params='ph'):
        super(MultinomialExponentialMixHMM, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

        if ('h' in params) and (self.hmms is None):
            self.hmms = [MultinomialExponentialHMM(
                self.n_states,
                emissionprob_prior=self.emissionprob_prior,
                rates_var=self.rates_var)
                for _ in range(self.n_components)]
            self.hmms[0]._init(obs)
            emissionprob = self.hmms[0].emissionprob_
            rates = self.hmms[0].rates_
            for hmm in self.hmms[1:]:
                if self.tied:
                    hmm.emissionprob_ = emissionprob
                    hmm.rates_ = rates
                else:
                    hmm._init(obs)

    def _initialize_inner_sufficient_statistics(self):
        stats = super(MultinomialExponentialMixHMM,
                      self)._initialize_inner_sufficient_statistics()
        stats['obs'] = [np.zeros((hmm.n_states, hmm.n_symbols))
                            for hmm in self.hmms]
        stats['post'] = [np.zeros(hmm.n_states) for hmm in self.hmms]
        stats['expon_obs'] = [np.zeros((hmm.n_states,)) for hmm in self.hmms]
        return stats

    def _accumulate_inner_sufficient_statistics(self, stats, obs, framelogprob,
                                                posteriors, fwdlattice,
                                                bwdlattice, params,
                                                k, currlogprob):
        super(MultinomialExponentialMixHMM,
              self)._accumulate_inner_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params, k, currlogprob)
        if 'h' in params:
            for t, symbol in enumerate(obs[:, 0]):
                stats['obs'][k][:, int(symbol)] += posteriors[t]
            stats['post'][k] += posteriors.sum(axis=0)
            stats['expon_obs'][k] += np.dot(posteriors.T, obs[:, 1])

    def _accumulate_sufficient_statistics(self, stats, inner_stats, params):
        super(MultinomialExponentialMixHMM,
              self)._accumulate_sufficient_statistics(
            stats, inner_stats, params)
        component_weights = log_normalize(inner_stats['component_weights'], 0)
        if 'h' in params:
            if self.tied:
                tmp_obs_stats = np.zeros(np.shape(inner_stats['obs'][0]))
                tmp_post_stats = np.zeros(np.shape(inner_stats['post'][0]))
                tmp_expon_obs_stats = np.zeros(np.shape(
                    inner_stats['expon_obs'][0]))
                for k in range(self.n_components):
                    tmp_obs_stats += component_weights[k] * \
                        inner_stats['obs'][k]
                    tmp_post_stats += component_weights[k] * \
                        inner_stats['post'][k]
                    tmp_expon_obs_stats += component_weights[k] * \
                        inner_stats['expon_obs'][k]
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['obs'] += tmp_obs_stats
                    stats['hmm_stats'][k]['post'] += tmp_post_stats
                    stats['hmm_stats'][k]['expon_obs'] += tmp_expon_obs_stats
            else:
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['obs'] += component_weights[k] * \
                        inner_stats['obs'][k]
                    stats['hmm_stats'][k]['post'] += component_weights[k] * \
                        inner_stats['post'][k]
                    stats['hmm_stats'][k]['expon_obs'] += component_weights[k] * \
                        inner_stats['expon_obs'][k]

    def _check_input_symbols(self, obs):
        """check if input can be used for Multinomial.fit input must be both
        positive integer array and every element must be continuous.
        e.g. x = [0, 0, 2, 1, 3, 1, 1] is OK and y = [0, 0, 3, 5, 10] not
        """

        symbols = obs[:, 0]

        if symbols.dtype.kind not in ('i', 'f'):
            # input symbols must be integer
            return False

        if len(symbols) == 1:
            # input too short
            return False

        if np.any(symbols < 0):
            # input contains negative intiger
            return False

        symbols = obs[:, 1]
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
        n_pars = self.n_components - 1
        for hmm in self.hmms:
            n_pars += hmm._n_free_parameters()
        if self.tied:
            n_pars -= (self.n_components - 1) * \
                (self.hmms[0].n_states * (self.hmms[0].n_symbols - 1))
            n_pars -= (self.n_components - 1) * self.hmms[0].n_states
        return n_pars

    def fit(self, sc, data, warm_start=False, **kwargs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.
        """
        err_msg = ("Input must be a list of non-negative integer arrays where "
                   "in all, every element must be continuous, but %s was "
                   "given.")

        modelBroadcast = sc.broadcast(self)

        cleaned_data = data.map(lambda seq: np.array(seq))

        if not cleaned_data.map(lambda seq: modelBroadcast.value._check_input_symbols(seq)).min():
            raise ValueError(err_msg % cleaned_data.take(5))
        elif np.any(np.diff(cleaned_data.flatMap(lambda seq: seq[:, 0]).distinct().sortBy(identity).collect()) > 1):
            raise ValueError(err_msg % cleaned_data.take(5))

        return super(MultinomialExponentialMixHMM, self).fit(self, sc, cleaned_data, warm_start, **kwargs)


class PoissonMixHMM(_BaseMixHMM):
    """Mixture of Hidden Markov Models with Poisson (discrete) emissions

    Attributes
    ----------
    n_components : int
        Number of HMMs in the model.

    n_states : int
        Number of states in each HMM

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    emissionprob : array, shape ('n_components`, 'n_symbols`)
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
    >>> from hmmlearn.mixhmm import PoissonMixHMM
    >>> PoissonMixHMM(n_components=2, n_states=3)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    PoissonMixHMM(...)

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_components=1, n_states=1,
                 hmms=None, component_weights=None,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 verbose=0, rates_var=1.0, tied=True):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_components : int
            Number of HMM components.
        """
        _BaseMixHMM.__init__(self, n_components, n_states,
                             hmms=hmms,
                             component_weights=component_weights,
                             random_state=random_state,
                             n_iter=n_iter,
                             thresh=thresh,
                             params=params,
                             init_params=init_params,
                             verbose=verbose,
                             tied=tied)
        self.rates_var = rates_var

    def _init(self, obs, params='ph'):
        super(PoissonMixHMM, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

        if ('h' in params) and (self.hmms is None):
            self.hmms = [PoissonHMM(self.n_states,
                                    rates_var=self.rates_var)
                         for _ in range(self.n_components)]
            self.hmms[0]._init(obs)
            rates = self.hmms[0].rates_
            for hmm in self.hmms[1:]:
                if self.tied:
                    hmm.rates_ = rates
                else:
                    hmm._init(obs)

    def _initialize_inner_sufficient_statistics(self):
        stats = super(PoissonMixHMM,
                      self)._initialize_inner_sufficient_statistics()
        stats['post'] = [np.zeros(hmm.n_states) for hmm in self.hmms]
        stats['obs'] = [np.zeros((hmm.n_states,)) for hmm in self.hmms]
        return stats

    def _accumulate_inner_sufficient_statistics(self, stats, obs, framelogprob,
                                                posteriors, fwdlattice,
                                                bwdlattice, params,
                                                k, currlogprob):
        super(PoissonMixHMM, self)._accumulate_inner_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params, k, currlogprob)
        if 'h' in params:
            stats['post'][k] += posteriors.sum(axis=0)
            stats['obs'][k] += np.dot(posteriors.T, obs)

    def _accumulate_sufficient_statistics(self, stats, inner_stats, params):
        super(PoissonMixHMM, self)._accumulate_sufficient_statistics(
            stats, inner_stats, params)
        component_weights = log_normalize(inner_stats['component_weights'], 0)
        if 'h' in params:
            if self.tied:
                tmp_post_stats = np.zeros(np.shape(inner_stats['post'][0]))
                tmp_obs_stats = np.zeros(np.shape(inner_stats['obs'][0]))
                for k in range(self.n_components):
                    tmp_post_stats += component_weights[k] * \
                        inner_stats['post'][k]
                    tmp_obs_stats += component_weights[k] * \
                        inner_stats['obs'][k]
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['post'] += tmp_post_stats
                    stats['hmm_stats'][k]['obs'] += tmp_obs_stats
            else:
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['post'] += component_weights[k] * \
                        inner_stats['post'][k]
                    stats['hmm_stats'][k]['obs'] += component_weights[k] * \
                        inner_stats['obs'][k]

    def _check_input_symbols(self, obs):
        """check if input can be used for PoissonMixHMM. Input must be a list
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
        n_pars = self.n_components - 1
        for hmm in self.hmms:
            n_pars += hmm._n_free_parameters()
        if self.tied:
            n_pars -= (self.n_components - 1) * self.hmms[0].n_states
        return n_pars

    def fit(self, sc, data, warm_start=False, **kwargs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.
        """
        err_msg = ("Input must be a list of non-negative integer arrays, \
                   but %s was given.")

        modelBroadcast = sc.broadcast(self)

        if not data.map(lambda seq: modelBroadcast.value._check_input_symbols(seq)).min():
            raise ValueError(err_msg % data.take(5))

        return super(PoissonMixHMM, self).fit(self, sc, data, warm_start, **kwargs)


class ExponentialMixHMM(_BaseMixHMM):
    """Mixture of Hidden Markov Models with Exponential emissions

    Attributes
    ----------
    n_components : int
        Number of HMMs in the model.

    n_states : int
        Number of states in each HMM

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    emissionprob : array, shape ('n_components`, 'n_symbols`)
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
    >>> from hmmlearn.mixhmm import ExponentialMixHMM
    >>> ExponentialMixHMM(n_components=2, n_states=3)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ExponentialMixHMM(...)

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_components=1, n_states=1,
                 hmms=None, component_weights=None,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 verbose=0, rates_var=1.0, tied=True):
        """Create a hidden Markov model with exponential emissions.

        Parameters
        ----------
        n_components : int
            Number of HMM components.
        """
        _BaseMixHMM.__init__(self, n_components, n_states,
                             hmms=hmms,
                             component_weights=component_weights,
                             random_state=random_state,
                             n_iter=n_iter,
                             thresh=thresh,
                             params=params,
                             init_params=init_params,
                             verbose=verbose,
                             tied=tied)
        self.rates_var = rates_var

    def _init(self, obs, params='ph'):
        super(ExponentialMixHMM, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

        if ('h' in params) and (self.hmms is None):
            self.hmms = [ExponentialHMM(self.n_states,
                                        rates_var=self.rates_var)
                         for _ in range(self.n_components)]
            self.hmms[0]._init(obs)
            rates = self.hmms[0].rates_
            for hmm in self.hmms[1:]:
                if self.tied:
                    hmm.rates_ = rates
                else:
                    hmm._init(obs)

    def _initialize_inner_sufficient_statistics(self):
        stats = super(ExponentialMixHMM,
                      self)._initialize_inner_sufficient_statistics()
        stats['post'] = [np.zeros(hmm.n_states) for hmm in self.hmms]
        stats['obs'] = [np.zeros((hmm.n_states,)) for hmm in self.hmms]
        return stats

    def _accumulate_inner_sufficient_statistics(self, stats, obs, framelogprob,
                                                posteriors, fwdlattice,
                                                bwdlattice, params,
                                                k, currlogprob):
        super(ExponentialMixHMM, self)._accumulate_inner_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params, k, currlogprob)
        if 'h' in params:
            stats['post'][k] += posteriors.sum(axis=0)
            stats['obs'][k] += np.dot(posteriors.T, obs)

    def _accumulate_sufficient_statistics(self, stats, inner_stats, params):
        super(ExponentialMixHMM, self)._accumulate_sufficient_statistics(
            stats, inner_stats, params)
        component_weights = log_normalize(inner_stats['component_weights'], 0)
        if 'h' in params:
            if self.tied:
                tmp_post_stats = np.zeros(np.shape(inner_stats['post'][0]))
                tmp_obs_stats = np.zeros(np.shape(inner_stats['obs'][0]))
                for k in range(self.n_components):
                    tmp_post_stats += component_weights[k] * \
                        inner_stats['post'][k]
                    tmp_obs_stats += component_weights[k] * \
                        inner_stats['obs'][k]
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['post'] += tmp_post_stats
                    stats['hmm_stats'][k]['obs'] += tmp_obs_stats
            else:
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['post'] += component_weights[k] * \
                        inner_stats['post'][k]
                    stats['hmm_stats'][k]['obs'] += component_weights[k] * \
                        inner_stats['obs'][k]

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
        n_pars = self.n_components - 1
        for hmm in self.hmms:
            n_pars += hmm._n_free_parameters()
        if self.tied:
            n_pars -= (self.n_components - 1) * self.hmms[0].n_states
        return n_pars

    def fit(self, sc, data, warm_start=False, **kwargs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.
        """
        err_msg = ("Input must be a list of non-negative real arrays, \
                   but %s was given.")

        modelBroadcast = sc.broadcast(self)

        if not data.map(lambda seq: modelBroadcast.value._check_input_symbols(seq)).min():
            raise ValueError(err_msg % data.take(5))

        return super(ExponentialMixHMM, self).fit(self, sc, data, warm_start, **kwargs)


class GaussianMixHMM(_BaseMixHMM):
    """Mixture of Hidden Markov Models with Gaussian emissions

    Attributes
    ----------
    n_components : int
        Number of HMMs in the model.

    n_states : int
        Number of states in each HMM

    n_features : int
        Dimensionality of the Gaussian emissions.

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    emissionprob : array, shape ('n_components`, 'n_symbols`)
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
    >>> from hmmlearn.mixhmm import ExponentialMixHMM
    >>> ExponentialMixHMM(n_components=2, n_states=3)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ExponentialMixHMM(...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_components=1, n_states=1, n_features=1,
                 hmms=None, component_weights=None,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 verbose=0, means_var=1.0, tied=True):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_components : int
            Number of HMM components.
        """
        _BaseMixHMM.__init__(self, n_components, n_states,
                             hmms=hmms,
                             component_weights=component_weights,
                             random_state=random_state,
                             n_iter=n_iter,
                             thresh=thresh,
                             params=params,
                             init_params=init_params,
                             verbose=verbose,
                             tied=tied)
        self.means_var = means_var

    def _init(self, obs, params='ph'):
        super(GaussianMixHMM, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

        if ('h' in params) and (self.hmms is None):
            self.hmms = [GaussianHMM(self.n_states,
                                     means_var=self.means_var)
                         for _ in range(self.n_components)]
            self.hmms[0]._init(obs)
            means = self.hmms[0].means_
            covars = self.hmms[0]._covars_
            for hmm in self.hmms[1:]:
                if self.tied:
                    hmm.means_ = means
                    hmm.covars_ = covars
                else:
                    hmm._init(obs)

    def _initialize_inner_sufficient_statistics(self):
        stats = super(GaussianMixHMM,
                      self)._initialize_inner_sufficient_statistics()
        stats['post'] = [np.zeros(hmm.n_states) for hmm in self.hmms]
        stats['obs'] = [np.zeros((hmm.n_states, hmm.n_features))
                        for hmm in self.hmms]
        stats['obs**2'] = [np.zeros((hmm.n_states, hmm.n_features))
                           for hmm in self.hmms]
        stats['obs*obs.T'] = [np.zeros((hmm.n_states, hmm.n_features,
                                        hmm.n_features))
                              for hmm in self.hmms]
        return stats

    def _accumulate_inner_sufficient_statistics(self, stats, obs, framelogprob,
                                                posteriors, fwdlattice,
                                                bwdlattice, params,
                                                k, currlogprob):
        super(GaussianMixHMM, self)._accumulate_inner_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params, k, currlogprob)
        if 'h' in params:
            stats['post'][k] += posteriors.sum(axis=0)
            stats['obs'][k] += np.dot(posteriors.T, obs)
            if self.hmms[k]._covariance_type in ('spherical', 'diag'):
                stats['obs**2'][k] += np.dot(posteriors.T, obs ** 2)
            elif self.hmms[k]._covariance_type in ('tied', 'full'):
                for t, o in enumerate(obs):
                    obsobsT = np.outer(o, o)
                    for c in range(self.hmms[k].n_states):
                        stats['obs*obs.T'][k][c] += posteriors[t, c] * obsobsT

    def _accumulate_sufficient_statistics(self, stats, inner_stats, params):
        super(GaussianMixHMM, self)._accumulate_sufficient_statistics(
            stats, inner_stats, params)
        component_weights = log_normalize(inner_stats['component_weights'], 0)
        if 'h' in params:
            if self.tied:
                tmp_post_stats = np.zeros(np.shape(inner_stats['post'][0]))
                tmp_obs_stats = np.zeros(np.shape(inner_stats['obs'][0]))
                if self.hmms[0]._covariance_type in ('spherical', 'diag'):
                    tmp_obs2_stats = np.zeros(np.shape(inner_stats['obs**2'][0]))
                elif self.hmms[0]._covariante_type in ('tied', 'full'):
                    tmp_obsobsT_stats = np.zeros(np.shape(inner_stats['obs*obs.T'][0]))
                for k in range(self.n_components):
                    tmp_post_stats += component_weights[k] * \
                        inner_stats['post'][k]
                    tmp_obs_stats += component_weights[k] * \
                        inner_stats['obs'][k]
                    if self.hmms[0]._covariance_type in ('spherical', 'diag'):
                        tmp_obs2_stats += component_weights[k] * \
                            inner_stats['obs**2'][k]
                    elif self.hmms[0]._covariance_type in ('tied', 'full'):
                       tmp_obsobsT_stats += component_weights[k] * \
                            inner_stats['obs*obs.T'][k]
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['post'] += tmp_post_stats
                    stats['hmm_stats'][k]['obs'] += tmp_obs_stats
                    if self.hmms[k]._covariance_type in ('spherical', 'diag'):
                        stats['hmm_stats'][k]['obs**2'] += tmp_obs2_stats
                    elif self.hmms[k]._covariance_type in ('tied', 'full'):
                        stats['hmm_stats'][k]['obs*obs.T'] += tmp_obsobsT_stats
            else:
                for k in range(self.n_components):
                    stats['hmm_stats'][k]['post'] += component_weights[k] * \
                        inner_stats['post'][k]
                    stats['hmm_stats'][k]['obs'] += component_weights[k] * \
                        inner_stats['obs'][k]
                    if self.hmms[k]._covariance_type in ('spherical', 'diag'):
                        stats['hmm_stats'][k]['obs**2'] += component_weights[k] * \
                            inner_stats['obs**2'][k]
                    elif self.hmms[k]._covariance_type in ('tied', 'full'):
                        stats['hmm_stats'][k]['obs*obs.T'] += \
                            component_weights[k] * inner_stats['obs*obs.T'][k]

    def fit(self, sc, data, warm_start=False, **kwargs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.
        """
        return super(GaussianMixHMM, self).fit(self, sc, data, warm_start, **kwargs)

    def _n_free_parameters(self):
        n_pars = self.n_components - 1
        for hmm in self.hmms:
            n_pars += hmm._n_free_parameters()
        if self.tied:
            covar_type = self.hmms[0]._covariance_type
            if covar_type == 'spherical':
                n_covar_pars = self.hmms[0].n_states
            elif covar_type == 'tied':
                n_covar_pars = ((self.hmms[0].n_features + 1) *
                                self.hmms[0].n_features) / 2
            elif covar_type == 'diag':
                n_covar_pars = self.hmms[0].n_states * self.hmms[0].n_features
            elif covar_type == 'full':
                n_covar_pars = self.hmms[0].n_states * \
                    ((self.hmms[0].n_features + 1) *
                     self.hmms[0].n_features) / 2
            n_pars -= (self.n_components - 1) * \
                (self.hmms[0].n_states * self.hmms[0].n_features +
                 n_covar_pars)
        return n_pars
