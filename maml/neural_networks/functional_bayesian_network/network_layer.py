__author__ = 'epyzerknapp', 'miguel'

import math
import theano
import theano.tensor as T


def network_layer(m_w_init, v_w_init, non_linear=True):
    """
    Overall function to drive the network layer functionality
    :param m_w_init: initial weights mean
    :param v_w_init: initial weights variance
    :param non_linear: bool  - whether the layer is linear or non-linear
    Note : currently implemented non-linear functionality in this net
    is the rectified linear unit (ReLU)
    :return: parameter dict
    """
    params = dict()
    # We create the theano variables for the means and variances

    params['m_w'] = theano.shared(value=m_w_init.astype(theano.config.floatX),
                                  name='m_w', borrow=True)
    params['v_w'] = theano.shared(value=v_w_init.astype(theano.config.floatX),
                                  name='v_w', borrow=True)

    # We store the type of activation function

    params['non_linear'] = non_linear
    return params


def _n_pdf(x):

    return 1.0 / T.sqrt(2 * math.pi) * T.exp(-0.5 * x**2)

def _n_cdf(x):

    return 0.5 * (1.0 + T.erf(x / T.sqrt(2.0)))

def _gamma(x):

    return _n_pdf(x) / _n_cdf(-x)

def _beta(x):

    return _gamma(x) * (_gamma(x) - x)

def output(m_w_previous, v_w_previous, params):
    """
    The output of a feed forward pass for the network layer.
    :param m_w_previous: The previous means for weights
    :param v_w_previous: The previous variances for the weights
    :param params: The params dictionary
    :return: means and variances resulting from the pass
    """

    # We add an additional deterministic input with mean 1 and variance 0

    m_w_previous_with_bias = \
        T.concatenate([ m_w_previous, T.alloc(1, 1) ], 0)
    v_w_previous_with_bias = \
        T.concatenate([ v_w_previous, T.alloc(0, 1) ], 0)

    # We compute the mean and variance after the linear operation

    m_linear = T.dot(params['m_w'], m_w_previous_with_bias)
    v_linear = T.dot(params['v_w'], v_w_previous_with_bias) + \
        T.dot(params['m_w']**2, v_w_previous_with_bias) + \
        T.dot(params['v_w'], m_w_previous_with_bias**2)

    if (params['non_linear']):

        # We compute the mean and variance after the ReLU activation

        alpha = m_linear / T.sqrt(v_linear)
        gamma = _gamma(-alpha)
        gamma_robust = -alpha - 1.0 / alpha + 2.0 / alpha**3
        gamma_final = T.switch(T.lt(-alpha, T.fill(alpha, 30)), gamma, gamma_robust)

        v_aux = m_linear + T.sqrt(v_linear) * gamma_final

        m_a = _n_cdf(alpha) * v_aux
        v_a = m_a * v_aux * _n_cdf(-alpha) + \
            _n_cdf(alpha) * v_linear * \
            (1 - gamma_final * (gamma_final + alpha))

        return m_a, v_a

    else:

        return m_linear, v_linear
