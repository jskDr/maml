__author__ = 'epyzerknapp', 'miguel'
import numpy as np
import theano
import theano.tensor as T
from maml.neural_networks.functional_bayesian_network.network_layer import network_layer


def network(m_w_init, v_w_init, a_init, b_init):
    params = dict()
    # We create the different layers

    params['layers'] = []

    if len(m_w_init) > 1:
        for m_w, v_w in zip(m_w_init[ : -1 ], v_w_init[ : -1 ]):
            params['layers'].append(network_layer(m_w, v_w, True))

    params['layers'].append(network_layer(m_w_init[ -1 ], v_w_init[ -1 ], False))

    # We create mean and variance parameters from all layers

    params['params_m_w'] = []
    params['params_v_w'] = []
    for layer in self.layers:
        params['params_m_w'].append(layer.m_w)
        params['params_v_w'].append(layer.v_w)

    # We create the theano variables for a and b

    params['a'] = theano.shared(float(a_init))
    params['b'] = theano.shared(float(b_init))
    return params

def output(m, params):
    """
    Recursively computes output of network
    :param params: params dictionary
    :param m: inputs
    :return:
    """
    v = T.zeros_like(m)
    for layer in params['layers']:
        m, v = layer.output(m, v)

    return m[0], v[0]

def predict(x, params):
    """
    Predict the targets for a set of inputs
    :param x: Input features for the neural network
    :param params : Params dictionary
    :return: means and variances for the prediction
    """
    m, v = output(x, params)

    return m, v + params['b'] / params['a'] - 1

def logZ_Z1_Z2(x, y, params):

    m, v = output(x, params)

    v_final = v + params['b'] / (params['a'] - 1)
    v_final1 = v + params['b'] / params['a']
    v_final2 = v + params['b'] / (params['a'] + 1)

    logz = -0.5 * (T.log(v_final) + (y - m)**2 / v_final)
    logz1 = -0.5 * (T.log(v_final1) + (y - m)**2 / v_final1)
    logz2 = -0.5 * (T.log(v_final2) + (y - m)**2 / v_final2)

    return logz, logz1, logz2

def generate_updates(logZ, logZ1, logZ2, params):
    """
    Generates weight updates for a pass through the network
    :param logZ:
    :param logZ1:
    :param logZ2:
    :param params:
    :return:
    """
    updates = []
    for i in range(len(params['params_m_w'])):
        updates.append((params['params_m_w'][i], params['params_m_w'][i] +
            params['params_v_w'][i] * T.grad(logZ, params['params_m_w'][i])))
        updates.append((params['params_v_w'][i], params['params_v_w'][i] -
                        params['params_v_w'][i]**2 * (T.grad(logZ, params['params_m_w'][i])**2
                                                        - 2 * T.grad(logZ, params['params_v_w'][i]))))

    updates.append((params['a'], 1.0 / (T.exp(logZ2 - 2 * logZ1 + logZ) *
        (params['a'] + 1) / params['a'] - 1.0)))
    updates.append((params['b'], 1.0 / (T.exp(logZ2 - logZ1) * (params['a'] + 1) /
        (params['b']) - T.exp(logZ1 - logZ) * params['a'] / params['b'])))

    return updates


def remove_invalid_updates(new_params, old_params, params):

    m_w_new = new_params['m_w']
    v_w_new = new_params['v_w']
    m_w_old = old_params['m_w']
    v_w_old = old_params['v_w']

    for i in range(len(params['layers'])):
        index1 = np.where(v_w_new[ i ] <= 1e-100)
        index2 = np.where(np.logical_or(np.isnan(m_w_new[i]),
            np.isnan(v_w_new[i])))

        index = [np.concatenate((index1[0], index2[0])),
            np.concatenate((index1[1], index2[1]))]

        if len(index[0]) > 0:
            m_w_new[i][index] = m_w_old[i][index]
            v_w_new[i][index] = v_w_old[i][index]
    params['m_w'] = m_w_new
    params['v_w'] = v_w_new
    return params
