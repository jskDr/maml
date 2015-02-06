__author__ = 'epyzerknapp', 'miguel'
import sys

import math

import numpy as np

import theano

import theano.tensor as T

import network

import prior

class SEP:

    def __init__(self, layer_sizes, mean_y_train, std_y_train):

        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        # We initialize the prior

        self.prior = prior.Prior(layer_sizes, var_targets)

        # We create the network

        params = self.prior.get_initial_params()
        self.network = network.Network(params[ 'm_w' ], params[ 'v_w' ],
            params[ 'a' ], params[ 'b' ])

        # We create the input and output variables

        self.x = T.vector('x')
        self.y = T.scalar('y')

        # A function for computing the value of logZ, logZ1 and logZ2

        self.logZ, self.logZ1, self.logZ2 = \
            self.network.logZ_Z1_Z2(self.x, self.y)

        # We create a theano function for updating the posterior

        self.adf_update = theano.function([ self.x, self.y ], self.logZ,
            updates = self.network.generate_updates(self.logZ, self.logZ1,
            self.logZ2))

        # We greate a theano function for the network predictive distribution

        self.predict = theano.function([ self.x ],
            self.network.predict(self.x))

    def do_sep_update(self, new_params, old_params, params_prior, eps, n):

        ret_params = { 'm_w': [], 'v_w': [], 'a': None, 'b': None }

        for i in range(len(self.network.layers)):

            v_w_nat_prior = 1.0 / params_prior[ 'v_w' ][ i ]
            m_w_nat_prior = params_prior[ 'm_w' ][ i ] / \
                params_prior[ 'v_w' ][ i ]
            v_w_nat_new = 1.0 / new_params[ 'v_w' ][ i ]
            v_w_nat_old = 1.0 / old_params[ 'v_w' ][ i ]
            m_w_nat_new = new_params[ 'm_w' ][ i ] / new_params[ 'v_w' ][ i ]
            m_w_nat_old = old_params[ 'm_w' ][ i ] / old_params[ 'v_w' ][ i ]
            v_hat_nat = v_w_nat_new - v_w_nat_old
            m_hat_nat = m_w_nat_new - m_w_nat_old

            v_w_nat_update = (1 - eps) * v_w_nat_old + eps * \
                (n * v_hat_nat + v_w_nat_prior)
            m_w_nat_update = (1 - eps) * m_w_nat_old + eps * \
                (n * m_hat_nat + m_w_nat_prior)

            ret_params[ 'm_w' ].append(m_w_nat_update / v_w_nat_update)
            ret_params[ 'v_w' ].append(1.0 / v_w_nat_update)

        a_nat_prior = params_prior[ 'a' ] - 1
        b_nat_prior = -params_prior[ 'b' ]
        a_nat_new = new_params[ 'a' ] - 1
        b_nat_new = -new_params[ 'b' ]
        a_nat_old = old_params[ 'a' ] - 1
        b_nat_old = -old_params[ 'b' ]
        a_hat_nat = a_nat_new - a_nat_old
        b_hat_nat = b_nat_new - b_nat_old

        a_nat_update = (1 - eps) * a_nat_old + eps * (n * a_hat_nat + \
            a_nat_prior)
        b_nat_update = (1 - eps) * b_nat_old + eps * (n * b_hat_nat + \
            b_nat_prior)

        a_nat_update = a_nat_prior + (a_nat_old - a_nat_prior) + a_hat_nat
        b_nat_update = b_nat_prior + (b_nat_old - b_nat_prior) + b_hat_nat

        ret_params[ 'a' ] = a_nat_update + 1
        ret_params[ 'b' ] = -b_nat_update

        return ret_params

    def do_sep(self, X_train, y_train, X_test, y_test, n_iterations):

        # We first do a single pass

        self.do_first_pass(X_train, y_train)

        # We refine the prior

        params = self.network.get_params()
        params = self.prior.refine_prior(params)
        self.network.set_params(params)

        error, ll = self.get_test_error_and_ll(X_test, y_test)
        print 0, error, ll
        sys.stdout.flush()

        for i in range(int(n_iterations) - 1):

            # We do one more pass

            params = self.prior.get_params()
            self.do_first_pass(X_train, y_train)

            # We refine the prior

            params = self.network.get_params()
            params = self.prior.refine_prior(params)
            self.network.set_params(params)

            error, ll = self.get_test_error_and_ll(X_test, y_test)
            print i, error, ll
            sys.stdout.flush()

    def get_predictive_variance_no_noise(self, X_test):

        v = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            _, v[ i ] = self.predict(X_test[ i, : ])
            v[ i ] -= self.network.b.get_value() / \
                (self.network.a.get_value() - 1)

        return v

    def get_test_error_and_ll(self, X_test, y_test):

        error = 0
        ll = 0
        for i in range(X_test.shape[ 0 ]):
            m, v = self.predict(X_test[ i, : ])
            m = m * self.std_y_train + self.mean_y_train
            v = v * self.std_y_train**2
            ll += -0.5 * np.log(2 * math.pi * v) - \
                0.5 * (m - y_test[ i ])**2 / v
            error += (y_test[ i ] - self.mean_y_train - \
                self.std_y_train * self.predict(X_test[ i, : ])[ 0 ])**2
        error = np.sqrt(error / X_test.shape[ 0 ])
        ll = ll / X_test.shape[ 0 ]

        return error, ll

    def get_predictive_mean_and_variance(self, X_test):

        mean = np.zeros(X_test.shape[ 0 ])
        variance = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            m, v = self.predict(X_test[ i, : ])
            v -= self.network.b.get_value() / \
                (self.network.a.get_value() - 1)
            m = m * self.std_y_train + self.mean_y_train
            v = v * self.std_y_train**2
            mean[ i ] = m
            variance[ i ] = v

        v_noise = self.network.b.get_value() / \
            (self.network.a.get_value() - 1) * self.std_y_train**2

        return mean, variance, v_noise

    def do_first_pass(self, X, y):

        permutation = np.random.choice(range(X.shape[ 0 ]), X.shape[ 0 ],
            replace = False)

        counter = 1
        for i in permutation:

            old_params = self.network.get_params()
            logZ = self.adf_update(X[ i, : ], y[ i ])
            new_params = self.network.get_params()
            self.network.remove_invalid_updates(new_params, old_params)
            self.network.set_params(new_params)

            if counter % 1000 == 0:
                print counter
                sys.stdout.flush()

            counter += 1

    def do_additional_pass(self, X, y, params_prior):

        permutation = np.random.choice(range(X.shape[ 0 ]), X.shape[ 0 ],
            replace = False)

        counter = 1
        for i in permutation:

            old_params = self.network.get_params()
            logZ = self.adf_update(X[ i, : ], y[ i ])
            new_params = self.network.get_params()
            self.network.remove_invalid_updates(new_params, old_params)
            updated_params = self.do_sep_update(new_params, old_params,
                params_prior, 0.5 / X.shape[ 0 ], X.shape[ 0 ])
            self.network.set_params(updated_params)

            if counter % 1000 == 0:
                print counter
                sys.stdout.flush()

            counter += 1
