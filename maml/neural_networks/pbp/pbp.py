
import sys
from copy import copy
import numpy as np
import theano
import theano.tensor as T
import network
import prior
from copy import deepcopy

class PBP:

    def __init__(self, layer_sizes, mean_y_train, std_y_train):

        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        # We initialize the prior

        self.prior = prior.Prior(layer_sizes, var_targets)
	self.initial_prior = deepcopy(self.prior)
        # We create the network

        params = self.prior.get_initial_params()
        self.initial_params = deepcopy(params)
	self.network = network.Network(params[ 'm_w' ], params[ 'v_w' ],
            params[ 'a' ], params[ 'b' ])

        # We create the input and output variables in theano

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

    def reset_pbp(self):
	    """
            Set the prior and parameters are reset to the intitialized values 
	    """
	    self.prior = deepcopy(self.initial_prior)
	    self.network.set_params(self.initial_params)


    def do_pbp(self, X_train, y_train, n_iterations, val_slice=0.05, tolerance=0.01, convergence=1e-06, stoch_select=False, 
		    normalize_targets=False, min_epochs=10, print_level=0):

        # We create the validation slice
        if normalize_targets:
	    y_train = (y_train - np.mean(y_train)) / np.std(y_train)
        if stoch_select:
            X_full = copy(X_train)
            y_full = copy(y_train)
        X_size = X_train.shape[0]
        stop_idx = int(X_size * (1. - val_slice))
        X_val = X_train[stop_idx:]
        y_val = y_train[stop_idx:]

        # We then remove it from the training set
        X_train = X_train[:stop_idx]
        y_train = y_train[:stop_idx]

        # We first do a single pass

        self.do_first_pass(X_train, y_train)

        # We refine the prior

        params = self.network.get_params()
        params = self.prior.refine_prior(params)
        self.network.set_params(params)
        # We get the initial error

        m, v, v_noise = self.get_predictive_mean_and_variance(X_val)
        mt, vt, vt_noise = self.get_predictive_mean_and_variance(X_train)
        v_err = np.sqrt(np.mean((y_val - m)**2))
        v_err_old = v_err
        d_err_old = 0
        vt_err_old = np.sqrt(np.mean((y_train - mt)**2))
        dt_err_old = 0
        for i in range(int(n_iterations) - 1):
            if stoch_select:
                permutation = np.random.choice(range(X_size), X_size, replace=False)
                index_train = permutation[0: stop_idx]
                index_test = permutation[stop_idx:]
                X_train = X_full[index_train, :]
                X_val = X_full[index_test, :]
                y_train = y_full[index_train]
                y_val = y_full[index_test]

            # We do one more pass

            params = self.prior.get_params()
            self.do_first_pass(X_train, y_train)

            # We refine the prior

            params = self.network.get_params()
            params = self.prior.refine_prior(params)
            self.network.set_params(params)

            # We test against validation set

            m, v, v_noise = self.get_predictive_mean_and_variance(X_val)
            mt, vt, vt_noise = self.get_predictive_mean_and_variance(X_train)
            v_err = np.sqrt(np.mean((y_val - m)**2))
            d_err = v_err - v_err_old
            d_d_err = d_err_old - d_err
            vt_err = np.sqrt(np.mean((y_train - mt)**2))
            dt_err = vt_err - vt_err_old
            d_dt_err = abs(dt_err_old - dt_err)
	    if print_level > 0:
	            print i + 1, vt_err, dt_err, d_dt_err, v_err, d_err, d_d_err
        	    sys.stdout.flush()
            if d_err > 0 + tolerance:
		if i + 1 > min_epochs:
		    print "Early Stopping Activated", d_err
                    break
            if d_dt_err < convergence:
		if i + 1 > min_epochs:    
                    print "Convergence Attained", d_dt_err
                    break
            v_err_old = v_err
            d_err_old = d_err
            vt_err_old = vt_err
            dt_err_old = dt_err

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

    def recalculate_mean_and_std(self, Y_train):
	    self.std_y_train = np.std(Y_train)
	    self.mean_y_train = np.mean(Y_train)


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
                #print counter
                sys.stdout.flush()

            counter += 1
