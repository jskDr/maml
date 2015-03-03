
import numpy as np

import pbp

class PBP_net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False, val_slice=0.05, tolerance=0.01, convergence=1e-06):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ])
            self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train

        # We construct the network

        n_units_per_layer = \
            np.concatenate(([ X_train.shape[ 1 ] ], n_hidden, [ 1 ]))
        self.pbp_instance = \
            pbp.PBP(n_units_per_layer, self.mean_y_train, self.std_y_train)

        # We iterate the learning process

        self.pbp_instance.do_pbp(X_train, y_train_normalized, n_epochs,val_slice=val_slice,
                                 tolerance=tolerance, convergence=convergence)

        # We are done!

    def predict(self, X_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            
    
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin = 2)

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        m, v, v_noise = self.pbp_instance.get_predictive_mean_and_variance(X_test)

        # We are done!

        return m, v, v_noise
