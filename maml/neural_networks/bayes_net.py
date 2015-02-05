__author__ = 'epyzerknapp'
import numpy as np
import maml.neural_networks.sep


class BayesNet:

    def __init__(self, x_train, y_train, n_hidden, n_epochs=40,
                 normalize=False):
        """
        Constructor for the class implementing a Bayesian neural network.

        This is currently written in Theano, but will be closer integrated with
        the codebase in the future

        :param x_train      Matrix with the features for the training data.
        :param y_train      Vector with the target variables for the
                            training data.
        :param n_hidden     Vector with the number of neurons for each
                            hidden layer.
        :param n_epochs     Number of epochs for which to train the
                            network. The recommended value 40 should be
                            enough.
        :param normalize    Whether to normalize the input features. This
                            is recommended unless the input vector is for
                            example formed by binary features (a
                            fingerprint). In that case we do not recommend
                            to normalize the features.

        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(x_train, 0)
            self.std_X_train[self.std_X_train == 0] = 1
            self.mean_X_train = np.mean(x_train, 0)
        else:
            self.std_X_train = np.ones(x_train.shape[1])
            self.mean_X_train = np.zeros(x_train.shape[1])

        x_train = (x_train - np.full(x_train.shape, self.mean_X_train)) / \
            np.full(x_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train

        # We construct the network

        n_units_per_layer = \
            np.concatenate(([ x_train.shape[ 1 ] ], n_hidden, [ 1 ]))
        self.sep_instance = \
            sep.SEP(n_units_per_layer, self.mean_y_train, self.std_y_train)

        # We iterate the learning process

        self.sep_instance.do_sep(x_train, y_train_normalized,
            x_train, y_train, n_epochs)

        # We are done!

    def predict(self, x_test):
        """
            Function for making predictions with the Bayesian neural network.

            :param x_test   The matrix of features for the test data


            :return m       The predictive mean for the test target variables.
            :return v       The predictive variance for the test target
                            variables.
            :return v_noise The estimated variance for the additive noise.

        """

        x_test = np.array(x_test, ndmin=2)

        # We normalize the test set

        x_test = (x_test - np.full(x_test.shape, self.mean_X_train)) / \
            np.full(x_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        m, v, v_noise = self.sep_instance.get_predictive_mean_and_variance(x_test)

        # We are done!

        return m, v, v_noise