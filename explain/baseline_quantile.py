import numpy as np

LINK_FUNCTIONS = {'softplus': lambda x: np.log(1 + np.exp(x)),
                  'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
                  'tanh': lambda x: np.tanh(x),
                  'linear': lambda x: x,
                  'relu': lambda x: np.multiply(x, (x > 0))}


class NeutralFairBaseline:
    def __init__(self, delta=None, tolerance=1e-4, neutral_level=0.5, verbose=False):
        """
        :param delta: step size
        :param tolerance: tolerance level in the optimization
        :param neutral_level: neutrality value
        :param verbose: default is without verbose mode
        """
        self.delta = delta
        self.tolerance = tolerance
        self.reference_y = neutral_level
        self.verbose = verbose

    def search_baseline_slp(self, a, w, b, link, x_train):
        """
        :param a: neutrality value
        :param w: weights of the slp
        :param b: biases of the slp
        :param link: link function of the slp
        :param x_train: training data
        :return: baseline and distance error from neutrality value
        """
        baseline = self.binary_search(w, x_train, lambda c: np.asscalar(LINK_FUNCTIONS[link](c @ w + b) - a))
        error = np.abs(np.asscalar(LINK_FUNCTIONS[link](baseline @ w + b) - a))
        return baseline, error

    def search_baseline_mlp(self, w, b, links, x_train):
        """
        :param w: weights of the MLP (as a list)
        :param b: biases of the MLP (as a list)
        :param links: link functions (as a list)
        :param x_train: training data
        :return: Returns baselines and distance errors as arrays
        """
        if self.delta is None:
            self.delta = 1 / (2 * x_train.shape[0])
        # Feedforward the x_train for statistics
        all_x_train = [x_train]
        for l in range(np.size(links)): all_x_train.append(self.next_layer(w[l], b[l], links[l], all_x_train[l]))
        baselines = [np.array([self.reference_y])]
        errors = []
        a_size = 1
        for l in range(np.size(links) - 1, -1, -1):
            if self.verbose: print('In Layer l: ', l)
            for _ in range(a_size):
                if self.verbose: print(' - neutral n: ', _)
                this_ref = baselines.pop(0)
                for j, a in enumerate(this_ref):
                    if self.verbose: print(' -- in SLP: ', j)
                    a_tilde, error = self.search_baseline_slp(a, np.array(w[l])[:, j], np.array(b[l])[j], links[l],
                                                              all_x_train[l])
                    errors.append(error)
                    baselines.append(a_tilde)
            a_size = a_size * w[l].shape[1]
        baselines = np.array(baselines).reshape(-1)
        return baselines, errors

    def binary_search(self, w, x, distance):
        """
        :param w: weights
        :param x: training data
        :param distance: distance metric
        :return: baseline
        """
        grid = np.linspace(0, 1, int(1 // self.delta))
        low = 0
        high = len(grid) - 1
        while low <= high:
            mid = (high + low) // 2
            c = np.zeros(np.size(x, 1))
            for i in range(np.size(w, 0)):
                if w[i] < 0:
                    c[i] = np.quantile(x[:, i], 1 - grid[mid])
                else:
                    c[i] = np.quantile(x[:, i], grid[mid])
            # Check if x is present at mid
            value_now = distance(c)
            if np.abs(value_now) < self.tolerance:
                return c
            else:
                if value_now < 0:
                    low = mid + 1
                else:
                    high = mid - 1
        # If we reach here, return closest value
        if self.verbose: print('  --- Tolerance level not matched; returning closest value.')
        return c

    # Function to move to the next layer
    @staticmethod
    def next_layer(w, b, link, x_train, x_test=None):
        """
        :param w: weights of the MLP (as a list)
        :param b: biases of the MLP (as a list)
        :param link: link functions (as a list)
        :param x_train: training data
        :param x_test: test data (optional, default is None)
        :return: next layer neurons
        """
        if x_test:
            x_test_next = np.zeros((x_test.shape[0], w.shape[1]))
        x_train_next = np.zeros((x_train.shape[0], w.shape[1]))
        for j in range(np.size(w, 1)):
            if x_test:
                x_test_next[:, j] = LINK_FUNCTIONS[link](x_test @ w[:, j] + b[j])
            x_train_next[:, j] = LINK_FUNCTIONS[link](x_train @ w[:, j] + b[j])
        if x_test:
            return x_train_next, x_test_next
        else:
            return x_train_next


if __name__ == "__main__":
    # Test
    np.random.seed(2)
    x_1 = np.random.normal(0, 1, 10000)
    x_2 = np.random.normal(0, 1, 10000)
    nf_bas = NeutralFairBaseline()
    b_, _ = nf_bas.search_baseline_mlp([np.array((1, -1)).reshape(-1, 1)], [np.array(1).reshape(-1)],
                                       ['sigmoid'], np.vstack((x_1, x_2)).T)
    print('This number should be close to 0: ', np.round(np.abs(b_[0] + b_[1]), 2))
