import numpy as np
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


def get_sparse_mlp(ws, bs, ls, reference):
    """
    :param ws: Weights of the MLP (as a list)
    :param bs: Biases of the MLP (as a list)
    :param ls: link functions
    :param reference: baselines (aka reference points)
    :return: (keras) sparse version of the model
    """
    # New MLP number of neurons
    times = np.ones(len(ls)).astype(np.int)
    for j in range(len(ls) - 2, -1, -1):
        times[j] = np.size(ws[j], 1) * times[j + 1]

    # build sparse model
    sparse_model = Sequential()
    for j in range(len(ls)):
        # Need to store dimensions in some scalar
        n_neurons_realnet = np.size(ws[j], 0)
        # Initialize Weight vector to 0s
        this_w = np.zeros((n_neurons_realnet * times[j], times[j]))
        # and biases
        this_b = np.zeros(times[j])
        # Fill the biases and the weights in the correct place
        col = 0
        for i in range(np.size(this_w, 1)):
            if col == np.size(ws[j], 1): col = 0
            this_w[i * n_neurons_realnet:(i + 1) * n_neurons_realnet, i] = ws[j][:, col]
            this_b[i] = np.asarray(bs[j])[col]
            col += 1
        # Add layer to the network
        this_dense = Dense(units=np.size(this_w, 1), activation=ls[j], input_shape=(np.size(this_w, 0),))
        sparse_model.add(this_dense)
        this_dense.set_weights([this_w, this_b])
    # compile the new (sparse) model
    opt = keras.optimizers.Adam()
    sparse_model.compile(loss=keras.losses.binary_crossentropy,
                         optimizer=opt,
                         metrics=['accuracy'])

    # Print reference
    at_reference = sparse_model.predict(np.array(reference).reshape(1, -1))
    print("Prediction at the reference point is: ", at_reference)
    return sparse_model
