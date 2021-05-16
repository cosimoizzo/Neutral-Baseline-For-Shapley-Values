import numpy as np
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tqdm.notebook import tqdm


def validate_mlp(x_train, y_train, runs, hidden_layers, nodes_hidden_layers, activation_layers=None, lr=0.05):
    """
    :param x_train: training data
    :param y_train: training target
    :param runs: number of sampling
    :param hidden_layers: range of hidden layers
    :param nodes_hidden_layers: range of nodes in hidden layers
    :param activation_layers: what type of activation layers
    :param lr: learning rate
    :return: best model, best structure, best performance
    """
    if activation_layers is None:
        activation_layers = ['sigmoid']

    cache = set()
    max_model = None
    max_conf = None
    max_performance = 0.0
    for _ in tqdm(range(runs)):

        conf = []
        for _ in range(np.random.choice(hidden_layers)):
            conf.append((np.random.choice(nodes_hidden_layers), np.random.choice(activation_layers)))

        conf.append((1, 'sigmoid'))

        conf = tuple(conf)
        if conf in cache:
            continue

        cache.add(conf)
        model_val, history_val = fit_mlp(x_train, y_train, conf, lr=lr)

        performance = max(history_val.history['val_accuracy'])

        if performance > max_performance:
            max_performance = performance
            max_conf = conf
            max_model = model_val

    return max_model, max_conf, max_performance


def fit_mlp(x_train, y_train, conf, seed=3, epochs=1500, batch_size=300, lr=0.05):
    """
    :param x_train: training data
    :param y_train: training target
    :param conf: model structure
    :param seed: seed
    :param epochs: number of epochs (using early stopping)
    :param batch_size: batch size
    :param lr: learning rate
    :return: model and history
    """
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               verbose=0,
                                               patience=3,
                                               restore_best_weights=True)
    model = Sequential()
    for n, a in conf:
        model.add(
            Dense(n,
                  activation=a,
                  kernel_initializer=keras.initializers.glorot_normal(seed=seed),
                  bias_initializer='zeros'))

    opt = keras.optimizers.Adam(lr=lr)
    if conf[-1][1] == 'linear':
        model.compile(loss=keras.losses.mse,
                      optimizer=opt)
    elif conf[-1][1] == 'sigmoid':
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])
    else:
        model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        verbose=0,
                        shuffle=False, callbacks=[early_stop])

    return model, history
