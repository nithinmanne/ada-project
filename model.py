from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Permute
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.activations import relu, linear


def create_dqn_model(num_actions):
    return Sequential([
        Permute((2, 3, 1)),
        Conv2D(32, 8, strides=4, activation=relu),
        Conv2D(64, 4, strides=2, activation=relu),
        Conv2D(64, 3, strides=1, activation=relu),

        Flatten(),
        Dense(512, activation=relu),
        Dense(num_actions, activation=linear),
    ])
