from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Permute
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.activations import relu, linear


class DQNModel(Model):
    def __init__(self, num_actions):
        super().__init__()
        self.permute = Permute((2, 3, 1))
        self.cnn1 = Conv2D(32, 8, strides=4, activation=relu)
        self.cnn2 = Conv2D(64, 4, strides=2, activation=relu)
        self.cnn3 = Conv2D(64, 3, strides=1, activation=relu)

        self.flatten = Flatten()
        self.nn1 = Dense(512, activation=relu)
        self.nn2 = Dense(num_actions, activation=linear)

    def call(self, inputs):
        out = self.permute(inputs)
        out = self.cnn1(out)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = self.cnn3(out)

        out = self.flatten(out)
        out = self.nn1(out)
        out = self.nn2(out)
        return out

    def get_config(self):
        return {'num_actions': self.num_actions}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
