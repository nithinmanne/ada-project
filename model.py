"""This is the main convolutional neural network that's used by the agent
   both in RLlib and the non-distributed variant."""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Permute, Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.activations import relu, linear
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


def create_dqn_model(num_actions):
    """Just creates the model and returns it, and is used by the agent to
       create the prediction and target networks."""
    return Sequential([
        Permute((2, 3, 1)),
        Conv2D(32, 8, strides=4, activation=relu),
        Conv2D(64, 4, strides=2, activation=relu),
        Conv2D(64, 3, strides=1, activation=relu),

        Flatten(),
        Dense(512, activation=relu),
        Dense(num_actions, activation=linear),
    ])


class DQNRLlibModel(TFModelV2):
    """This is the same model as above for RLlib. It requires the class to implement
       some API which is needed for RLlib to generate the graph."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        inputs = Input(obs_space.shape)
        cnn1 = Conv2D(32, 8, strides=4, activation=relu)(inputs)
        cnn2 = Conv2D(64, 4, strides=2, activation=relu)(cnn1)
        cnn3 = Conv2D(64, 3, strides=1, activation=relu)(cnn2)
        flatten = Flatten()(cnn3)
        fnn1 = Dense(512, activation=relu)(flatten)
        fnn2 = Dense(num_outputs, activation=linear)(fnn1)
        value_out = Dense(1, activation=None)(fnn2)
        self.base_model = Model(inputs, [fnn2, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict['obs'])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def import_from_h5(self, h5_file: str) -> None:
        self.base_model.load_weights(h5_file)
