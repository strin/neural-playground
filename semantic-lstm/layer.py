from __init__ import *
from keras.layers import LSTM, Layer
from keras import backend as K
from keras import activations, initializations
from keras.engine import InputSpec
from keras.layers.recurrent import time_distributed_dense


class SemConLSTM(LSTM):
    '''Semantically Conditioned LSTM unit.'''
    def __init__(self, output_dim, da_dim, **kwargs):
        super(SemConLSTM, self).__init__(output_dim, **kwargs)
        self.da_dim = da_dim
        self.consume_less = 'gpu'


    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2] - self.da_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.W = self.init((self.input_dim, 4 * self.output_dim),
                            name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, 4 * self.output_dim),
                                    name='{}_U'.format(self.name))

        self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                        K.get_value(self.forget_bias_init((self.output_dim,))),
                                        np.zeros(self.output_dim),
                                        np.zeros(self.output_dim))),
                            name='{}_b'.format(self.name))
        self.trainable_weights = [self.W, self.U, self.b]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights



    def step(self, x, states):
        wx = x[:, :self.da_dim]
        da = x[:, self.da_dim:]
        h, states = LSTM.step(self, wx, states)
        states[1] = states[1] + self.activation(da) # c
        return h, states[:2]

