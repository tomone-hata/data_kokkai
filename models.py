import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils

class Simplenn(object):
    def __init__(self, x_dim, n_classes, hidden_size=[32,32], batch_size=32, \
                 learning_rate=0.01, epochs=10, random_seed=None, class_weight=None, dropuout=0):
        model = keras.models.Sequential()
        self.x_dim = x_dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.class_weight = class_weight
        self.dropuout = dropuout
        self.model = Sequential()
        print('Start to building a neural network.')
        self.build()

    def build(self):
        self.model.add(keras.layers.Dense(units=self.hidden_size[0], \
                                     input_dim=self.x_dim,
                                     kernel_initializer='glorot_uniform', \
                                     bias_initializer='zeros', \
                                     activation='relu' \
                                            ))
        self.model.add(Dropout(self.dropuout))
        self.model.add(keras.layers.Dense(units=self.hidden_size[1], \
                                     input_dim=self.hidden_size[0],
                                     kernel_initializer='glorot_uniform', \
                                     bias_initializer='zeros', \
                                     activation='relu' \
                                            ))
        self.model.add(Dropout(self.dropuout))
        self.model.add(keras.layers.Dense(units=self.n_classes, \
                                     input_dim=self.hidden_size[1],
                                     kernel_initializer='glorot_uniform', \
                                     bias_initializer='zeros', \
                                     activation='softmax' \
                                            ))
        #sgd_optimizer = keras.optimizers.SGD(lr=self.learning_rate)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=self.learning_rate),
                      metrics=['accuracy'])
        print('Scucceeded in building a neural network.')


    def train(self, x_train, y_train, x_valid, y_valid, one_hot=False):
        print('Start to train.')
        if one_hot:
            y_train = np_utils.to_categorical(y_train)
            y_valid = np_utils.to_categorical(y_valid)

        self.model.fit(x_train, y_train, \
                       batch_size=self.batch_size, \
                       epochs=self.epochs, \
                       verbose=1, \
                       validation_data=(x_valid, y_valid),\
                       class_weight = self.class_weight)
        print('Scucceeded in training.')


    def test(self, x_test, y_test, one_hot=False):
        if one_hot:
            y_test = np_utils.to_categorical(y_test)

        score = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
