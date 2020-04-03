import datetime as dt
import enum
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from data_processor import DataLoader
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from numpy import newaxis
from utils import Timer


class PredictionType(enum.Enum):
    multiSequence = "Multi-Sequence"
    fullSequence = "Full-Sequence"
    pointByPoint = "Point-By-Point"


class DataLoaderType(enum.Enum):
    stocks = "Stocks"

    def __str__(self):
        return self.value


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


class Model:
    """A class for an building and inferencing an lstm model"""
    '''
    data_configs: {
            "sequence_length": 50, "normalise": True,
            "split": 0.85, "cols": ("1. open", "2. high", "3. low", "4. close", "5. volume"),
            "type": "stocks", #DataLoaderType.stocks
            "identifier": {"name": "MSFT"}}
    '''

    def __init__(self, configs=None,
                 data_configs=None):

        self.model = Sequential()
        self.configs = json.load(
            open('../static/model_configs/basic_stock_market_lstm.json', 'r')) if configs is None else configs
        self.data = DataLoader(data_configs)

        self.build_model(self.configs)
        print("Model Configs: " + str(self.configs))
        print("Data Configs: " + str(self.data.data_configs))
        if not os.path.exists(self.configs['model']['save_dir']): os.makedirs(self.configs['model']['save_dir'])
        self.predictions = None
        self.actual = None
        self.prediction_type = self.data.data_configs["prediction_type"]

    def predict(self, prediction_type=PredictionType.multiSequence, plot_graphs=False):

        x, y = self.data.get_train_data(
            seq_len=self.data.data_configs['sequence_length'],
            normalise=self.data.data_configs['normalise']
        )

        '''
    # in-memory training
    model.train(
        x,
        y,
        epochs = self.configs['training']['epochs'],
        batch_size = self.configs['training']['batch_size'],
        save_dir = self.configs['model']['save_dir']
    )
    '''
        # out-of memory generative training
        steps_per_epoch = math.ceil(
            (self.data.len_train - self.data.data_configs['sequence_length']) / self.configs['training'][
                'batch_size'])
        self.train_generator(
            data_gen=self.data.generate_train_batch(
                seq_len=self.data.data_configs['sequence_length'],
                batch_size=self.configs['training']['batch_size'],
                normalise=self.data.data_configs['normalise']
            ),
            epochs=self.configs['training']['epochs'],
            batch_size=self.configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=self.configs['model']['save_dir']
        )

        x_test, y_test = self.data.get_test_data(
            seq_len=self.data.data_configs['sequence_length'],
            normalise=self.data.data_configs['normalise']
        )
        self.actual = y_test
        if prediction_type == PredictionType.multiSequence:
            self.predictions = self.predict_sequences_multiple(data=x_test,
                                                               prediction_len=self.data.data_configs['sequence_length'],
                                                               window_size=self.data.data_configs['sequence_length'])

        elif prediction_type == PredictionType.fullSequence:
            self.predictions = self.predict_sequence_full(x_test, self.data.data_configs['sequence_length'])

        else:
            self.predictions = self.predict_point_by_point(x_test)

        if plot_graphs:
            self.plot_graphs(self.predictions, self.actual, self.prediction_type)

        # TODO:
        #  return a __str__ method override or toJson for models class to send class data and graph data to the response
        return {"model_configs": self.configs,
                "data_configs": self.data.data_configs, "predictions": self.predictions, "actual": self.actual}

    def plot_graphs(self, predictions, actual, prediction_type):
        if prediction_type == PredictionType.multiSequence:
            plot_results_multiple(predictions, actual, data.data_configs['sequence_length'])
        elif prediction_type == PredictionType.fullSequence:
            plot_results_multiple(predictions, actual, self.data.data_configs['sequence_length'])
        else:
            plot_results(self.predictions, self.actual)

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator(self, data_gen, steps_per_epoch, save_dir, epochs=2, batch_size=32):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size=50, prediction_len=50):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size=50):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted
