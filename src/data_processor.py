import enum

import numpy as np
from alpha_vantage.timeseries import TimeSeries


class PredictionType(enum.Enum):
    multiSequence = "Multi-Sequence"
    fullSequence = "Full-Sequence"
    pointByPoint = "Point-By-Point"


class DataLoaderType(enum.Enum):
    stocks = "Stocks"

    def __str__(self):
        return self.value


class DataLoader:
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, data_configs=None):

        self.data_configs = {
            "prediction_type": PredictionType.multiSequence,
            "sequence_length": 50, "normalise": True,
            "split": 0.85, "cols": ["1. open", "2. high", "3. low", "4. close", "5. volume"],
            "type": DataLoaderType.stocks,
            "identifier": {"name": "MSFT"}} if data_configs is None else data_configs

        if self.data_configs["type"] == DataLoaderType.stocks:
            ts = TimeSeries(key='T6P5PZDEXZTMHC5V', output_format='pandas')

            dataframe, meta_data = ts.get_daily(symbol=self.data_configs["identifier"]["name"],
                                                outputsize='full')  # full 20 years of daily data

            i_split = int(len(dataframe) * self.data_configs["split"])

            self.data_train = dataframe.get(self.data_configs["cols"]).values[:i_split]
            self.data_test = dataframe.get(self.data_configs["cols"]).values[i_split:]

        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len=50, normalise=True):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_train_data(self, seq_len=50, normalise=True):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len=50, batch_size=32, normalise=True):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i + seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
