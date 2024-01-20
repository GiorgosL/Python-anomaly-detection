import pandas as pd
import logging
import configparser
import matplotlib.pyplot as plt
from logging.handlers import RotatingFileHandler
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from sklearn import preprocessing
from keras.optimizers import Adam

def load_config(file_path):
    """
    Load the configuration file.
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def setup_logging(log_file='script.log', log_level=logging.INFO):
    """
    Set up logging configuration.

    Args:
        log_file (str): Name of the log file.
        log_level (int): Logging level (default is INFO).
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )

def log_message(message, level=logging.INFO):
    """
    Log a message with the specified logging level.

    Args:
        message (str): The message to log.
        level (int): Logging level (default is INFO).
    """
    logging.log(level, message)



class Preprocess:
    """
    Preprocess pipeline 
    """
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)
        self.df = self.df.sort_values('Class')
        
    def scale_df(self):
        sc = StandardScaler()
        self.df['Time'] = sc.fit_transform(self.df['Time'].values.reshape(-1, 1))
        self.df['Amount'] = sc.fit_transform(self.df['Amount'].values.reshape(-1, 1))
    
    def min_max_df(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        self.x_scaled = min_max_scaler.fit_transform(self.df)
        self.df = pd.DataFrame(self.x_scaled,columns=self.df.columns.tolist())
    
    def split_data(self, fraction, flag):
        if flag == 'train':
            self.df = self.df[:round(len(self.df)*fraction/100)]
        if flag == 'test':
            self.df = self.df[round(len(self.df)*fraction/100):]
        x = self.df.loc[:, self.df.columns != 'Class']
        return x


class autoencoder:
    """
    Autoencoder pipeline 
    """
    def __init__(self, x_train, x_test):
        self.x_train = x_train
        self.x_test = x_test
    
    def create_model(self):
        input_img = keras.Input(shape=(self.x_train.shape[1],))
        encoded = layers.Dense(15, activation='relu')(input_img)
        encoded = layers.Dropout(0.4)(encoded)
        encoded = layers.Dense(8, activation='relu')(encoded)
        encoded = layers.Dropout(0.4)(encoded)
        encoded = layers.Dense(4, activation='relu')(encoded)
        decoded = layers.Dropout(0.4)(encoded)
        decoded = layers.Dense(8, activation='relu')(encoded)
        decoded = layers.Dropout(0.4)(decoded)
        decoded = layers.Dense(15, activation='relu')(decoded)
        decoded = layers.Dense(self.x_train.shape[1], activation='sigmoid')(decoded)
        self.autoencoder = keras.Model(input_img, decoded)
    
    def compile_model(self):
        self.autoencoder.compile(loss='binary_crossentropy',optimizer='adam')
        self.autoencoder.summary()
        
    
    def fit_model(self):
        early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.0001,
                    patience=10,
                    verbose=1, 
                    mode='min',
                    restore_best_weights=True)
        self.history = self.autoencoder.fit(self.x_train, self.x_train,
                epochs=50,
                batch_size=64,
                shuffle=True,
                validation_data=(self.x_test, self.x_test),
                callbacks = [early_stop],
                verbose=1)
        
    def plot_model(self):
        plt.plot(self.history.history['loss'], linewidth=2, label='Train')
        plt.plot(self.history.history['val_loss'], linewidth=2, label='Test')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        
    def save_model(self,path):
        self.autoencoder.save(path)