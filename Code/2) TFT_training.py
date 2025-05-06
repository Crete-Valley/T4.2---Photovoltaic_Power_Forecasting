import metpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import gc
import pickle
import warnings
import random
import tensorflow as tf
import lightning.pytorch as pl
import pytorch_forecasting as pf
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")
# Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


class MinValidationLossCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float("inf")  # Initialize with a very high value

    def on_validation_end(self, trainer, pl_module):
        # Access the latest validation loss
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Worker {worker_id} initialized with seed {seed}")


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred))))

def create_Time_Dataset(train, test):

    max_encoder_length = 288
    max_prediction_length = 24

    training_cutoff = len(train) - max_prediction_length


    # Define the source training dataset
    training = TimeSeriesDataSet(
        train[lambda x: x.time_index <= training_cutoff],
        time_idx="time_index",
        target="ActivePower",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=["Season"],
        time_varying_known_reals=['AmbientTemperature',
                                  'WindSpeed',
                                  'Irradiation',
                                  'Irradiance',
                                  'PVTemperature',
                                  'Month_Sin',
                                  'Month_Cos',
                                  'Hour_Sin',
                                  'Hour_Cos',
                                  'Minute_Sin',
                                  'Minute_Cos'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["ActivePower"],
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        allow_missing_timesteps=False,
        #categorical_encoders={"Month": NaNLabelEncoder(add_nan=True)},
        scalers = None
    )

    # Define the source validation and testing dataset
    validation = TimeSeriesDataSet.from_dataset(training, train, predict=True, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, test)


    return training, validation, testing

def TFT_model(dataset):

    model = TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size = 64,
        hidden_continuous_size = 8,
        attention_head_size = 4,
        dropout = 0.25,
        learning_rate = 0.001,
        lstm_layers = 2,
        loss = QuantileLoss(),
        optimizer = "Adam",
        log_interval = 10,
        log_val_interval = 10,
        reduce_on_plateau_patience = 4
    )

    return model



def fit_predict_TFT(train, test, scaler):

    training, validation, testing = create_Time_Dataset(train, test)

    batch_size = 128

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=3, pin_memory=True, prefetch_factor=5, worker_init_fn=worker_init_fn, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=3, pin_memory=True, prefetch_factor=5, worker_init_fn=worker_init_fn, persistent_workers=True)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=3, pin_memory=True, prefetch_factor=5, worker_init_fn=worker_init_fn, persistent_workers=True)


    y_test = test['ActivePower'][288:-24]

    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    logger = TensorBoardLogger("lightning_logs", log_graph=False)
    min_val_loss_callback = MinValidationLossCallback()

    model = TFT_model(training)

    trainer = pl.Trainer(
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs = 50,
        enable_model_summary=True,
        gradient_clip_val = 0.3,
        callbacks = [early_stop_callback, min_val_loss_callback],
        logger = logger,
        enable_checkpointing = False,
        log_every_n_steps=10,  # Log every step
        num_sanity_val_steps=0  # To avoid running validation before training starts
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.save_checkpoint("models/tft_pv.ckpt")

    model = TemporalFusionTransformer.load_from_checkpoint("models/tft_pv.ckpt")

    predictions_scaled_gpu_test = model.predict(test_dataloader)
    predictions_scaled_test = (predictions_scaled_gpu_test.cpu()).numpy()
    predictions_scaled_test = predictions_scaled_test[:,:1].ravel()
    predictions_scaled_test = predictions_scaled_test[1:]

    np.save('predictions/TFT/tft_pv_predictions.npy', predictions_scaled_test)

    predictions_test = ((scaler.inverse_transform(predictions_scaled_test.reshape(-1, 1)))).reshape(-1)
    true_test = ((scaler.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)

    rmse_test = np.sqrt(mean_squared_error(true_test, predictions_test))
    mae_test = mean_absolute_error(true_test, predictions_test)
    mape_test = mean_absolute_percentage_error(true_test, predictions_test)
    r2_score_test = r2_score(true_test, predictions_test)*100

    tft_results = ['Vestas_data_history', 'Test_forecast', mae_test, rmse_test, mape_test, r2_score_test]
    print(tft_results)

    del model, trainer, training, validation, testing, train_dataloader, val_dataloader, test_dataloader, predictions_scaled_gpu_test, predictions_scaled_test
    torch.cuda.empty_cache()
    gc.collect()
    

def replace_nans(df1):

    df = df1.copy()

    for i in range(len(df)):
        
        for col in df.columns:
            
            if col != "Season":

                if np.isnan(df.loc[i, col]):

                    print(i)
                    print(col)
                    
                    if i+288 < len(df):
                        df.loc[i, col] = df.loc[i+288, col]
                    else:
                        df.loc[i, col] = df.loc[i-288, col] 

    return df



np.random.seed(42)
torch.manual_seed(42)


with open('scalers/TFT_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


train = pd.read_pickle('PV Data SARAFALI/PV_SARAFALI_train.pkl')
test = pd.read_pickle('PV Data SARAFALI/PV_SARAFALI_test.pkl')

#print(train.isna().values.any())
#print(test.isna().values.any())

#train_2 = replace_nans(train)
#test_2 = replace_nans(test)

#train_2.to_pickle('PV Data SARAFALI/PV_SARAFALI_train.pkl')
#test_2.to_pickle('PV Data SARAFALI/PV_SARAFALI_test.pkl')
print(train)
print(test)

fit_predict_TFT(train, test, scaler)
