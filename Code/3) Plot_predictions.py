import matplotlib.pyplot as plt
import numpy as np
import sklearn
import random
from sklearn.preprocessing import MinMaxScaler
#import seaborn as sns
import pickle
import pandas as pd
import tabulate

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def metrics_total_results(model):

    if model == 'TFT':

        preds = np.load("predictions/TFT/tft_pv_predictions.npy")

        RMSE_values = []
        MAE_values = []
        MAPE_values = []
        R2_values = []
            
        y_test = pv_test['ActivePower'][288:-24]

        preds_unscaled = ((scaler.inverse_transform(preds.reshape(-1, 1)))).reshape(-1)
        actuals_unscaled = ((scaler.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)

        rmse_test = np.sqrt(mean_squared_error(actuals_unscaled[1:], preds_unscaled[:-1]))
        mae_test = mean_absolute_error(actuals_unscaled[1:], preds_unscaled[:-1])
        mape_test = mean_absolute_percentage_error(actuals_unscaled[1:], preds_unscaled[:-1])
        r2_score_test = r2_score(actuals_unscaled[1:], preds_unscaled[:-1])*100

        RMSE_values.append(rmse_test)
        MAE_values.append(mae_test)
        MAPE_values.append(mape_test)
        R2_values.append(r2_score_test)

        
        metrics_df = pd.DataFrame({
            'Dataset' : "SARAFALI",
            'Model' : "TFT",
            'RMSE': [RMSE_values[0]],
            'MAE': [MAE_values[0]],
            'MAPE': [MAPE_values[0]],
            'R2': [R2_values [0]]
            })

        metrics_df = metrics_df.round(3)

        fig,ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')

        # Adjust vertical space by setting row heights
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Scale width by 1, height by 2 (change 2 to a larger value for more height)

        for key, cell in table.get_celld().items():
            cell.set_fontsize(12)  # Adjust font size if needed
            cell.set_linewidth(0.5)  # Adjust line width if needed

        plt.tight_layout()
        plt.suptitle("TFT model total results", y=0.8)
        plt.savefig('results/TFT/tft_total_results.png', bbox_inches='tight', dpi=300)
        plt.show()


def plot_results(model, period):

    if model == 'TFT':

        y_test = pv_test['ActivePower'][288:-24]

        preds_scaled = np.load('predictions/TFT/tft_pv_predictions.npy')

        preds = ((scaler.inverse_transform(preds_scaled.reshape(-1, 1)))).reshape(-1)
        preds = preds.ravel()

        actuals = ((scaler.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)

        x = np.arange(0,period)
        zeros = np.zeros(period)

        errors = preds[288:288+period] - actuals[288+1:288+period+1].squeeze()
        errors = errors.reshape(-1)


        plt.figure(figsize=(20, 10))
        plt.plot(x, actuals[288+1:288+period+1], color='#fe9929', alpha=1, linestyle='--',  linewidth = 1.7, label='Actual Values')
        plt.plot(x, preds[288:288+period], color='#8856a7', alpha=0.8, linewidth = 1.5, label='TFT Predictions')
        plt.fill_between(x, zeros, errors, color='#31a354', alpha=0.4, label='TFT Error')

        plt.xlabel('Time-step')
        plt.ylabel('Power Production')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/TFT_plot.png', dpi=300)
        plt.show()
    


#Load the scalers

with open('scalers/TFT_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


#Load the test data

pv_test = pd.read_pickle('PV Data SARAFALI/PV_SARAFALI_test.pkl')

#Check results for TFT

#metrics_total_results('TFT')
plot_results('TFT', 288)
