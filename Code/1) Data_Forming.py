import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import math
import pickle
import os
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter("ignore", UserWarning)




def fill_missing_timesteps(df, start_time, end_time, mode):
    # Convert to datetime


    if mode == "inv":

        df["Start Time"] = df["Start Time"].astype(str).str.replace(" DST", "", regex=False)

        df["Start Time"] = pd.to_datetime(df["Start Time"])

        # Generate full range of timestamps
        full_range = pd.date_range(start=start_time, end=end_time, freq="5min")

        # Set index to timestamp column
        df = df.set_index("Start Time")

        # Reindex to include missing timestamps
        df = df.reindex(full_range)

        # Forward-fill all columns EXCEPT for ActivePower
        cols_except_active_power = [col for col in df.columns if col != "Active power(kW)"]
        df[cols_except_active_power] = df[cols_except_active_power].ffill()

        # Set ActivePower to 0 only for the new rows
        df["Active power(kW)"] = df["Active power(kW)"].fillna(0)

        # Reset index
        df = df.reset_index().rename(columns={"index": "Start Time"})

    elif mode == "emi":

        full_range = pd.date_range(start=start_time, end=end_time, freq="5min")

        df = df.set_index("Start Time")
        df = df.reindex(full_range)

        for col in df.columns:
            if col != "Start Time":
                df[col] = df[col].fillna(df[col].shift(288))
        
        df = df.reset_index().rename(columns={"index": "Start Time"})
        
    return df


def remove_extra_timesteps(df, start_date, stop_date):
    
    #print(len(df))

    # Filter for the given date range
    df = df[(df["Start Time"] >= pd.to_datetime(start_date)) & (df["Start Time"] <= pd.to_datetime(stop_date))]
    
    # Drop duplicates while keeping the first occurrence (assumes the first occurrence is in the correct order)
    df = df.sort_values(by=["Start Time"]).drop_duplicates(subset=["Start Time"], keep='first')

    df = df.reset_index(drop=True)

    #print(len(df))
    
    return df


def check_missing_extra_rows(df, datetime_col, start_date, stop_date):

    # Ensure the column is in datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Generate the expected date range
    expected_range = len(pd.date_range(start=start_date, end=stop_date, freq='5min'))
    
    df_range = len(df)

    if df_range - expected_range < 0:
        print("Missing timestamps")
        return -1
    elif df_range - expected_range > 0:
        print("Extra timestamps")
        return 1
    elif df_range - expected_range == 0:
        print("Fine")
        return 0


def data_collection():

    Sarafali_data = pd.DataFrame(columns=['Timestep', 'ActivePower', 'AmbientTemperature', 'Irradiation', 'Irradiance', 'PVTemperature', 'WindDirection', 'WindSpeed'])


    for year in ["2023","2024","2025"]:

        if year == "2023":

            months = ["08","09","10","11","12"]
            start_dates = ["2023-08-12 01:00:00", "2023-08-31 23:00:00", "2023-09-30 23:00:00", "2023-11-01 00:00:00", "2023-12-01 00:00:00"]
            stop_dates = ["2023-08-31 22:55:00", "2023-09-30 22:55:00", "2023-10-31 23:55:00", "2023-11-30 23:55:00", "2023-12-31 23:55:00"]
        
        elif year == "2024":
        
            months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
            start_dates = ["2024-01-01 00:00:00", "2024-02-01 00:00:00", "2024-03-01 00:00:00", "2024-03-31 23:00:00", "2024-04-30 23:00:00", "2024-05-31 23:00:00", "2024-06-30 23:00:00", "2024-07-31 23:00:00", "2024-08-31 23:00:00", "2024-09-30 23:00:00", "2024-11-01 00:00:00", "2024-12-01 00:00:00"]
            stop_dates = ["2024-01-31 23:55:00", "2024-02-29 23:55:00", "2024-03-31 22:55:00", "2024-04-30 22:55:00", "2024-05-31 22:55:00", "2024-06-30 22:55:00", "2024-07-31 22:55:00", "2024-08-31 22:55:00", "2024-09-30 22:55:00", "2024-10-31 23:55:00", "2024-11-30 23:55:00", "2024-12-31 23:55:00"]

        elif year == "2025":

            months = ["01","02"]
            start_dates = ["2025-01-01 00:00:00", "2025-01-18 00:00:00"]
            stop_dates = ["2025-01-17 23:55:00", "2025-02-18 20:10:00"]


        for month, starting_date, stoping_date in zip(months, start_dates, stop_dates):
            
            print("Calculating for month {} of year {}".format(month, year))

            start_date = pd.to_datetime(starting_date)
            stop_date = pd.to_datetime(stoping_date)

            path = "PV Data SARAFALI/{}-{}".format(year,month)

            month_df = pd.DataFrame()

            files_list = os.listdir(path)
            
            active_power_list = []

            for file in files_list:

                if 'EMI' in file:
                    
                    if year == "2025" and month == "01":
                        time_range = pd.date_range(start=start_date, end=stop_date, freq="5min")
                        emi_df = pd.DataFrame(time_range, columns=["Start Time"])
                        columns = ["Ambient temperature(℃)", "Daily irradiation(MJ/㎡)", "Irradiance(W/㎡)", "PV Temperature(℃)", "Wind direction(°)", "Wind speed(m/s)"]
                        for col in columns:
                            emi_df[col] = np.nan  # Initialize with NaN values
                        #print(len(emi_df))

  
                    else:
                        emi_df = pd.read_excel(path+"/"+file, header=3, usecols=['Daily irradiation(MJ/㎡)', 'Ambient temperature(℃)', 'Wind speed(m/s)', 'Start Time', 'PV Temperature(℃)', 'Wind direction(°)', 'Irradiance(W/㎡)'])
                        emi_df["Start Time"] = emi_df["Start Time"].astype(str).str.replace(" DST", "", regex=False)
                        emi_df["Start Time"] = pd.to_datetime(emi_df["Start Time"])
                        status = check_missing_extra_rows(emi_df, "Start Time" ,start_date, stop_date)
                        if status == 1:
                            emi_df = remove_extra_timesteps(emi_df, start_date, stop_date)
                        elif status == -1:
                            emi_df = fill_missing_timesteps(emi_df, start_date, stop_date, "emi")
                        


                    month_df['Timestep'] = emi_df["Start Time"]
                    month_df['AmbientTemperature'] = emi_df["Ambient temperature(℃)"]
                    month_df['Irradiation'] = emi_df["Daily irradiation(MJ/㎡)"]
                    month_df['Irradiance'] = emi_df["Irradiance(W/㎡)"]
                    month_df['PVTemperature'] = emi_df["PV Temperature(℃)"]
                    month_df['WindDirection'] = emi_df["Wind direction(°)"]
                    month_df['WindSpeed'] = emi_df["Wind speed(m/s)"]

                
                elif 'Inverter' in file:
                    #print('Inverter')
                    if year == "2025" and month == "01":
                        print("Filling nan data")
                    else:
                        inv_df = fill_missing_timesteps(pd.read_excel(path+"/"+file, header=3, usecols=["Start Time", "Active power(kW)"]), start_date, stop_date, "inv")
                        #print(len(inv_df))
                        #print(file+" :")
                        #print(inv_df[:200])
                        #if month == "09":
                        #    print(inv_df)

                        active_power_list.append(inv_df["Active power(kW)"])
            
            if year == "2025" and month == "01":
                month_df['ActivePower'] = np.nan
            else:
                active_power_df = pd.concat(active_power_list, axis=1)

                #print(active_power_df)

                aggregated_active_power = active_power_df.sum(axis=1)

                #print(aggregated_active_power)

                month_df['ActivePower'] = aggregated_active_power

                del aggregated_active_power, active_power_df

            #print(month_df)
            
            Sarafali_data = pd.concat([Sarafali_data, month_df], ignore_index=True)

            del month_df

            #print(Sarafali_data)
        
    
    return(Sarafali_data)

            
def replace_outliers(df, column, outlier_value=0, time_interval=288):

    df1 = df.copy()  # To avoid modifying the original DataFrame
    
    for i in range(len(df)):
        if df1.loc[i, column] == outlier_value:  # Check if value is an outlier
            j = i + time_interval  # Move forward by one day (288 steps)
            while j < len(df):
                if pd.isna(df.loc[j, column]) == False and df.loc[j, column] != outlier_value:
                    df1.loc[i, column] = df.loc[j, column]
                    break
                else:
                    j += time_interval  # Keep checking the next day
            
            if df1.loc[i, column] == outlier_value:
                j = i - time_interval
                while j > 0:
                    if pd.isna(df.loc[j, column]) == False and df.loc[j, column] != outlier_value:
                        df1.loc[i, column] = df.loc[j, column]
                        break
                    else:
                        j -= time_interval  # Keep checking the next day
    
    return df1


def replace_nans(df1):

    print("Fixing nans for 1 month period")

    df1["Timestep"] = pd.to_datetime(df1["Timestep"])

    df = df1.copy()

    for i in range(len(df)):

        if df.loc[i, "Timestep"] >= pd.to_datetime("2025-01-01 00:00:00") and df.loc[i, "Timestep"] < pd.to_datetime("2025-01-17 23:55:00"):
            
            date = df.loc[i, "Timestep"]

            for col in df.columns:

                if col != "Timestep":

                    matched_value = df.loc[df["Timestep"] == date - relativedelta(years=1), [col]]

                    if not matched_value.empty:  # Ensure a match is found
                        df.loc[i, col] = matched_value.values[0]  # Assign the value

    
    return df


def sin_month(x):

    final = np.sin(2*math.pi*(x/12))

    return final

def sin_minute(x):

    final = np.sin(2*math.pi*(x/288))

    return final

def sin_hour(x):

    final = np.sin(2*math.pi*(x/24))

    return final

def cos_month(x):

    final = np.cos(2*math.pi*(x/12))

    return final

def cos_minute(x):

    final = np.cos(2*math.pi*(x/288))

    return final

def cos_hour(x):

    final = np.cos(2*math.pi*(x/24))

    return final


def set_categories(df,cols):

    df = df.copy()

    for col in cols:

        df[col] = df[col].astype(str).astype("category")

    return df


def split(data):

    train = data.iloc[:int(len(data)*0.9),:]
    test = data.iloc[int(len(data)*0.9):,]
    return train,test


def create_time_index(df):

    df.insert(0, 'time_index', range(1, len(df) + 1))


def scale_tft(train_df, test_df):

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    feature_columns1 = [col for col in train_df.columns if col not in ["time_index", "group_id", "ActivePower", "Season"]]
    feature_columns2 = [col for col in train_df.columns if col=="ActivePower"]

    scaler1.fit(train_df.loc[:, feature_columns1])
    train_df.loc[:, feature_columns1] = scaler1.transform(train_df.loc[:, feature_columns1]).astype('float64')
    test_df.loc[:, feature_columns1] = scaler1.transform(test_df.loc[:, feature_columns1]).astype('float64')

    scaler2.fit(train_df.loc[:, feature_columns2])
    train_df.loc[:, feature_columns2] = scaler2.transform(train_df.loc[:, feature_columns2]).astype('float64')
    test_df.loc[:, feature_columns2] = scaler2.transform(test_df.loc[:, feature_columns2]).astype('float64')

    return train_df, test_df, scaler1, scaler2



data = data_collection()

data = replace_outliers(data, 'AmbientTemperature')
data = replace_outliers(data, 'PVTemperature')

data = replace_nans(data)


cols = ['AmbientTemperature', 'Irradiation', 'Irradiance', 'PVTemperature', 'WindDirection', 'WindSpeed']


for col in cols:

    x = np.arange(0,len(data))
    plt.figure(figsize=(20, 10))
    plt.plot(x, data[col], label=col, color='blue', alpha=0.7)

    plt.title("{} plot".format(col))
    plt.xlabel('Time-step')
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.show()

    
data.drop(columns=["WindDirection"], inplace=True)


seasons_dict = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}

data['Timestep'] = pd.to_datetime(data['Timestep'])

data['Month'] = data['Timestep'].dt.month
data['Hour'] = data['Timestep'].dt.hour
data['Minute'] = data['Timestep'].dt.minute
data['Season'] = data['Month'].map(seasons_dict)

data['Month_Sin'] = data['Month'].apply(sin_month)
data['Month_Cos'] = data['Month'].apply(cos_month)

data['Hour_Sin'] = data['Hour'].apply(sin_hour)
data['Hour_Cos'] = data['Hour'].apply(cos_hour)

data['Minute_Sin'] = data['Minute'].apply(sin_minute)
data['Minute_Cos'] = data['Minute'].apply(cos_minute)

data.drop(columns=["Month", "Hour", "Minute", "Timestep"], inplace=True)

print(data)


create_time_index(data)

data['group_id'] = 0

train_data, test_data =  split(data)
test_data = test_data.reset_index(drop=True)

train_data_scaled, test_data_scaled, scaler1, scaler2  = scale_tft(train_data, test_data)

with open('scalers/TFT_scaler1.pkl', 'wb') as f:
    pickle.dump(scaler1, f)


with open('scalers/TFT_scaler2.pkl', 'wb') as f:
    pickle.dump(scaler2, f)

train_scaled = set_categories(train_data_scaled, ['Season'])
test_scaled = set_categories(test_data_scaled, ['Season'])


train_scaled.to_pickle('PV Data SARAFALI/PV_SARAFALI_train.csv')
test_scaled.to_pickle('PV Data SARAFALI/PV_SARAFALI_test.csv')



