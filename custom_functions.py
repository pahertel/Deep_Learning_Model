# Custom Functions

# Initial imports

# General
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import date
import datetime
import json
import time

# scikit-learn / TensorFlow / Keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from joblib import dump, load

#------------------------------------------- API Integration ----------------------------------

def Piotroski_F_score(df, ni, ta, cfo, ltl, ca, cl, d_shr_out, gp, rev):

    """
    Calculates the Piotroski F-score
    """

    # df = ttm_fund_data_combined
    # ni = 'Net Income (Common)'
    # ta = 'Total Assets'
    # cfo = 'Net Cash from Operating Activities'
    # ltl = 'Total Noncurrent Liabilities'
    # ca = 'Total Current Assets'
    # cl = 'Total Current Liabilities'
    # d_shr_out = 'Shares (Diluted)'
    # gp = 'Gross Profit'
    # rev = 'Revenue'

    # main column name
    f_score = 'Piotroski F-score'

    # Piotroski F-score ROA
    # Begining year total assets
    df[f_score] = (np.where(

        ( df[ni] / df[ta] )
        >
        0
        , 1.0, 0.0))

    # Piotroski F-score Cash Flow from Operating (scaled by Total Assets)
    # Begining year total assets
    df[f_score] = (df[f_score] + np.where(

        ( df[cfo] / df[ta] )
        > 
        0
        , 1.0, 0.0))

    # Piotroski F-score Change in ROA
    # Begining year total assets
    df[f_score] = (df[f_score] + np.where(

        ( ( df[ni] / df[ta] ) - ( df[ni].shift(4) / df[ta].shift(4) ) )
        > 
        0
        , 1.0, 0.0))

    # Piotroski F-score ROA - CFO
    # are the assets right?
    df[f_score] = (df[f_score] + np.where(

        ( df[ni] / df[ta] )
        > 
        ( df[cfo] / df[ta] )
        , 1.0, 0.0))   

    # Piotroski F-score Change in Long Term Debt Ratio
    # Average total assets
    # Total Noncurrent Liabilities vs Total debt
    df[f_score] = (df[f_score] + np.where(

        ( ( df[ltl].shift(4) / df[ta].shift(4) ) - ( df[ltl] / df[ta] ) )
        > 
        0
        , 1.0, 0.0))  

    # Piotroski F-score Change in Current ratio
    df[f_score] = (df[f_score] + np.where(

        ( ( df[ca] / df[cl] ) - ( df[ca].shift(4) / df[cl].shift(4) ) )
        > 
        0
        , 1.0, 0.0))

    # Piotroski F-score Change in shares outstanding
    df[f_score] = (df[f_score] + np.where(

        ( ( df[d_shr_out].shift(4) ) - ( df[d_shr_out] ) )
        > 
        0
        , 1.0, 0.0))

    # Piotroski F-score Change in Gross Margin Ratio
    df[f_score] = (df[f_score] + np.where(

        ( df[gp] / df[rev] )
        > 
        ( df[gp].shift(4) / df[rev].shift(4) )
        , 1.0, 0.0)) 

    # Piotroski F-score Change in Asset Turnover Ratio
    # Begining year total assets
    df[f_score] = (df[f_score] + np.where(

        ( df[rev] / df[ta] )
        > 
        ( df[rev].shift(4) / df[ta].shift(4) )
        , 1.0, 0.0))
    
    return df[f_score]


def col_name_change(df, specified_ending):
    """
    Creates a list of column names and adds a specified ending to the column name
    """
    
    list_col = df.columns.tolist()
    columns = []

    #column_name = specified_ending # " specified_ending"
    for col in list_col:
        col = col + specified_ending # " specified_ending"
        columns.append(col)
    return columns



def join_obj_loop(main_df, join_df, join_obj):
    
    """
    Joins the data calculated sum data into each stock in the main df
    """
    
    # main_df = ttm_data_all
    # join_df = ttm_data_all_clean_all
    # join_obj = level_0 or Sector or Industry
    
    # Creates date as index and tickers a value for creating a list of tickers for index values
    #single_obj = main_df.copy().reset_index().set_index(['level_1'])

    # Creates a list of tickers for index values
    join_list = main_df.copy().reset_index().set_index([join_obj]).index.drop_duplicates()

    # Resets index back into ticker and date
    single_obj = main_df.reset_index().set_index([join_obj, 'level_1'])

    # Create an empty dataframe that will be the combined version
    combined_df = pd.DataFrame()

    count = 0
    for ticker in join_list:
        # create new data fram with ticker and industry data
        single_combined = single_obj.loc[[ticker]].join(join_df)

        # add dataframes topgether
        combined_df = combined_df.append(single_combined)
        
        # Calculates percentage of completion
        count = count + 1
        percent = (count/len(join_list)) * 100
        time.sleep(.01)
        print(f"{round(percent, 1)}%", end="\r")
    
    return combined_df

    

def enterprise_value(df, mc, std, ltd, cash):
    
    """
    Calculates the Enterprise Value but excludes long term cash and long term cash equivelents
    """
    
    # df = aapl_comparable_calculations
    # Market Cap = 'Market Cap'
    # std = 'Short Term Debt'
    # ltd = 'Long Term Debt'
    # cash = 'Cash, Cash Equivalents & Short Term Investments'
    
    df["Enterprise Value"] = df[mc] + df[std] + df[ltd] - df[cash]
    
    return df["Enterprise Value"]



def shareholder_value(df, repur, debt, div):
    
    """
    Calculates shareholder value
    """
    
    # df = aapl_comparable_calculations
    # repur = 'Cash from (Repurchase of) Equity'
    # debt = 'Cash from (Repayment of) Debt'
    # div = 'Dividends Paid'
    
    df["Shareholder Value"] = -df[repur] + -df[debt] + -df[div]
    
    return df["Shareholder Value"]



def shareholder_yield(df, sharval, mc):

    """
    Calculates the shareholder value with market cap or something such as EV
    """
    
    # df = aapl_comparable_calculations
    # sharval = 'Shareholder Value'
    # mc = 'Market Cap'
    
    df["Shareholder Yield"] = (df[sharval] / df[mc]) * 100
    
    return df["Shareholder Yield"]



def div_repurchase_value(df, repur, div):

    """
    Calculates divind and share repurchase value
    """
    
    # df = aapl_comparable_calculations
    # repur = 'Cash from (Repurchase of) Equity'
    # div = 'Dividends Paid'
    
    df["Div + Repurchase Value"] = -df[repur] + -df[div]
    
    return df["Div + Repurchase Value"]



def div_repurchase_yield(df, divrepur, mc):

    """
    Calculates the divind and share repurchase yield with market cap or something such as EV
    """
    
    # df = aapl_comparable_calculations
    # divrepur = 'Div + Repurchase Value'
    # mc = 'Market Cap'
    
    df["Div + Repurchase Yield"] = (df[divrepur] / df[mc]) * 100
    
    return df["Div + Repurchase Yield"]



def pe_ratio(df, mc, ni):
    
    """
    Calculates the P/E ratio
    """
    
    # df = aapl_comparable_calculations
    # mc = 'Market Cap'
    # ni = 'Net Income (Common)'
    
    df["PE Market Cap to Earnings"] = df[mc] / df[ni]
    
    return df["PE Market Cap to Earnings"]


# Coverage Ratio
def coverage_ratio(df, ni, div):
    
    """
    Calculates the coverage ratio of dividneds bases on things such as earning or free cash flow
    """
    
    # df = aapl_comparable_calculations
    # ni = 'Net Income (Common)'
    # div = 'Dividends Paid'
    
    df["Coverage Ratio"] = df[ni] / -(df[div])
    
    return df["Coverage Ratio"]



def yield_calc(df, obj, mc):
    
    """
    Calculates the yield of somethings such as EBIT to market cap or enterprise value
    """
    
    # df = aapl_comparable_calculations
    # obj = 'Free Cash Flow'
    # mc = 'Market Cap'
    
    df["Yield"] = (df[obj] / df[mc]) * 100
    
    return df["Yield"]


def ebit(df, ni, intr, tax):

    """
    Calculates the EBIT value
    """

    # df = aapl_comparable_calculations
    # ni = 'Net Income (Common)'
    # intr = 'Interest Expense, Net'
    # tax = 'Income Tax (Expense) Benefit, Net'
    
    df["EBIT"] = df[ni] + -(df[intr]) + -(df[tax])
    
    return df["EBIT"]


def ebitda(df, ebit, da):

    """
    Calculates the EBITDA value
    """

    # df = aapl_comparable_calculations
    # ebit = 'EBIT'
    # da = 'Depreciation & Amortization'
    
    df["EBITDA"] = df[ebit] + df[da]
    
    return df["EBITDA"]



def weighted_average(df, obj, mc, i):

    """
    Calculates a weighted Average when placed between the clean data before the for loop to merge dataframes.
    """
    
    # df = ttm_data_all_wieghted
    # i = level_0 or Sector or Industry
    # obj = 'Close'
    # mc = 'Market Cap'
    
    df['Weighted Return'] = df[obj] * ( df[mc] / df[f'{mc}{i}'] )
    
    return df["Weighted Return"]


#------------------------------------------ Deep Learning ----------------------------------------

def LSTM_Model(df_main, target_col):

    # Select number of epochs
    num_epochs = 500

    # Select number of epochs when model stops after no imporvement 
    early_stop = 30

    # Select the ratio between test and train data
    ratio = .70

    # Select window period to be used
    window = 4

    # copy Dataframe
    df = df_main.copy()

    # shift data to avoid look-ahead bias
    df_column_list = df.columns.tolist()
    for col in df_column_list:
        if col == target_col:
            # Leave Target Vector
            print('')
        else:
            # Shift Features
            df[col] = df[col].shift(1)
    # Drop NAs and Replace Infs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the MinMaxScaler object with the features data X
    scaler.fit(df)
    # Scale the features
    df_scaled = scaler.transform(df)

    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df_scaled[i : (i + window), 1: ]
        target = df_scaled[(i + window), 0]
        X.append(features)
        y.append(target)
    X, y = np.array(X), np.array(y).reshape(-1, 1)

    # Ratio Selection (ie. 70%) Test Train Split
    split = int(ratio * len(X))

    X_train = X[: split]
    X_test = X[split:]

    y_train = y[: split]
    y_test = y[split:]

    # number of nuerons in each layer
    layer_1_nuerons = 50
    layer_2_nuerons = 50
    layer_3_nuerons = 50 # have this equal to window maybe

    # Dropout Fraction
    dropout_fraction = 0.2

    # Define the LSTM RNN model.
    model = Sequential()

    model.add(LSTM(
        units=layer_1_nuerons,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=True))
    model.add(Dropout(dropout_fraction))

    # Layer 2
    model.add(LSTM(units=layer_2_nuerons,
                   return_sequences=True))
    model.add(Dropout(dropout_fraction))

    # Layer 3
    model.add(LSTM(units=layer_3_nuerons,
                   return_sequences= False))
    model.add(Dropout(dropout_fraction))

    # Output layer
    model.add(Dense(y_train.shape[1]))

    #from tensorflow.keras.models import Sequential
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = early_stop,
        mode = 'min')

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=num_epochs,
        shuffle=False,
        batch_size=90,
        validation_split = 0.1,
        verbose=1,
        callbacks = [early_stopping])

    model_loss = pd.DataFrame({
        "Train Loss": history.history["loss"],
        "Val Loss": history.history["val_loss"]
    })
    model_loss = model_loss.rename_axis('Epochs')

    # Evaluate the model
    model_eval = model.evaluate(X_test, y_test, verbose=0)

    # Make predictions using the testing data X_test
    predicted = model.predict(X_test)

    # Recover the original prices instead of the scaled version
    predicted_copies = np.repeat(predicted, df_scaled.shape[1], axis = -1)
    predicted_prices = scaler.inverse_transform(predicted_copies)

    predicted_df = []
    for i in range(len(predicted_prices)):
        pred_pri = predicted_prices[i][0]
        predicted_df.append(pred_pri)

    predicted_df = pd.DataFrame(predicted_df,columns=[f'Predicted {target_col}'])

    actual_df = df_main[target_col].iloc[-predicted_df.shape[0]:]

    actual_df = actual_df.reset_index()

    # Merge dataframes
    aapl_lstm_df = pd.concat([actual_df, predicted_df], axis = 1)
    aapl_lstm_df = aapl_lstm_df.reset_index().set_index(['Date'])
    aapl_lstm_df.drop(columns = ['index'], inplace=True)

    return model, aapl_lstm_df, model_loss, model_eval


