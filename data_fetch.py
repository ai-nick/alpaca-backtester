import pandas as pd
import numpy as np
import os
from statistics import mode
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import datetime as dt
import alpaca_trade_api as alpaca
#import matplotlib.pyplot as plt


api_creds = {
    "id": "",
    "key": ""
}

class AlpacaWrapper(object):
    bardata_points = ["date", "open", "high", "low", "close", "adj_close", "volume"]

    num_points = len(bardata_points)

    backtest_start_unix = 1435449600

    currentHists = {}

    train_data_directory = "./train_data/stonks/"

    live_data_directory = "./live_data/stonks/"

    long_short_dict = {
        "SPX": {
            "LONG": "SPXL",
            "SHORT": "SPXS"
        },
        "FA": {
            "LONG": "FAS",
            "SHORT": "FAZ"
        },
        "TA": {
            "LONG": "TNA",
            "SHORT": "TZA"
        },
        "SOX": {
            "LONG": "SOXL",
            "SHORT": "SOXS"
        },
        "TEC": {
            "LONG": "TECL",
            "SHORT": "TECS"
        },
        "LAB": {
            "LONG": "LABU",
            "SHORT": "LABD"
        }
    }

    def __init__(self, config={}):
        self.start_idxs = {}
        self.alpaca_client = alpaca.REST(config["id"], config["key"], base_url='https://paper-api.alpaca.markets')
        return

    def request_yahoo_data(self, sym):
        dl_url = self.five_year_template.format(sym, str((dt.date.today().toordinal() - dt.date(1970, 1, 1).toordinal())  * 24*60*60))
        hist_response = urlopen(dl_url)
        df = pd.read_csv(hist_response)
        df.to_csv("./hist_data/yahoo_long_short/" + sym + ".txt")

    def get_bar_data(self, symbols, limit=100, from_date=None):
        if from_date == None:
            bar_data = self.alpaca_client.get_barset(symbols, "day", limit=100).df
            return bar_data
        else:
            bar_data = self.alpaca_client.get_barset(symbols, "day", start=from_date, limit=1000).df
            return bar_data

    def fetch_and_store_long_short(self):
        self.save_historical_alpaca_data()

    def load_long_short(self):
        return_dict = {}
        for x in self.long_short_dict:
            x_long = self.long_short_dict[x]["LONG"]
            x_short = self.long_short_dict[x]["SHORT"]
            df_long = pd.read_csv("./hist_data/yahoo_long_short/" + x_long + ".txt")
            df_short = pd.read_csv("./hist_data/yahoo_long_short/" + x_short + ".txt")
            df_long = self.apply_features(df_long).reset_index()
            df_short = self.apply_features(df_short).reset_index()
            return_dict[x] = {"BULL": df_long, "BEAR": df_short}
        return return_dict

    def load_long_short_live(self):
        return_dict = {}
        for x in self.long_short_dict:
            x_long = self.long_short_dict[x]["LONG"]
            x_short = self.long_short_dict[x]["SHORT"]
            df_long = pd.read_csv("./live_data/yahoo_long_short/" + x_long + ".txt")
            df_short = pd.read_csv("./live_data/yahoo_long_short/" + x_short + ".txt")
            df_long = self.apply_features_live(df_long).reset_index()
            df_short = self.apply_features_live(df_short).reset_index()
            print(df_long.head())
            print(df_long.tail())
            return_dict[x] = {"BULL": df_long, "BEAR": df_short}
        return return_dict

    def load_live_frames(self):
        return_dict = {}
        for x in self.long_short_dict:
            x_long = self.long_short_dict[x]["LONG"]
            x_short = self.long_short_dict[x]["SHORT"]
            df_long = pd.read_csv(self.live_data_directory + x_long + ".txt")
            df_short = pd.read_csv(self.live_data_directory + x_short + ".txt")
            df_long = self.apply_features_live(df_long).reset_index()
            df_short = self.apply_features_live(df_short).reset_index()
            print(df_long.head())
            print(df_long.tail())
            df_long["date"] = df_long["Unnamed: 0"]
            df_short["date"] = df_short["Unnamed: 0"]
            return_dict[x] = {"BULL": df_long, "BEAR": df_short}
        return return_dict

    def load_historical_frames(self):
        return_dict = {}
        for x in self.long_short_dict:
            x_long = self.long_short_dict[x]["LONG"]
            x_short = self.long_short_dict[x]["SHORT"]
            df_long = pd.read_csv(self.train_data_directory + x_long + ".txt")
            df_short = pd.read_csv(self.train_data_directory + x_short + ".txt")
            df_long = self.apply_features_live(df_long).reset_index()
            df_short = self.apply_features_live(df_short).reset_index()
            if len(df_long) != len(df_short):
                if len(df_long) > len(df_short):
                    difference = len(df_long) - len(df_short)
                    df_long = df_long[difference:].reset_index()
                else:
                    difference = len(df_short) - len(df_long)
                    df_short = df_short[difference:].reset_index()
            df_long["date"] = df_long["Unnamed: 0"]
            df_short["date"] = df_short["Unnamed: 0"]
            return_dict[x] = {"BULL": df_long, "BEAR": df_short}
        return return_dict

    def save_live_alpaca_data(self):
        sym_list = [self.long_short_dict[x]["LONG"] for x in self.long_short_dict] + [self.long_short_dict[x]["SHORT"] for x in self.long_short_dict]
        bar_data = self.get_bar_data(sym_list)
        for s in sym_list:
            print(s + " :", bar_data[s].head())
            bar_data[s].to_csv(self.live_data_directory + s + ".txt")
        return

    def save_historical_alpaca_data(self):
        sym_list = [self.long_short_dict[x]["LONG"] for x in self.long_short_dict] + [self.long_short_dict[x]["SHORT"] for x in self.long_short_dict]
        start_time = dt.datetime.fromtimestamp(self.backtest_start_unix).isoformat()
        bar_data = self.get_bar_data(sym_list, from_date=start_time)
        for s in sym_list:
            print(s + " :", bar_data[s].head())
            bar_data[s].to_csv(self.train_data_directory + s + ".txt")
        return

    def load_long_short_single(self, sym, live=False):
        long_df = self.apply_features(pd.read_csv("./hist_data/yahoo_long_short/" + self.long_short_dict[sym]["LONG"] + ".txt"))
        short_df = self.apply_features(pd.read_csv("./hist_data/yahoo_long_short/" + self.long_short_dict[sym]["SHORT"] + ".txt"))
        long_df.reset_index(inplace=True)
        short_df.reset_index(inplace=True)
        return_dict = {sym: {"BEAR": short_df, "BULL": long_df}}
        return return_dict
        
    def load_long_short_single_live(self, sym, live=False):
        long_df = self.apply_features_live(pd.read_csv("./live_data/yahoo_long_short/" + self.long_short_dict[sym]["LONG"] + ".txt"))
        short_df = self.apply_features_live(pd.read_csv("./live_data/yahoo_long_short/" + self.long_short_dict[sym]["SHORT"] + ".txt"))
        long_df.reset_index(inplace=True)
        short_df.reset_index(inplace=True)
        return_dict = {sym: {"BEAR": short_df, "BULL": long_df}}
        return return_dict

    def get_train_frames_all_syms(self, restrict_val = 0, feature_columns = ["roc_short", "close_short", "short_mid", "mid_long", "avg_vol_mid", "std_close", "std_low", "std_open"]):
        df_dict = self.load_historical_frames()
        coin_and_hist_index = 0
        currentHists = df_dict
        hist_shaped = {}
        coin_dict = {}
        self.prefixes = []
        hist_lengths = {}
        for s in df_dict:
            self.prefixes.append(s)
            df_bull = currentHists[s]["BULL"][feature_columns].copy()
            df_bear = currentHists[s]["BEAR"][feature_columns].copy()
            print(df_bear.head())
            self.start_idxs[s] = df_bull.index[0]
            hist_lengths[s] = len(df_bull)
            as_array_bull = np.array(df_bull)
            as_array_bear = np.array(df_bear)
            hist_shaped[coin_and_hist_index] = as_array_bull
            coin_dict[s] = coin_and_hist_index
            coin_and_hist_index += 1
            hist_shaped[coin_and_hist_index] = as_array_bear
            coin_and_hist_index += 1
        hist_shaped = pd.Series(hist_shaped)
        return coin_dict, currentHists, hist_shaped, hist_lengths

    def get_train_frames_single_sym(self, symbol, restrict_val=0, feature_columns = ['std_low', 'std_open', 'avg_vol_89', "roc_close_short", "rolling_8", "rolling_55"]):
        df_dict = self.load_long_short_single(symbol)
        coin_and_hist_index = 0
        currentHists = df_dict
        hist_shaped = {}
        coin_dict = {}
        prefixes = []
        df_bull = df_dict[symbol]["BULL"][feature_columns].copy()
        self.start_idx = df_bull.index[0]
        df_bear = df_dict[symbol]["BEAR"][feature_columns].copy()
        hist_full_size = len(df_bull)
        as_array_bull = np.array(df_bull)
        as_array_bear = np.array(df_bear)
        print(as_array_bear[0])
        hist_shaped[coin_and_hist_index] = as_array_bull
        coin_and_hist_index += 1
        coin_dict[coin_and_hist_index] = as_array_bear
        hist_shaped = pd.Series(hist_shaped)
        print(hist_shaped.shape)
        return symbol, currentHists, hist_shaped, hist_full_size

    def get_live_frames_single_sym(self, symbol, restrict_val=0, feature_columns = ['std_low', 'std_open', 'avg_vol_89', "roc_close_short", "rolling_8", "rolling_55"]):
        self.save_live_data()
        df_dict = self.load_long_short_single_live(symbol)
        coin_and_hist_index = 0
        currentHists = df_dict
        hist_shaped = {}
        coin_dict = {}
        prefixes = []
        df_bull = df_dict[symbol]["BULL"][feature_columns].copy()
        self.start_idx = df_bull.index[0]
        df_bear = df_dict[symbol]["BEAR"][feature_columns].copy()
        hist_full_size = len(df_bull)
        as_array_bull = np.array(df_bull)
        as_array_bear = np.array(df_bear)
        print(as_array_bear[0])
        hist_shaped[coin_and_hist_index] = as_array_bull
        coin_and_hist_index += 1
        coin_dict[coin_and_hist_index] = as_array_bear
        hist_shaped = pd.Series(hist_shaped)
        print(hist_shaped.shape)
        return symbol, currentHists, hist_shaped, hist_full_size

    def get_live_frames_all_syms(self, restrict_val=0, feature_columns = ["roc_short", "close_short", "short_mid", "mid_long", "avg_vol_mid", "std_close", "std_low", "std_open"]):
        self.save_live_alpaca_data()
        df_dict = self.load_live_frames()
        coin_and_hist_index = 0
        currentHists = df_dict
        hist_shaped = {}
        coin_dict = {}
        self.prefixes = []
        hist_lengths = {}
        for s in df_dict:
            self.prefixes.append(s)
            df_bull = currentHists[s]["BULL"][feature_columns].copy()
            df_bear = currentHists[s]["BEAR"][feature_columns].copy()
            self.start_idxs[s] = df_bull.index[0]
            hist_lengths[s] = len(df_bull)
            as_array_bull = np.array(df_bull)
            as_array_bear = np.array(df_bear)
            hist_shaped[coin_and_hist_index] = as_array_bull
            coin_dict[s] = coin_and_hist_index
            coin_and_hist_index += 1
            hist_shaped[coin_and_hist_index] = as_array_bear
            coin_and_hist_index += 1
        hist_shaped = pd.Series(hist_shaped)
        return coin_dict, currentHists, hist_shaped, hist_lengths
        return

    def apply_features(self, df):
        df['std_close'] = df['Close']/df['High']
        df['std_low'] = df['Low']/df['High']
        df['std_open'] = df['Open']/df['High']
        df['avg_vol_mid'] = pd.Series(np.where(df.Volume.rolling(8).mean() > df.Volume, 1, -1), df.index)
        #df = self.add_rsi(df)
        #df = self.add_willr(df)
        df['vol_cross'] =  df.Volume / df.Volume.rolling(8).mean()
        df["ma_short"] = df.Close.rolling(3).mean()
        df["ma_mid"] = df.Close.rolling(13).mean()
        df["ma_long"] = df.Close.rolling(55).mean()
        df["close_short"] = df.Close / df.ma_short
        df["short_mid"] = df.ma_short / df.ma_mid
        df["mid_long"] = df.ma_mid / df.ma_long
        #df["roc"] = df["Open"].pct_change(periods=8)
        #df["roc_long"] = df["Close"].pct_change(periods=34)
        #df["roc_mid"] = df["Close"].pct_change(periods=13)
        df["roc_short"] = df["Close"].pct_change(periods=5)
        df.dropna(inplace=True)
        self.start_idx = df.index[0]
        return df
    
    def apply_features_live(self, df):
        df['std_close'] = df['close']/df['high']
        df['std_low'] = df['low']/df['high']
        df['std_open'] = df['open']/df['high']
        df['avg_vol_mid'] = pd.Series(np.where(df.volume.rolling(8).mean() > df.volume, 1, -1), df.index)
        df['vol_cross'] =  df.volume / df.volume.rolling(8).mean()
        df["ma_short"] = df.close.rolling(3).mean()
        df["ma_mid"] = df.close.rolling(13).mean()
        df["ma_long"] = df.close.rolling(55).mean()
        df["close_short"] = df.close / df.ma_short
        df["short_mid"] = df.ma_short / df.ma_mid
        df["mid_long"] = df.ma_mid / df.ma_long
        df["roc_short"] = df["close"].pct_change(periods=5)
        df.dropna(inplace=True)
        self.start_idx = df.index[0]
        return df

#wrapper = AlpacaWrapper(api_creds)
#wrapper.save_historical_alpaca_data()
