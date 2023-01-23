import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import config

class MinMaxScaler:
    """Scale data using min max standardisation"""
    def __init__(self):
        self.min = 0
        self.max = 1

    def __str__(self):
        return f"<MinMaxSclaer(min={self.min}, max={self.max})>"

    def fit(self, data):
        self.min = min(data)
        self.max = max(data)

    def encode(self, data):
        scaler = lambda x: (x-self.min)/(self.max-self.min)
        scaled_data = []
        scaled_data = [scaler(i) for i in data]
        return data

    def decode(self, scaled_data):
        unscaler = lambda x: x*(self.max-self.min) + self.min
        data = []
        scaled_data = [unscaler(i) for i in scaled_data]
        return data

class LabelEncoder:
    """Encode labels into unique indices."""
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

def create_features(df):
    """
    Create new features from existing columns
    """
    df['Age'] = 2022 - df['Year_Birth']
    get_customer_year = lambda x: int(x.split('/')[-1])
    df['Enrollment'] = 2022 - df['Dt_Customer'].apply(get_customer_year)
    return df

def clean_marital_data(marital_data):
    """
    Clean Marital Status feature by aggregating miscellaneous replies into others
    """
    fixed_marital_status = ['Divorced', 'Single', 'Married', 'Together', 'Widow', 'Others']
    clean_marital_status = lambda x: x if x in fixed_marital_status else 'Others'
    cleaned_marital_data = marital_data.apply(clean_marital_status)
    return cleaned_marital_data

def label_encoder(data):
    """
    Label Encoder
    """
    encoder = LabelEncoder()
    labels = data.unique()
    encoder.fit(labels)
    return encoder

def create_and_fit_all_label_encoder(df, cols):
    """
    Create and encode all categorical columns
    """
    label_encoder_store = {}
    for col in cols:
        encoder = label_encoder(df[col])
        label_encoder_store[col] = encoder
        df[col] = encoder.encode(df[col])
    return df, label_encoder_store

def data_minmax_scaler(data):
    """
    Min max scaler for continuous variable
    """
    scaler = MinMaxScaler()
    scaler.fit(data.values)
    return scaler

def create_and_fit_all_data_scaler(df, cols):
    """
    Create and scale al continuous columns
    """
    data_scaler_store = {}
    for col in cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        scaler = data_minmax_scaler(df[col])
        data_scaler_store[col] = [median, scaler]
        df[col] = scaler.encode(df[col].values)
    return df, data_scaler_store

def clean_data(df):
    """
    Data cleaning
    """
    df['Marital_Status'] = clean_marital_data(df['Marital_Status'])
    return df

def preprocess(df, xcols, y_cols):
    """
    Preprocess raw data into final data for model fitting
    """
    df = create_features(df)
    df = clean_data(df)
    df, label_encoder = create_and_fit_all_label_encoder(df, config.CATEGORICAL_COLS)
    df, data_minmax_scaler = create_and_fit_all_data_scaler(df, config.CONTINUOUS_COLS)
    X, y = df[xcols], df[y_cols]
    return (X,y), label_encoder, data_minmax_scaler

def get_data_splits(X, y, train_size=0.7, seed=42):
    """
    Generate balanced data splits.
    """
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test