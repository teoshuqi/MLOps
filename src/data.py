import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import config

class MinMaxScaler:
    """Scale data using min max standardisation"""
    def __init__(self):
        self.min = 0
        self.max = 1
        self.median = 0.5

    def __str__(self):
        return f"<MinMaxSclaer(min={self.min}, max={self.max})>"

    def fit(self, data):
        self.min = min(data)
        self.max = max(data)
        self.median = np.nanmedian(data)

    def encode(self, data):
        scaler = lambda x: (x-self.min)/(self.max-self.min)
        scaled_data = []
        for d in data:
            if pd.notna(d):
                scaled_data.append(scaler(d))
            else:
                scaled_data.append(scaler(self.median))
        return scaled_data

    def decode(self, scaled_data):
        unscaler = lambda x: x*(self.max-self.min) + self.min
        unscaled_data = []
        unscaled_data = [unscaler(i) for i in scaled_data]
        return unscaled_data

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

def fit_all_label_encoder(df, label_encoders, cols):
    """
    Encode all categorical columns
    """
    for col in cols:
        if col in label_encoders:
            encoder = label_encoders[col]
            df[col] = encoder.encode(df[col])
    return df

def create_all_label_encoder(df, cols):
    """
    Create all categorical columns
    """
    label_encoder_store = {col:label_encoder(df[col]) for col in cols}
    return label_encoder_store

def data_minmax_scaler(data):
    """
    Min max scaler for continuous variable
    """
    scaler = MinMaxScaler()
    scaler.fit(data.values)
    return scaler

def fit_all_data_scaler(df, data_scalers, cols):
    """
    Fit scaler for all continuous columns
    """
    for col in cols:
        if col in data_scalers:
            scaler = data_scalers[col]
            df[col] = scaler.encode(df[col])
    return df

def create_all_data_scaler(df, cols, fit=True):
    """
    Create scaler for all continuous columns
    """
    data_scaler_store = {col:data_minmax_scaler(df[col]) for col in cols}
    return data_scaler_store

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
    label_encoder_store = create_all_label_encoder(df, config.CATEGORICAL_COLS)
    df = fit_all_label_encoder(df, label_encoder_store, config.CATEGORICAL_COLS)
    minmax_scaler_store = create_all_data_scaler(df, config.CONTINUOUS_COLS)
    df = fit_all_data_scaler(df, minmax_scaler_store, config.CONTINUOUS_COLS)
    X, y = df[xcols], df[y_cols]
    return (X,y), label_encoder_store, minmax_scaler_store

def preprocess_predict(df, artifacts):
    """
    Preprocess raw data into final data for prediction
    """
    df = create_features(df)
    df = clean_data(df)
    label_encoder_store = artifacts['label_encoder']

    df = fit_all_label_encoder(df, label_encoder_store, config.CATEGORICAL_COLS)
    minmax_scaler_store = artifacts['minmax_scaler']
    df = fit_all_data_scaler(df, minmax_scaler_store, config.CONTINUOUS_COLS)
    return df

def get_data_splits(X, y, train_size=0.7, seed=42):
    """
    Generate balanced data splits.
    """
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test