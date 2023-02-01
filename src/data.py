import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import config
from typing import Dict, List, Tuple

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

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns

    Args:
        df (pd.DataFrame): pandas data frame with raw customer data

    Returns:
        pd.DataFrame: Dataframe with addtional derived features
    """
    df['Age'] = 2022 - df['Year_Birth']
    get_customer_year = lambda x: int(x.split('/')[-1])
    df['Enrollment'] = 2022 - df['Dt_Customer'].apply(get_customer_year)
    return df

def clean_marital_data(marital_data: pd.Series) -> pd.Series:
    """
    Clean Marital Status feature by aggregating miscellaneous replies into others

    Args:
        marital_data (pd.Series): dataframe series with marital data of customers
    
    Retunrs:
        pd.Series: cleaned marital data for customers
    """
    fixed_marital_status = ['Divorced', 'Single', 'Married', 'Together', 'Widow', 'Others']
    clean_marital_status = lambda x: x if x in fixed_marital_status else 'Others'
    cleaned_marital_data = marital_data.apply(clean_marital_status)
    return cleaned_marital_data

def label_encoder(data: pd.Series) -> LabelEncoder:
    """
    Fit label encoder on feature provided

    Args:
        data (pd.Series): feature with labels to encode

    Returns:
        LabelEncoder: LabelEncoder Class for feature provided
    """
    encoder = LabelEncoder()
    labels = data.unique()
    encoder.fit(labels)
    return encoder

def fit_all_label_encoder(df: pd.DataFrame, 
                          label_encoders:Dict = {}, 
                          cols:List = config.CATEGORICAL_COLS) -> pd.DataFrame:
    """
    Encode all categorical columns

    Args:
        df (pd.DataFrame): Customer data to transform
        label_encoders (Dict): Dict of LabelEncoder class for each feature
        cols (List): List of features to fit label encoders
    
    Returns:
        pd.DataFrame: Customer data with labels encoded
    """
    for col in cols:
        if col in label_encoders:
            encoder = label_encoders[col]
            df[col] = encoder.encode(df[col])
    return df

def create_all_label_encoder(df:pd.DataFrame, 
                            cols:List = config.CATEGORICAL_COLS) -> Dict:
    """
    Fit label encoder for all given features of customer data

    Args:
        data (pd.Series): feature with labels to encode
        cols (List): List of features to fit label encoders

    Returns:
        Dict: Dict of LabelEncoder class for each feature
    """
    label_encoder_store = {col:label_encoder(df[col]) for col in cols}
    return label_encoder_store

def data_minmax_scaler(data: pd.Series) -> MinMaxScaler:
    """
    Fit min max scaler on feature provided

    Args:
        data (pd.Series): feature with data to scale

    Returns:
        MinMaxScaler: MinMaxScaler Class for feature provided
    """
    scaler = MinMaxScaler()
    scaler.fit(data.values)
    return scaler

def fit_all_data_scaler(df:pd.DataFrame, 
                        data_scalers:Dict = {}, 
                        cols:List = config.CONTINUOUS_COLS) -> pd.DataFrame:
    """
    Encode all continuous columns

    Args:
        df (pd.DataFrame): Customer data to transform
        data_scalers (Dict): Dict of MinMaxScaler class for each feature
        cols (List): List of features to scale
    
    Returns:
        pd.DataFrame: Customer data with features scaled
    """
    for col in cols:
        if col in data_scalers:
            scaler = data_scalers[col]
            df[col] = scaler.encode(df[col])
    return df

def create_all_data_scaler(df: pd.DataFrame, 
                           cols:List = config.CONTINUOUS_COLS
                           ) -> Dict:
    """
    Fit data scaler for all given features of customer data

    Args:
        data (pd.Series): feature with labels to encode
        cols (List): List of features to scale

    Returns:
        Dict: Dict of MinMaxScaler class for each feature
    """
    data_scaler_store = {col:data_minmax_scaler(df[col]) for col in cols}
    return data_scaler_store

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning
    """
    df['Marital_Status'] = clean_marital_data(df['Marital_Status'])
    return df

def preprocess(df:pd.DataFrame, 
               xcols:List, 
               y_cols: List) -> Tuple:
    """
    Preprocess raw data into final data for model fitting

    Args:
        df (pd.DataFrame): Customer data to transform
        xcols: list of features
        ycols: response name
    
    Returns:
        Tuple: transformed data and its corresponding artifacts

    """
    df = create_features(df)
    df = clean_data(df)
    label_encoder_store = create_all_label_encoder(df, config.CATEGORICAL_COLS)
    df = fit_all_label_encoder(df, label_encoder_store, config.CATEGORICAL_COLS)
    minmax_scaler_store = create_all_data_scaler(df, config.CONTINUOUS_COLS)
    df = fit_all_data_scaler(df, minmax_scaler_store, config.CONTINUOUS_COLS)
    X, y = df[xcols], df[y_cols]
    return (X,y), label_encoder_store, minmax_scaler_store

def preprocess_predict(df:pd.DataFrame, 
                       artifacts:Dict) -> Tuple:
    """
    Preprocess raw data into final data for prediction

    Args:
        df (pd.DataFrame): Customer data to transform
        artifacts (Dict): Artifacts to be used to preprocess data
    
    Returns:
        Tuple: transformed data and its corresponding artifacts

    """
    df = create_features(df)
    df = clean_data(df)
    label_encoder_store = artifacts['label_encoder']

    df = fit_all_label_encoder(df, label_encoder_store, config.CATEGORICAL_COLS)
    minmax_scaler_store = artifacts['minmax_scaler']
    df = fit_all_data_scaler(df, minmax_scaler_store, config.CONTINUOUS_COLS)
    return df

def get_data_splits(X:pd.DataFrame, 
                    y:pd.Series, 
                    train_size:float=0.7, seed:int=42):
    """Generate balanced data splits.
    Args:
        X (pd.Series): input features.
        y (np.ndarray): encoded labels.
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.
        seed (int, optional): seed number. Default to 42
    Returns:
        Tuple: data splits as Numpy arrays.
    """
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test