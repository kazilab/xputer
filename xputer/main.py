import torch
import nimfa
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.base import BaseEstimator
from numpy.linalg import svd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from xgboost import XGBRegressor, XGBClassifier
from joblib import parallel_backend
from tqdm import tqdm
from collections.abc import Iterable
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('Agg')


def count_dtypes(series):
    """
    Check the data type in a column.
    Args:
        series: data in a column.
    Returns:
        Count of each data type
    """
    if not isinstance(series, Iterable):
        raise ValueError("Input should be an iterable")

    counts = {'int_float': 0, 'str': 0, 'bool': 0, 'datetime': 0, 'other': 0}
    for item in series:
        if pd.isnull(item):  # Directly checks if the item is NaN, using pandas
            counts['other'] += 1
        elif isinstance(item, bool):  # Checks if the item is a boolean
            counts['bool'] += 1
        elif isinstance(item, str):
            if not item:  # Checks if the string is empty
                counts['other'] += 1
            else:
                try:
                    # Try to convert to a float
                    _ = float(item)
                    # If the value is an integer or float
                    counts['int_float'] += 1
                except ValueError:
                    # If it can't be converted to a float, consider it a string
                    counts['str'] += 1
        elif isinstance(item, datetime):  # Checks if the item is a datetime object
            counts['datetime'] += 1
        elif isinstance(item, (int, float)):
            counts['int_float'] += 1
        else:
            counts['other'] += 1
    return counts


def preprocessing_df(df, impute_zeros=False, pre_imputation='MixType',
                     test_mode=False, result_path=None, date_time=None):
    """
    Prepare df to use as data for XGBoost imputation. This will be done by checking each column, replacing 'object'
    data by numbers using label encoder, replacing any object data in numerical data column, etc.
    Args:
        df: data to be processed.
        impute_zeros: If zeros to be imputed
        pre_imputation: initial value for imputation
        test_mode: If true will save test control files
        result_path: Where results to be saved
        date_time: To note run time

    Returns:
        df_clean: same as original dataframe with NaN values
        df_encoded: labels are encoded with label encoder and then "np.nan" values were restored
        df_pre_imputed: df_encoded was imputed with mean or most-frequent values

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')

    if pre_imputation not in ['ColumnMean', 'MixType', 'KNNImputer']:
        raise ValueError('pre_imputation must be one of the following: ColumnMean, MixType or  KNNImputer.')

    df_pre_imputed = df.copy()
    # Define a possible list of missing value indicators including the list from pandas
    missing_value_indicators = ['NAN', 'Nan', 'NA#', '#VALUE!', '#DIV/0!', '-', '_', '--', '---', '__', '___',
                                '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND',
                                '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null',
                                '#NUM!']

    # Replace all missing value indicators with np.nan, update the df inplace
    df_pre_imputed.replace(missing_value_indicators, np.nan, inplace=True)
    if test_mode:
        df_pre_imputed.to_csv(result_path+'00_missing_values_replaced_df_pre_imputed_{}.csv'.format(date_time))

    # check each column,
    # find unwanted values for example letters in a mostly numeric column or numeric value in mostly object data column
    # first replace with np.nan, then for continuous data impute with column mean.
    #  if column contain mostly Nan, replace with zero

    df_clean = pd.DataFrame(index=df.index, columns=df.columns)
    df_encoded = pd.DataFrame(index=df.index, columns=df.columns)

    for column in df.columns:
        counts = count_dtypes(df[column])
        int_float = counts['int_float'] / df[column].shape[0]
        string = counts['str'] / df[column].shape[0]
        date_time_cont = counts['datetime'] / df[column].shape[0]
        bool_cont = counts['bool'] / df[column].shape[0]
        other = counts['other'] / df[column].shape[0]

        if int_float > 0.5:
            # Convert any string to NaN
            df_pre_imputed[column] = pd.to_numeric(df_pre_imputed[column], errors='coerce')
            # If we plan to replace zeros by a number
            if impute_zeros:
                df_pre_imputed[column].replace(0, np.nan, inplace=True)
            # append in dfs to create df with NaN
            df_clean[column] = df_pre_imputed[column]
            df_encoded[column] = df_pre_imputed[column]
            # Impute with mean
            if pre_imputation in ['ColumnMean', 'MixType']:
                df_pre_imputed[column].fillna(df_pre_imputed[column].mean(), inplace=True)
            else:
                impute_knn = KNNImputer(n_neighbors=5)
                imputed = impute_knn.fit_transform(df_pre_imputed[column].values.reshape(-1, 1))
                df_pre_imputed[column] = imputed.reshape(-1)

        elif string > 0.5:
            # Fill NaN with unique identifier 'ZZZ'
            df_pre_imputed[column].fillna(np.nan, inplace=True)
            df_pre_imputed[column].replace('nan', np.nan, inplace=True)
            df_pre_imputed[column] = df_pre_imputed[column].apply(lambda x: np.nan
                                                                  if not np.isnan(pd.to_numeric(x, errors='coerce'))
                                                                  else x)
            # print('pl1', df_pre_imputed[column].ravel())
            # make a copy to save the column in dfs
            df_clean[column] = df_pre_imputed[column]
            # print('pl2', df_pre_imputed[column].ravel())

            # Perform label encoding only on the non-nan part of the column
            non_nan_locs = df_pre_imputed[column].notnull()
            # print('pl3', non_nan_locs.ravel())
            le = LabelEncoder()
            df_pre_imputed.loc[non_nan_locs, column] = le.fit_transform(df_pre_imputed.loc[non_nan_locs, column])
            df_encoded[column] = df_pre_imputed[column]
            if pre_imputation == 'ColumnMean':
                # Impute with most frequent value
                imputer = SimpleImputer(strategy='most_frequent')
                imputed = imputer.fit_transform(df_pre_imputed[column].values.reshape(-1, 1))
                df_pre_imputed[column] = imputed.reshape(-1)
            else:
                impute_knn = KNNImputer(n_neighbors=5)
                imputed = impute_knn.fit_transform(df_pre_imputed[column].values.reshape(-1, 1))
                df_pre_imputed[column] = imputed.reshape(-1)

            df_pre_imputed[column] = df_pre_imputed[column].astype(int)
            # print('in preprocessing ',df_pre_imputed[column])

        elif date_time_cont > 0.5:
            raise ValueError(f'Xputer currently does not support datetime imputation, '
                             f'please remove datetime column: {column}')

        elif bool_cont > 0.5:
            df_pre_imputed[column] = df_pre_imputed[column].astype(str)
            df_pre_imputed[column] = df_pre_imputed[column].map({'True': True, 'False': False, 'nan': np.nan})
            df_clean[column] = df_pre_imputed[column]
            non_nan_locs = df_pre_imputed[column].notnull()
            df_pre_imputed.loc[non_nan_locs, column] = df_pre_imputed.loc[non_nan_locs, column].astype(int)
            df_encoded[column] = df_pre_imputed[column]
            if pre_imputation == 'ColumnMean':
                # Impute with most frequent value
                imputer = SimpleImputer(strategy='most_frequent')
                imputed = imputer.fit_transform(df_pre_imputed[column].values.reshape(-1, 1))
                df_pre_imputed[column] = imputed.reshape(-1)
            else:
                impute_knn = KNNImputer(n_neighbors=5)
                imputed = impute_knn.fit_transform(df_pre_imputed[column].values.reshape(-1, 1))
                df_pre_imputed[column] = imputed.reshape(-1)

            df_pre_imputed[column] = df_pre_imputed[column].astype(int)

        elif other > 0.5:
            # Replace with 0
            df_clean[column] = df_pre_imputed[column]
            df_pre_imputed[column] = 0
            df_encoded[column] = df_pre_imputed[column]

    return df_clean, df_encoded, df_pre_imputed


def cnmf(df_encoded, df_pre_imputed):
    """
        Calculate missing values using NMF.
        Args:
            df_encoded: Original dataframe that contain NaN values, will be used only to replace NaN values.
            df_pre_imputed: A secondary df that to be used for NMF,
            df_with_na and df can be identical or already imputed.
        Returns:
            Df with replaced at NaN after NMF and fully-transformed df.
        """
    # Impute missing values in df using nimfa.Nmf
    df = df_pre_imputed.copy()
    original_index = df.index
    df = df.reset_index(drop=True)

    ratio = df.shape[0] // df.shape[1]
    if ratio > 10:
        n_folds = 10
    elif ratio > 5:
        n_folds = 5
    elif ratio > 2:
        n_folds = 3
    else:
        n_folds = 2
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # pre_imputation a list to store the results
    reconstructions = []

    # Run NMF in a loop
    for i in range(10):
        for _, subset_indices in kf.split(df):
            subset = df.iloc[subset_indices]
        # pre_imputation and fit the NMF model
            nmf = nimfa.Nmf(subset.fillna(0).values, max_iter=200, random_state=i,
                            update='divergence', objective='div', rank=int(df.shape[1] * 0.95))
            nmf_fit = nmf()
            w = nmf_fit.basis()
            h = nmf_fit.coef()

            # Reconstruct the data
            reconstruction = np.dot(w, h)

            # Convert the reconstruction into a DataFrame with the same indexes as the subset
            df_reconstruction = pd.DataFrame(reconstruction, index=subset.index, columns=subset.columns)

            # Store the DataFrame in the list
            reconstructions.append(df_reconstruction)

    # Concatenate all the reconstructed DataFrames
    df_reconstructed = pd.concat(reconstructions)

    # If there are duplicate indexes, group by index and compute the mean
    df_nmf_transformed = df_reconstructed.groupby(df_reconstructed.index).mean()
    df_nmf_transformed.sort_index(inplace=True)
    df_nmf_transformed.index = original_index

    df_nmf_nan_imputed = df_encoded.copy()
    df_nmf_nan_imputed.to_csv('df_nmf_nan_imputed.csv')
    df_nmf_transformed.to_csv('df_nmf_transformed.csv')
    # df_nmf_nan_imputed[df_nmf_nan_imputed.isna()] = df_nmf_transformed
    for col in df_nmf_nan_imputed.columns:
        df_nmf_nan_imputed.loc[df_nmf_nan_imputed[col].isna(), col] = df_nmf_transformed[col]

    return df_nmf_nan_imputed, df_nmf_transformed


def run_svd(df_encoded, df_pre_imputed):
    """
        Calculate missing values using NMF.
        Args:
            df_encoded: Original dataframe that contain NaN values, will be used only to replace NaN values.
            df_pre_imputed: A secondary df that to be used for NMF,
            df_with_na and df can be identical or already imputed.
        Returns:
            Df with replaced at NaN after NMF and fully-transformed df.
        """
    # Impute missing values in df using nimfa.Nmf
    df = df_pre_imputed.copy()
    u, s, vt = svd(df)
    full_s = np.zeros(df.shape)
    full_s[:len(s), :len(s)] = np.diag(s)
    df_svd_transformed = pd.DataFrame(np.dot(u, np.dot(full_s, vt)), index=df.index, columns=df.columns)

    df_svd_nan_imputed = df_encoded.copy()
    # df_svd_nan_imputed[df_svd_nan_imputed.isna()] = df_svd_transformed
    for col in df_svd_nan_imputed.columns:
        df_svd_nan_imputed.loc[df_svd_nan_imputed[col].isna(), col] = df_svd_transformed[col]
    return df_svd_nan_imputed, df_svd_transformed


def optuna_xgb(x, y, label_d_type, tree_method, trials):

    """
    Perform Optuna hyperparameter search for a given model.

    Args:
        x: data matrix.
        y: labels.
        label_d_type: data type label - 'integer' or 'object'
        tree_method: tree method for GPU, 'gpu_hist'or 'auto'
        trials: number of trials, minimum 5.
    Returns:
        The best hyperparameter
    """

    n_trials = max(5, min(trials, 50))

    def cs_class(y_train, train_pred, y_valid, valid_pred):
        train_acc = accuracy_score(y_train, train_pred)
        valid_acc = accuracy_score(y_valid, valid_pred)
        diff = ((train_acc - valid_acc) ** 2) ** 0.5
        loss = 1 - valid_acc

        if loss == 0 and diff >= 0:
            cs = diff
        elif (diff / loss) * 100 < 2.5:
            cs = loss
        else:
            cs = (diff * loss) ** 0.5
        return cs

    def cs_reg(y_train, train_pred, y_valid, valid_pred):
        train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
        valid_rmse = mean_squared_error(y_valid, valid_pred) ** 0.5
        diff = ((train_rmse - valid_rmse) ** 2) ** 0.5

        if valid_rmse == 0 and diff >= 0:
            cs = diff
        elif (diff / valid_rmse) * 100 < 2.5:
            cs = valid_rmse
        else:
            cs = (diff * valid_rmse) ** 0.5
        return cs

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.1, step=0.005)
        max_depth = trial.suggest_int("max_depth", 3, 10, step=1)
        gamma = trial.suggest_float("gamma", 0, 4, step=0.5)
        reg_alpha = trial.suggest_float("reg_alpha", 0, 5, step=0.5)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 4, step=1)

        params_ = dict(learning_rate=learning_rate,
                       max_depth=max_depth,
                       gamma=gamma,
                       reg_alpha=reg_alpha,
                       min_child_weight=min_child_weight,
                       )
        kf = KFold(n_splits=4, random_state=18, shuffle=True)
        cv_score_array = []
        xx = x.to_numpy()
        yy = y.to_numpy()
        for train_index, test_index in kf.split(xx):
            x_train, x_valid = xx[train_index], xx[test_index]
            y_train, y_valid = yy[train_index], yy[test_index]
            if label_d_type == 'object':
                clf_opti = XGBClassifier(**params_, subsample=0.7, random_state=18, verbosity=0,
                                         tree_method=tree_method, n_estimators=100,
                                         scale_pos_weight=(len(y_train.ravel())
                                                           - sum(y_train.ravel())) / sum(y_train.ravel()))
            else:
                clf_opti = XGBRegressor(**params_, subsample=0.7, random_state=18, verbosity=0,
                                        tree_method=tree_method, n_estimators=100)

            clf_opti.fit(x_train.astype(float), y_train, verbose=False)
            train_pred = clf_opti.predict(x_train)
            valid_pred = clf_opti.predict(x_valid)
            if label_d_type == 'object':
                cs = cs_class(y_train, train_pred, y_valid, valid_pred)
            else:
                cs = cs_reg(y_train, train_pred, y_valid, valid_pred)
            cv_score_array.append(cs)
            # print('cv_score: ', cs)
        avg = np.mean(cv_score_array)
        # print('Average cv_score: ', avg)
        return avg

    study = optuna.create_study(direction="minimize",
                                study_name='Optuna optimization',
                                sampler=TPESampler()
                                )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Use joblib's parallel_backend to parallelize the optimization
    if tree_method == 'gpu_hist':
        study.optimize(objective, n_trials=n_trials, timeout=120)
    else:
        with parallel_backend('threading', n_jobs=-1):
            study.optimize(objective, n_trials=n_trials, timeout=120)

    parameters = study.best_params

    return parameters


def rxgb(df_encoded, df_nan_imputed, column, d_type, xgb_models, optuna_for_xgb, optuna_n_trials):
    """
    To run XGBRegression using a column as a label
    Args:
        df_encoded: The df we get after using pre_df function
        df_nan_imputed: The df we get after using cnmf function
        column: Selected column to use as label
        optuna_for_xgb: Whether Optuna be used for parameter searching
        d_type: data type label - 'integer' or 'object'
        xgb_models: number of iterations for xgb prediction
        optuna_n_trials: number of trials for Optuna

    Returns:
        Imputed column by XGBoost
    """
    # Split the data into two datasets:
    # - df_train: Rows where 'column' is not NaN
    # - df_pred: Rows where 'column' is NaN
    mask = df_encoded[column].isnull()
    if not mask.any():
        # No imputation necessary
        y = df_encoded[column].copy()
        return y

    if torch.cuda.is_available():
        tree_method = 'gpu_hist'
    else:
        tree_method = 'auto'

    xgb_models = max(3, min(xgb_models, 9))

    df_ori_nan_train = df_encoded[df_encoded[column].notna()]

    df_ori_imp_train = df_nan_imputed[df_encoded[column].notna()]
    df_ori_imp_pred = df_nan_imputed[df_encoded[column].isna()]

    # The target variable is 'column'
    y_train = df_ori_nan_train[column]

    if d_type == 'object':
        y_train = y_train.astype(int)
    else:
        y_train = y_train.astype(float)
    # print('y_train', y_train)

    # The features are all columns in the dataframe except 'column'
    x_train = df_ori_imp_train.drop(column, axis=1)
    x_pred = df_ori_imp_pred.drop(column, axis=1)

    train_spl = df_ori_nan_train.shape[0]
    train_feat = df_ori_nan_train.shape[1]
    num_columns_with_nan = df_encoded.isna().any().sum()

    if optuna_for_xgb and train_spl >= 50 and train_spl / train_feat >= 4 and num_columns_with_nan <= 120:

        parameters = optuna_xgb(df_ori_imp_train, y_train, d_type, tree_method, optuna_n_trials)

        # List to store the predicted values
        y_preds = []

        # Run the model n_iter times
        for i in range(xgb_models):

            if d_type == 'object':
                model = XGBClassifier(**parameters, subsample=0.7, random_state=i**2, verbosity=0,
                                      tree_method=tree_method, n_estimators=100,
                                      scale_pos_weight=(len(y_train.ravel())-sum(y_train.ravel()))/sum(y_train.ravel()))
            else:
                model = XGBRegressor(**parameters, subsample=0.7, random_state=i**2, verbosity=0,
                                     tree_method=tree_method, n_estimators=100)

            # Fit the model to the training data
            model.fit(x_train.astype(float), y_train, verbose=False)

            # Predict the missing values and store them in the list
            y_pred = model.predict(x_pred.astype(float))
            y_preds.append(y_pred)

        # Take the average of the predicted values
        y_pred_avg = np.mean(y_preds, axis=0)
    else:
        parameters = {'learning_rate': 0.1
                      }
        y_preds = []
        # Run the model n_iter times
        for i in range(xgb_models):
            if d_type == 'object':
                model = XGBClassifier(**parameters, subsample=0.7, n_estimators=100, random_state=i**2, verbosity=0,
                                      tree_method=tree_method,
                                      scale_pos_weight=(len(y_train.ravel())-sum(y_train.ravel()))/sum(y_train.ravel()))
            else:
                model = XGBRegressor(**parameters, subsample=0.7, n_estimators=100, random_state=i**2, verbosity=0,
                                     tree_method=tree_method)

            # Fit the model to the training data
            model.fit(x_train.astype(float), y_train, verbose=False)

            # Predict the missing values and store them in the list
            y_pred = model.predict(x_pred.astype(float))
            y_preds.append(y_pred)

        # Take the average of the predicted values
        y_pred_avg = np.mean(y_preds, axis=0)

    # Create an array to store the predicted values and the original non-missing values
    y = df_encoded[column].copy()
    y.loc[df_encoded[column].isna()] = y_pred_avg

    return y, parameters


def iter_rxgb(df_encoded, df_nan_imputed, column, d_type, xgb_models, xgb_parameters):
    """
    To run XGBRegression using a column as a label
    Args:
        df_encoded: The df we get after using pre_df function
        df_nan_imputed: The df we get after using cnmf function
        column: Selected column to use as label
        xgb_parameters: Parameters for XGBoost
        d_type: data type label - 'integer' or 'object'
        xgb_models: number of iterations for xgb prediction

    Returns:
        Imputed column by XGBoost
    """
    # Split the data into two datasets:
    # - df_train: Rows where 'column' is not NaN
    # - df_pred: Rows where 'column' is NaN
    mask = df_encoded[column].isnull()
    if not mask.any():
        # No imputation necessary
        y = df_encoded[column].copy()
        return y

    if torch.cuda.is_available():
        tree_method = 'gpu_hist'
    else:
        tree_method = 'auto'

    xgb_models = max(3, min(xgb_models, 9))

    df_ori_nan_train = df_encoded[df_encoded[column].notna()]

    df_ori_imp_train = df_nan_imputed[df_encoded[column].notna()]
    df_ori_imp_pred = df_nan_imputed[df_encoded[column].isna()]

    # The target variable is 'column'
    y_train = df_ori_nan_train[column]

    if d_type == 'object':
        y_train = y_train.astype(int)
    else:
        y_train = y_train.astype(float)
    # print('y_train', y_train)

    # The features are all columns in the dataframe except 'column'
    x_train = df_ori_imp_train.drop(column, axis=1)
    x_pred = df_ori_imp_pred.drop(column, axis=1)

    train_spl = df_ori_nan_train.shape[0]
    train_feat = df_ori_nan_train.shape[1]

    if train_spl >= 50 and train_spl / train_feat >= 4:

        parameters = xgb_parameters

        # List to store the predicted values
        y_preds = []

        # Run the model n_iter times
        for i in range(xgb_models):

            if d_type == 'object':
                model = XGBClassifier(**parameters, subsample=0.7, n_estimators=100, random_state=i**2, verbosity=0,
                                      tree_method=tree_method,
                                      scale_pos_weight=(len(y_train.ravel())-sum(y_train.ravel()))/sum(y_train.ravel()))
            else:
                model = XGBRegressor(**parameters, subsample=0.7, n_estimators=100, random_state=i**2, verbosity=0,
                                     tree_method=tree_method)
            # Fit the model to the training data
            model.fit(x_train.astype(float), y_train, verbose=False)
            # Predict the missing values and store them in the list
            y_pred = model.predict(x_pred.astype(float))
            y_preds.append(y_pred)

        # Take the average of the predicted values
        y_pred_avg = np.mean(y_preds, axis=0)
    else:
        parameters = xgb_parameters
        y_preds = []
        # Run the model n_iter times
        for i in range(xgb_models):
            if d_type == 'object':
                model = XGBClassifier(**parameters, subsample=0.7, n_estimators=100, random_state=i**2, verbosity=0,
                                      tree_method=tree_method,
                                      scale_pos_weight=(len(y_train.ravel())-sum(y_train.ravel()))/sum(y_train.ravel()))
            else:
                model = XGBRegressor(**parameters, subsample=0.7, n_estimators=100, random_state=i**2, verbosity=0,
                                     tree_method=tree_method)

            # Fit the model to the training data
            model.fit(x_train, y_train, verbose=False)

            # Predict the missing values and store them in the list
            y_pred = model.predict(x_pred)
            y_preds.append(y_pred)

        # Take the average of the predicted values
        y_pred_avg = np.mean(y_preds, axis=0)

    # Create an array to store the predicted values and the original non-missing values
    y = df_encoded[column].copy()
    y.loc[df_encoded[column].isna()] = y_pred_avg

    return y


def first_run(df_clean, df_encoded, df_nan_imputed, xgb_models, optuna_for_xgb, optuna_n_trials):
    """
    To run XGBRegression using a column as a label
    Args:
        df_clean: The df we get after preprocessing
        df_encoded: Cleaned df encoded by labeled encoder
        df_nan_imputed: The df received from previous imputation.
        xgb_models: number of iterations for xgb prediction
        optuna_for_xgb: Parameters optimization for XGBoost
        optuna_n_trials: number of trails
    Returns:
        Imputed column by XGBoost
    """
    xgb_parameters_dict = {}
    imputed = pd.DataFrame(False, index=df_clean.index, columns=df_clean.columns)
    df_clean_nan_imputed = df_clean.copy()
    df_encoded_nan_imputed = df_encoded.copy()
    df_clean_update = df_clean.copy()
    df_encoded_update = df_encoded.copy()

    # Iterate over all columns
    for column in tqdm(df_clean.columns):  # Wrap df.columns with tqdm(...)
        counts = count_dtypes(df_clean[column])
        int_float = counts['int_float'] / df_clean[column].shape[0]
        string = counts['str'] / df_clean[column].shape[0]
        date_time_cont = counts['datetime'] / df_clean[column].shape[0]
        bool_cont = counts['bool'] / df_clean[column].shape[0]
        other = counts['other'] / df_clean[column].shape[0]
        xgb_parameters = None
        # Check if the column has missing values
        if df_clean_update[column].isna().sum() > 0:
            # Mark the missing values in the 'imputed' dataframe
            imputed.loc[df_clean_update[column].isna(), column] = True
            # Check if the column is numerical or categorical
            if int_float > 0.5:
                y, xgb_parameters = rxgb(df_encoded, df_nan_imputed, column, 'integer',
                                         xgb_models, optuna_for_xgb, optuna_n_trials)
                df_clean_update[column] = y
                df_encoded_update[column] = y
            elif string > 0.5:
                # Perform label encoding only on the non-nan part of the column
                non_nan_locs = df_clean_update[column].notnull()

                # Label encoding
                le = LabelEncoder()
                df_clean_update.loc[non_nan_locs, column] = le.fit_transform(df_clean_update.loc[non_nan_locs, column])
                # print('Xpute2',df_copy[column])

                y, xgb_parameters = rxgb(df_encoded, df_nan_imputed, column, 'object',
                                         xgb_models, optuna_for_xgb, optuna_n_trials)
                df_clean_update[column] = y.astype(int)
                df_encoded_update[column] = y.astype(int)
                # After imputation, inverse transform the encoded labels back to original form
                df_clean_update.loc[df_clean_update[column].notna(), column] = le.inverse_transform(
                    df_clean_update.loc[df_clean_update[column].notna(), column])

            elif date_time_cont > 0.5:
                raise ValueError(f'Xputer currently does not support datetime imputation, '
                                 f'please remove datetime column: {column}')

            elif bool_cont > 0.5:
                # Perform label encoding only on the non-nan part of the column
                non_nan_locs = df_clean_update[column].notnull()
                df_clean_update.loc[non_nan_locs, column] = df_clean_update.loc[non_nan_locs, column].astype(int)

                y, xgb_parameters = rxgb(df_encoded, df_nan_imputed, column, 'object',
                                         xgb_models, optuna_for_xgb, optuna_n_trials)
                df_clean_update[column] = y.astype(int)
                df_encoded_update[column] = y.astype(int)
                # Convert int back to bool
                df_clean_update[column] = df_clean_update[column].astype(bool)

            elif other > 0.5:
                df_clean_update[column] = df_clean_update[column]
                df_encoded_update[column] = df_encoded_update[column]
                xgb_parameters = None

            df_clean_nan_imputed.loc[df_clean_nan_imputed[column].isna(), column] = df_clean_update[column]
            df_encoded_nan_imputed.loc[df_encoded_nan_imputed[column].isna(), column] = df_encoded_update[column]
        xgb_parameters_dict[column] = xgb_parameters
    return xgb_parameters_dict, imputed, df_clean_nan_imputed, df_encoded_nan_imputed


def iterative(df_clean, df_encoded, df_nan_imputed_prev, xgb_models, xgb_parameters_dict):
    """
    To run XGBRegression using a column as a label
    Args:
        df_clean: The df we get after preprocessing
        df_encoded: Cleaned df encoded by labeled encoder
        df_nan_imputed_prev: The df received from previous imputation.
        xgb_models: number of iterations for xgb prediction
        xgb_parameters_dict: Parameters for XGBoost
    Returns:
        Imputed column by XGBoost
    """

    df_clean_nan_imputed = df_clean.copy()
    df_encoded_nan_imputed = df_encoded.copy()
    df_clean_update = df_clean.copy()
    df_encoded_update = df_encoded.copy()

    # Iterate over all columns
    for column in tqdm(df_clean.columns):  # Wrap df.columns with tqdm(...)
        counts = count_dtypes(df_clean[column])
        int_float = counts['int_float'] / df_clean[column].shape[0]
        string = counts['str'] / df_clean[column].shape[0]
        date_time_cont = counts['datetime'] / df_clean[column].shape[0]
        bool_cont = counts['bool'] / df_clean[column].shape[0]
        other = counts['other'] / df_clean[column].shape[0]
        # Check if the column has missing values
        if df_clean_update[column].isna().sum() > 0:
            # Check if the column is numerical or categorical
            if int_float > 0.5:
                y = iter_rxgb(df_encoded, df_nan_imputed_prev, column, 'integer', xgb_models,
                              xgb_parameters_dict[column])
                df_clean_update[column] = y
                df_encoded_update[column] = y
            elif string > 0.5:
                # Perform label encoding only on the non-nan part of the column
                non_nan_locs = df_clean_update[column].notnull()

                # Label encoding
                le = LabelEncoder()
                df_clean_update.loc[non_nan_locs, column] = le.fit_transform(df_clean_update.loc[non_nan_locs, column])
                # print('Xpute2',df_copy[column])

                y = iter_rxgb(df_encoded, df_nan_imputed_prev, column, 'object', xgb_models,
                              xgb_parameters_dict[column])
                # print('string-pred done', y)
                df_clean_update[column] = y.astype(int)
                df_encoded_update[column] = y.astype(int)

                # After imputation, inverse transform the encoded labels back to original form
                df_clean_update.loc[df_clean_update[column].notna(), column] = le.inverse_transform(
                    df_clean_update.loc[df_clean_update[column].notna(), column])

            elif date_time_cont > 0.5:
                raise ValueError(f'Xputer currently does not support datetime imputation, '
                                 f'please remove datetime column: {column}')

            elif bool_cont > 0.5:
                # Perform label encoding only on the non-nan part of the column
                non_nan_locs = df_clean_update[column].notnull()
                df_clean_update.loc[non_nan_locs, column] = df_clean_update.loc[non_nan_locs, column].astype(int)

                y = iter_rxgb(df_encoded, df_nan_imputed_prev, column, 'object', xgb_models,
                              xgb_parameters_dict[column])
                df_clean_update[column] = y.astype(int)
                df_encoded_update[column] = y.astype(int)
                # Convert int back to bool
                df_clean_update[column] = df_clean_update[column].astype(bool)

            elif other > 0.5:
                df_clean_update[column] = df_clean_update[column]
                df_encoded_update[column] = df_encoded_update[column]

            df_clean_nan_imputed.loc[df_clean_nan_imputed[column].isna(), column] = df_clean_update[column]
            df_encoded_nan_imputed.loc[df_encoded_nan_imputed[column].isna(), column] = df_encoded_update[column]

    return df_clean_nan_imputed, df_encoded_nan_imputed


def plot_all_columns(df_original, df_imputed, pdf_filename):
    """
    Visualize original and imputed values in all columns of a DataFrame and save plots in a PDF file.
    Args:
        df_original: The original DataFrame before imputation.
        df_imputed: The DataFrame after imputation.
        pdf_filename: The filename for the output PDF file.
    """
    # Select only numeric columns
    # df_clean_numeric = df_original.select_dtypes(include='number')
    # identify columns with missing values
    # missing_cols = df_clean_numeric.columns[df_clean_numeric.isna().any()].tolist()
    with PdfPages(pdf_filename) as pdf:
        print('Saving plots')
        # for column in tqdm(df_original.columns):
        for column in tqdm(df_original.columns):
            # Check if the column contains NaN values
            # if df_original[column].isnull().any():
            if df_original[column].dtype in ['int64', 'float64']:
                # Create a mask for imputed values
                imputed_mask = df_original[column].isnull() & df_imputed[column].notnull()

                # Create a scatter plot for original values
                plt.figure(figsize=(6, 4))
                plt.scatter(df_original.index, df_original[column], color='gray', label='Original', s=2)

                # Add imputed values to the plot in red
                plt.scatter(df_imputed.index[imputed_mask], df_imputed[column][imputed_mask],
                            color='red', label='Imputed', s=2)

                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title('Imputed Values in Column: ' + column)
                plt.legend()

                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()


def plot_nan_col_dot_plot(df_clean, df_clean_nan_imputed_dict, pdf_filename):

    # Select only numeric columns
    df_clean_numeric = df_clean.select_dtypes(include='number')
    # identify columns with missing values
    missing_cols = df_clean_numeric.columns[df_clean_numeric.isna().any()].tolist()

    # create a pdf file to save all the plots
    with PdfPages(pdf_filename) as pdf:
        for col in tqdm(missing_cols):
            plt.figure(figsize=(10, 6))

            colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray',
                      'lightblue', 'cyan', 'magenta', 'lime', 'yellow']  # define a color list
            sns.scatterplot(x=df_clean.index, y=df_clean[col], color='gray', label='Original', s=25)
            for i in range(len(df_clean_nan_imputed_dict)):
                imputed_mask = df_clean[col].isnull() & df_clean_nan_imputed_dict[i][col].notnull()

                # Plot density for each iteration's DataFrame
                sns.scatterplot(x=df_clean_nan_imputed_dict[i].index[imputed_mask],
                                y=df_clean_nan_imputed_dict[i][col][imputed_mask],
                                label=f'Iteration {i}', color=colors[i], s=15)

            plt.title(f'Dot Plot for {col} Across Iterations')
            plt.xlabel(col)
            plt.ylabel('Value')
            plt.legend()

            # Save the current figure to the pdf file
            pdf.savefig()
            plt.close()


def plot_nan_col_kdeplot(df_clean, df_clean_nan_imputed_dict, pdf_filename):

    # Select only numeric columns
    df_clean_numeric = df_clean.select_dtypes(include='number')
    # identify columns with missing values
    missing_cols = df_clean_numeric.columns[df_clean_numeric.isna().any()].tolist()

    # create a pdf file to save all the plots
    with PdfPages(pdf_filename) as pdf:
        for col in tqdm(missing_cols):
            plt.figure(figsize=(10, 6))
            sns.kdeplot(df_clean[col], label='Original', color='black')
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray',
                      'lightblue', 'cyan', 'magenta', 'lime', 'yellow']  # define a color list
            for i in range(len(df_clean_nan_imputed_dict)):
                # Plot density for each iteration's DataFrame
                sns.kdeplot(df_clean_nan_imputed_dict[i][col], label=f'Iteration {i}', color=colors[i])

            plt.title(f'Density Plot for {col} Across Iterations')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()

            # Save the current figure to the pdf file
            pdf.savefig()
            plt.close()


# Function to predict missing values


class Xpute(BaseEstimator):

    def __init__(self,
                 impute_zeros=False,
                 pre_imputation='MixType',
                 xgb_models=3,
                 mf_for_xgb=False,
                 use_transformed_df=False,
                 optuna_for_xgb=False,
                 optuna_n_trials=5,
                 n_iterations=1,
                 save_imputed_df=False,
                 save_plots=False,
                 test_mode=False
                 ):
        """
        Impute using XGBoost
        Args:
            impute_zeros: If zeros to be imputed
            pre_imputation: To pre-impute with a value, 'mean' will initiate simple imputer, otherwise KNNImputer
            xgb_models: how many times each column to be predicted in a single run
            mf_for_xgb: Whether NMF to be used for pre-imputation
            use_transformed_df: Whether fully NMF transformed df to be used or only NaN transformed to be used
            optuna_for_xgb: Whether XGBoost parameters to be optimized by Optuna
            optuna_n_trials: How many optuna trials to be used
            n_iterations: number of iterations
            save_imputed_df: To save xputed df
            save_plots: To save plots
            test_mode: If true will save test control files

        Returns:
            Imputed df
        """
        self.impute_zeros = impute_zeros
        self.pre_imputation = pre_imputation
        self.xgb_models = xgb_models
        self.mf_for_xgb = mf_for_xgb
        self.use_transformed_df = use_transformed_df
        self.optuna_for_xgb = optuna_for_xgb
        self.optuna_n_trials = optuna_n_trials
        self.n_iterations = n_iterations
        self.save_imputed_df = save_imputed_df
        self.save_plots = save_plots
        self.test_mode = test_mode
        self.df_clean_nan_imputed_dict = {}

    def fit(self, df):
        """
        Impute using XGBoost
        Args:
            df: The df we get after using pre_df function

        Returns:
            Imputed df
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        result_path = None
        date_time = None
        if self.save_imputed_df or self.save_plots or self.test_mode:
            user_documents = os.path.expanduser("~/Documents")
            result_path = os.path.join(user_documents, "Xputed_results/")
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")

        df_copy = df.copy()

        # Preprocess df to get a clean copy of df, encoded df with NaN and simply imputed encoded df
        print("Preprocessing data")
        df_clean, df_encoded, df_pre_imputed = preprocessing_df(df_copy, self.impute_zeros,
                                                                self.pre_imputation, self.test_mode,
                                                                result_path, date_time)

        if self.test_mode:
            df_clean.to_csv(result_path+'01_preprocessing_df_df_clean_{}.csv'.format(date_time))
            df_encoded.to_csv(result_path+'01_preprocessing_df_df_encoded_{}.csv'.format(date_time))
            df_pre_imputed.to_csv(result_path+'01_preprocessing_df_df_pre_imputed_{}.csv'.format(date_time))

        # Use cNMF or svd to transform data
        min_value = np.min(np.min(df_pre_imputed, axis=0))

        if min_value >= 0 and self.use_transformed_df:
            print("Computing cNMF, full df transformed values will be used")
            _, df_nan_imputed = cnmf(df_encoded, df_pre_imputed)
        elif min_value >= 0 and self.mf_for_xgb:
            print("Computing cNMF, only NaN transformed values will be used, initial values remain the same")
            df_nan_imputed, _ = cnmf(df_encoded, df_pre_imputed)
        elif min_value < 0 and self.use_transformed_df:
            print("Computing SVD, full df transformed values will be used")
            _, df_nan_imputed = run_svd(df_encoded, df_pre_imputed)
        elif min_value < 0 and self.mf_for_xgb:
            print("Computing SVD, only NaN transformed values will be used, initial values remain the same")
            df_nan_imputed, _ = run_svd(df_encoded, df_pre_imputed)
        else:
            df_nan_imputed = df_pre_imputed

        if self.test_mode:
            df_nan_imputed.to_csv(result_path+'02_cnmf_df_nan_imputed_{}.csv'.format(date_time))

        xgb_parameters_dict, imputed, df_clean_nan_imputed, df_encoded_nan_imputed = first_run(df_clean,
                                                                                               df_encoded,
                                                                                               df_nan_imputed,
                                                                                               self.xgb_models,
                                                                                               self.optuna_for_xgb,
                                                                                               self.optuna_n_trials)

        if self.test_mode:
            pd.DataFrame(xgb_parameters_dict).to_csv(result_path+'03_first_run_xgb_parameters_dict_{}.csv'.format(date_time))
            imputed.to_csv(result_path+'03_first_run_imputed_{}.csv'.format(date_time))
            df_clean_nan_imputed.to_csv(result_path+'03_first_run_df_clean_nan_imputed_{}.csv'.format(date_time))
            df_encoded_nan_imputed.to_csv(result_path+'03_first_run_df_encoded_nan_imputed_{}.csv'.format(date_time))

        n_iterations = max(1, min(self.n_iterations, 9))
        # pre_imputation dictionaries to store the output dataframes

        # Add initial inputs to dictionaries
        self.df_clean_nan_imputed_dict[0] = df_clean_nan_imputed
        for i in range(n_iterations):
            df_clean_nan_imputed, df_encoded_nan_imputed = iterative(df_clean, df_encoded,
                                                                     df_encoded_nan_imputed, self.xgb_models,
                                                                     xgb_parameters_dict)
            # Add the output dataframes to the dictionaries
            self.df_clean_nan_imputed_dict[i+1] = df_clean_nan_imputed
            if self.test_mode:
                df_clean_nan_imputed.to_csv(result_path+f'04_iterative_df_clean_nan_imputed_{i}_{date_time}.csv')
                df_encoded_nan_imputed.to_csv(result_path+f'04_iterative_df_encoded_nan_imputed_{i}_{date_time}.csv')

        if self.save_plots:
            plot_nan_col_kdeplot(df_clean, self.df_clean_nan_imputed_dict,
                                 result_path + 'Xputed_value_density_plots_{}.pdf'.format(date_time))
            plot_nan_col_dot_plot(df_clean, self.df_clean_nan_imputed_dict,
                                  result_path + 'Xputed_value_dot_plots_{}.pdf'.format(date_time))

        if self.save_imputed_df:
            df_clean_nan_imputed.to_csv(result_path + 'Xputed_data_{}.csv'.format(date_time))

        # if self.save_plots:
        #    plot_all_columns(df_clean, df_clean_nan_imputed,
        #                     result_path + 'Xputed_value_plots_{}.pdf'.format(date_time))

        return df_clean_nan_imputed


def xpute(df,
          impute_zeros=False,
          pre_imputation='MixType',
          xgb_models=3,
          mf_for_xgb=False,
          use_transformed_df=False,
          optuna_for_xgb=False,
          optuna_n_trials=5,
          n_iterations=1,
          save_imputed_df=False,
          save_plots=False,
          test_mode=False):
    """
    Impute using XGBoost
    Args:
        df: The df we get after using pre_df function
        impute_zeros: If zeros to be imputed
        pre_imputation: To pre-imputate with a value, 'mean' will initiate simple imputer, otherwise KNNImputer
        xgb_models: how many times each column to be predicted in a single run
        mf_for_xgb: Whether NMF to be used for pre-imputation
        use_transformed_df: Whether fully NMF transformed df to be used or only NaN transformed to be used
        optuna_for_xgb: Whether XGBoost parameters to be optimized by Optuna
        optuna_n_trials: How many optuna trials to be used
        n_iterations: number of iterations
        save_imputed_df: To save xputed df
        save_plots: To save plots
        test_mode: If true will save test control files

    Returns:
        Imputed df
    """

    xp = Xpute(impute_zeros, pre_imputation, xgb_models, mf_for_xgb, use_transformed_df, optuna_for_xgb,
               optuna_n_trials, n_iterations, save_imputed_df, save_plots, test_mode).fit(df)

    return xp
