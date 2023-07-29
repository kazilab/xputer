import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator
from .utils import preprocessing_df, cnmf, run_svd, first_run, iterative, plot_all_columns
# Function to predict missing values


class Xpute(BaseEstimator):

    def __init__(self,
                 impute_zeros=False,
                 initialize='mean',
                 xgb_iter=0,
                 mf_for_xgb=False,
                 use_transformed_df=False,
                 optuna_for_xgb=False,
                 optuna_n_trials=5,
                 iterations=False,
                 n_iterations=0,
                 save_imputed_df=False,
                 save_plots=False,
                 test_mode=False
                 ):
        """
        Impute using XGBoost
        Args:
            impute_zeros: If zeros to be imputed
            initialize: To initialize a value, 'mean' will initiate simple imputer, otherwise KNNImputer
            xgb_iter: how many times each column to be predicted in a single run
            mf_for_xgb: Whether NMF to be used for pre-imputation
            use_transformed_df: Whether fully NMF transformed df to be used or only NaN transformed to be used
            optuna_for_xgb: Whether XGBoost parameters to be optimized by Optuna
            optuna_n_trials: How many optuna trials to be used
            iterations: Whether iterative imputation to be used
            n_iterations: number of iterations
            save_imputed_df: To save xputed df
            save_plots: To save plots
            test_mode: If true will save test control files

        Returns:
            Imputed df
        """
        self.impute_zeros = impute_zeros
        self.initialize = initialize
        self.xgb_iter = xgb_iter
        self.mf_for_xgb = mf_for_xgb
        self.use_transformed_df = use_transformed_df
        self.optuna_for_xgb = optuna_for_xgb
        self.optuna_n_trials = optuna_n_trials
        self.iterations = iterations
        self.n_iterations = n_iterations
        self.save_imputed_df = save_imputed_df
        self.save_plots = save_plots
        self.test_mode = test_mode

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

        df_copy = df.copy()

        # Preprocess df to get a clean copy of df, encoded df with NaN and simply imputed encoded df
        print("Preprocessing data")
        df_clean, df_encoded, df_pre_imputed = preprocessing_df(df_copy, self.impute_zeros,
                                                                self.initialize, self.test_mode)

        if self.test_mode:
            df_clean.to_csv('01_preprocessing_df_df_clean.csv')
            df_encoded.to_csv('01_preprocessing_df_df_encoded.csv')
            df_pre_imputed.to_csv('01_preprocessing_df_df_pre_imputed.csv')

        # Use cNMF or svd to transform data
        min_value = np.mean(np.mean(df_pre_imputed, axis=0))

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
            df_nan_imputed.to_csv('02_cnmf_df_nan_imputed.csv')

        xgb_parameters_dict, imputed, df_clean_nan_imputed, df_encoded_nan_imputed = first_run(df_clean,
                                                                                               df_encoded,
                                                                                               df_nan_imputed,
                                                                                               self.xgb_iter,
                                                                                               self.optuna_for_xgb,
                                                                                               self.optuna_n_trials)

        if self.test_mode:
            pd.DataFrame(xgb_parameters_dict).to_csv('03_first_run_xgb_parameters_dict.csv')
            imputed.to_csv('03_first_run_imputed.csv')
            df_clean_nan_imputed.to_csv('03_first_run_df_clean_nan_imputed.csv')
            df_encoded_nan_imputed.to_csv('03_first_run_df_encoded_nan_imputed.csv')

        if self.iterations:
            n_iterations = max(1, min(self.n_iterations, 9))
            for _ in range(n_iterations):
                df_clean_nan_imputed, df_encoded_nan_imputed = iterative(df_clean, df_encoded,
                                                                         df_encoded_nan_imputed, self.xgb_iter,
                                                                         xgb_parameters_dict)
                if self.test_mode:
                    df_clean_nan_imputed.to_csv('04_iterative_df_clean_nan_imputed.csv')
                    df_encoded_nan_imputed.to_csv('04_iterative_df_encoded_nan_imputed.csv')
        else:
            df_clean_nan_imputed = df_clean_nan_imputed

        result_path = None
        date_time = None
        if self.save_imputed_df or self.save_plots:
            user_documents = os.path.expanduser("~/Documents")
            result_path = os.path.join(user_documents, "Xputed_results/")
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")

        if self.save_imputed_df:
            df_clean_nan_imputed.to_csv(result_path + 'Xputed_data_{}.csv'.format(date_time))

        if self.save_plots:
            plot_all_columns(df_clean, df_clean_nan_imputed,
                             result_path + 'Xuted_value_plots_{}.pdf'.format(date_time))

        return df_clean_nan_imputed


def xpute(df,
          impute_zeros=False,
          initialize='mean',
          xgb_iter=0,
          mf_for_xgb=False,
          use_transformed_df=False,
          optuna_for_xgb=False,
          optuna_n_trials=5,
          iterations=False,
          n_iterations=0,
          save_imputed_df=False,
          save_plots=False,
          test_mode=False):
    """
    Impute using XGBoost
    Args:
        df: The df we get after using pre_df function
        impute_zeros: If zeros to be imputed
        initialize: To initialize a value, 'mean' will initiate simple imputer, otherwise KNNImputer
        xgb_iter: how many times each column to be predicted in a single run
        mf_for_xgb: Whether NMF to be used for pre-imputation
        use_transformed_df: Whether fully NMF transformed df to be used or only NaN transformed to be used
        optuna_for_xgb: Whether XGBoost parameters to be optimized by Optuna
        optuna_n_trials: How many optuna trials to be used
        iterations: Whether iterative imputation to be used
        n_iterations: number of iterations
        save_imputed_df: To save xputed df
        save_plots: To save plots
        test_mode: If true will save test control files

    Returns:
        Imputed df
    """

    xp = Xpute(impute_zeros, initialize, xgb_iter, mf_for_xgb, use_transformed_df, optuna_for_xgb, optuna_n_trials,
               iterations, n_iterations, save_imputed_df, save_plots, test_mode).fit(df)

    return xp
