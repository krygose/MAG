import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gamma
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from mag_data_creation import get_data_as_df, get_data_from_csv_file, get_halfnormal_data, get_synthetic_data_based_on_similarity_to_real_points, get_half_synthetic_data_based_on_similarity_to_real_points
from mag_data_creation import get_data_without_columns_with_nan_as_df,get_data_with_human_added_values_as_df,get_data_with_additional_columns_as_df
from mag_data_creation import get_gamma_data, get_uniform_data, get_data_from_kernel_density_estimation, get_data_from_normal_distribution
from ML_technics import predict_by_autoencoder, predict_outlier_by_MLP, train_model_MLP,regression_by_linear, regression_by_elastic_net, regression_by_elastic_net_with_PCA, regression_by_XGBoost, regression_by_random_forest, regression_by_polynomial
from ploting import plot_correlation_matrix, plot_generated_values_for_df



df_set = [
    ("original", get_data_as_df()),
    ("no_nan", get_data_without_columns_with_nan_as_df()),
    ("human_added", get_data_with_human_added_values_as_df()),
    ("additional_columns", get_data_with_additional_columns_as_df()),
    
]


for name, df_test in df_set:
    generated_df_set = [
        ("halfnormal", get_halfnormal_data(df_test)),
        ("synthetic_similar", get_synthetic_data_based_on_similarity_to_real_points(df_test)),
        ("half_synthetic", get_half_synthetic_data_based_on_similarity_to_real_points(df_test)),
        ("gamma", get_gamma_data(df_test)),
        ("uniform", get_uniform_data(df_test)),
        ("kde", get_data_from_kernel_density_estimation(df_test)),
        ("normal", get_data_from_normal_distribution(df_test))
        ("chatgpt",get_data_from_csv_file('ai_generated_daata.csv')),
    ]
    
    for gen_name, generated_df in generated_df_set:
        full_name = f"{name}_{gen_name}"
        predict_by_autoencoder(df_test, name=full_name)
        predict_outlier_by_MLP(generated_df,df_test, name=full_name)
        regression_by_linear(generated_df, df_test, name=full_name)
        regression_by_polynomial(generated_df,df_test, name=full_name)
        regression_by_elastic_net(generated_df,df_test, name=full_name)
        regression_by_elastic_net_with_PCA(generated_df,df_test, name=full_name)
        regression_by_XGBoost(generated_df,df_test, name=full_name)
        regression_by_random_forest(generated_df,df_test, name=full_name)
        train_model_MLP(generated_df,df_test, name=full_name)
        plot_correlation_matrix(generated_df, title=f'Correlation Matrix for {full_name}', name = name)
        plot_generated_values_for_df(generated_df, name=full_name)


for name, df in df_set:
    df_test = df.sample(frac=0.2, random_state=42)  # Sample 20% of the data for testing
    df_train = df.drop(df_test.index)  # Remaining data for training
    predict_by_autoencoder(df_test, name=name)
    predict_outlier_by_MLP(df_train,df_test, name=name)
    regression_by_linear(df_train, df_test, name=name)
    regression_by_polynomial(df_train,df_test, name=name)
    regression_by_elastic_net(df_train,df_test, name=name)
    regression_by_elastic_net_with_PCA(df_train,df_test, name=name)
    regression_by_XGBoost(df_train,df_test, name=name)
    regression_by_random_forest(df,df, name=name)
    train_model_MLP(df_train,df_test, name=name)
    plot_correlation_matrix(df, title=f'Correlation Matrix for {name}', name = name)
    plot_generated_values_for_df(generated_df, name=name)

# chat_df = get_data_from_csv_file('ai_generated_daata.csv')
# df_test = get_data_without_columns_with_nan_as_df()
# train_model_MLP(df_train,df_test, name=name)
# regression_by_random_forest(chat_df,df_test, name="chatgpt")


