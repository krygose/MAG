import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def plot_correlation_matrix(df, title='Correlation Matrix', name=''):
    plt.figure(figsize=(16, 14))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(f'Confusion Matrix:  ({title})')
    os.makedirs('matrix_corr', exist_ok=True)
    plt.savefig(f'matrix_corr/{name}_{title}_confusion_matrix.png')
    plt.close()

def plot_histogram(df, column, title='Histogram', bins=30):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


def plot_results(y_test, best_preds, model_name='MLP', name= ''):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))
    axes = axes.flatten() # Flatten the 2x2 grid of axes to easily iterate


    for i, col in enumerate(y_test.columns):
        ax = axes[i] # Select the current axis

        ax.scatter(range(len(y_test[col])), y_test[col], color='blue', label='Actual', s=10)
        ax.scatter(range(len(best_preds[:, i])), best_preds[:, i], color='red', label='Predicted', s=10)


        # Connect corresponding actual and predicted points with vertical lines
        for j in range(len(y_test[col])):
            ax.plot([j, j], [y_test[col].iloc[j], best_preds[j, i]], color='gray', linestyle='--', linewidth=0.7, alpha=0.7)

        ax.set_title(f'{model_name}: Actual vs. Predicted Values for {col}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(col)
        ax.legend()
    for j in range(len(y_test.columns), len(axes)):
        fig.delaxes(axes[j])

    save_image_to_folder_with_name(fig, 'images - kopia', f'{name}_{model_name}_results.png')
    plt.close(fig)  # Close the figure to free up memory


def save_image_to_folder_with_name(fig, folder_path, file_name):
    """
    Save a matplotlib figure to a specified folder with a given file name.
    
    Parameters:
    fig (matplotlib.figure.Figure): The figure to save.
    folder_path (str): The path to the folder where the image will be saved.
    file_name (str): The name of the file to save the image as.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fig.savefig(os.path.join(folder_path, file_name))


def save_metrics_to_excel_file_results(metrics, file_name='results.xlsx'):
    header = ['MSE_Ra 226', 'MSA_Ra 226', 'MSE_Ra 228', 'MSA_Ra 228',
              'MSE_U 238', 'MSA_U 238', 'MSE_U 234', 'MSA_U 234', 'generation_method','ML_algorithm']

    # Spłaszcz listę krotek i dodaj stringi na końcu
    if isinstance(metrics, list) and all(isinstance(x, tuple) for x in metrics[:-2]) and isinstance(metrics[-1], str) and isinstance(metrics[-2], str):
        flat_metrics = [item for tup in metrics[:-2] for item in tup] + [metrics[-2], metrics[-1]]
        df_new = pd.DataFrame([flat_metrics], columns=header)
    elif isinstance(metrics, dict):
        df_new = pd.DataFrame([metrics], columns=header)
    else:
        df_new = pd.DataFrame(metrics, columns=header)

    if os.path.exists(file_name):
        df_existing = pd.read_excel(file_name)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(file_name, index=False)
    else:
        df_header = pd.DataFrame(columns=header)
        df_combined = pd.concat([df_header, df_new], ignore_index=True)
        df_combined.to_excel(file_name, index=False)

def plot_results_outlier(y_true, y_pred, model_name, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: MLP Outlier Detection ({name})')
    os.makedirs('images - kopia', exist_ok=True)
    plt.savefig(f'images - kopia/{name}_{model_name}_confusion_matrix.png')
    plt.close()


def plot_generated_values_for_df(generated_df, name=''):
    """
    Plot the generated values against the test data for comparison in a 2x2 grid for Ra 226, Ra 228, U234, U238.

    Parameters:
    generated_df (pd.DataFrame): The DataFrame containing generated values.
    df_test (pd.DataFrame): The DataFrame containing test data.
    name (str): Name to be used in the plot title and saved file.
    """
    columns_to_plot = ['Ra 226 [mBq/dm^3]', 'Ra 228', 'U 234', 'U 238']
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        if col in generated_df.columns:
            ax.scatter(generated_df.index, generated_df[col], label=f'Generated {col}', alpha=0.5)
            ax.set_title(f'{name} Generated data for: {col}')
            ax.set_xlabel('Index')
            ax.set_ylabel(col)
            ax.legend()

    plt.suptitle(f'Generated vs Test Data Values ({name})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs('generated_data_for_prediction', exist_ok=True)
    plt.savefig(f'generated_data_for_prediction/{name}_generated_vs_test_values.png')
    plt.ylabel('Values')
    plt.legend()
    plt.close(fig)

def plot_results_log(y_test, best_preds, model_name='MLP', name=''):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))
    axes = axes.flatten()

    for i, col in enumerate(y_test.columns):
        ax = axes[i]
        ax.scatter(range(len(y_test[col])), y_test[col], color='blue', label='Actual', s=10)
        ax.scatter(range(len(best_preds[:, i])), best_preds[:, i], color='red', label='Predicted', s=10)

        for j in range(len(y_test[col])):
            ax.plot([j, j], [y_test[col].iloc[j], best_preds[j, i]], color='gray', linestyle='--', linewidth=0.7, alpha=0.7)

        ax.set_title(f'{model_name}: Actual vs. Predicted Values for {col}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(col)
        ax.legend()
        ax.set_yscale('log')  # Set y-axis to log scale

    for j in range(len(y_test.columns), len(axes)):
        fig.delaxes(axes[j])

    plt.show()
    #save_image_to_folder_with_name(fig, 'images - kopia', f'{name}_{model_name}_results.png')
    plt.close(fig)


def plot_results_x_equals_y(y_test, best_preds, model_name='MLP', name=''):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))
    axes = axes.flatten()

    for i, col in enumerate(y_test.columns):
        ax = axes[i]
        # X: actual values, Y: predicted values
        ax.scatter(y_test[col], best_preds[:, i], color='purple', label='Predicted vs Actual', s=10)
        # Draw x=y line for reference
        min_val = min(y_test[col].min(), best_preds[:, i].min())
        max_val = max(y_test[col].max(), best_preds[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='x = y')
        ax.set_title(f'{model_name}: Predicted vs. Actual Values for {col}')
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Predicted Value')
        ax.legend()

    for j in range(len(y_test.columns), len(axes)):
        fig.delaxes(axes[j])

    plt.show()
    #save_image_to_folder_with_name(fig, 'images - kopia', f'{name}_{model_name}_results.png')
    plt.close(fig)
