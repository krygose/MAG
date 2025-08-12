import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


from mag_data_creation import get_data_as_df
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import ElasticNet
from ploting import plot_results, plot_results_log, plot_results_outlier, plot_results_x_equals_y, save_metrics_to_excel_file_results
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class MLP(nn.Module):
    def __init__(self, input_size, units_input,
                 num_hidden_layers, units_hidden, output_size=4):
        super(MLP, self).__init__()
        layers = []
        activation_fn = nn.ReLU()

        layers.append(nn.Linear(input_size, units_input))
        layers.append(activation_fn)
        layers.append(nn.BatchNorm1d(units_input))
        layers.append(nn.Dropout(0.1))

        last_size = units_input
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(last_size, units_hidden))
            layers.append(activation_fn)
            layers.append(nn.BatchNorm1d(units_hidden))
            layers.append(nn.Dropout(0.2))
            last_size = units_hidden

        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x)) 
        return x



def predict_by_autoencoder(df, name):
  class Autoencoder(nn.Module):
      def __init__(self, input_dim):
          super(Autoencoder, self).__init__()
          self.encoder = nn.Sequential(
              nn.Linear(input_dim, 64),
              nn.ReLU(True),
              nn.Linear(64, 32),
              nn.ReLU(True),
              nn.Linear(32, 2)
          )
          self.decoder = nn.Sequential(
              nn.Linear(2, 32),
              nn.ReLU(True),
              nn.Linear(32, 64),
              nn.ReLU(True),
              nn.Linear(64, input_dim),
              nn.ReLU() # Use Sigmoid if data is normalized to [0, 1]
          )

      def forward(self, x):
          x = self.encoder(x)
          x = self.decoder(x)
          return x


  df['outlier'] = ((df['U 234'] > 3) | (df['U 238'] > 3)).astype(int)


  features_for_ae = [col for col in df.columns if col != 'outlier' and col != 'U 234' and col != 'U 238' and col != 'Ra 226 [mBq/dm^3]' and col != 'Ra 228']
  X = df[features_for_ae]

  X_tensor = torch.tensor(X.values, dtype=torch.float32)

  X_normal = df[df['outlier'] == 0][features_for_ae]
  X_normal_tensor = torch.tensor(X_normal.values, dtype=torch.float32)

  # Create DataLoader
  dataset = TensorDataset(X_normal_tensor)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  # Initialize the Autoencoder model
  input_dim = X_normal_tensor.shape[1]
  model = Autoencoder(input_dim)

  # Define loss function and optimizer
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Train the Autoencoder
  num_epochs = 100
  for epoch in range(num_epochs):
      for data in dataloader:
          inputs = data[0]
          outputs = model(inputs)
          loss = criterion(outputs, inputs)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

  model.eval()  # Ensure the autoencoder model is in evaluation mode
  X_tensor_all = torch.tensor(df[features_for_ae].values, dtype=torch.float32)
  with torch.no_grad():
      encoded_df = model.encoder(X_tensor_all).numpy()

  X = encoded_df
  y = df['outlier'] # Target


  # Split data into training and testing sets
  if y.value_counts().min() < 2:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

  X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
  X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) # Add dimension for binary classification
  y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

  # Create DataLoader
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

  train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


  # Initialize the MLP Classifier model
  mlp_input_dim = X_train_tensor.shape[1]
  mlp_model = MLPClassifier(mlp_input_dim)

  # Define loss function and optimizer
  mlp_criterion = nn.BCELoss() # Binary Cross-Entropy Loss
  mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

  # Train the MLP Classifier
  num_epochs_mlp = 100
  for epoch in range(num_epochs_mlp):
      mlp_model.train()
      running_loss = 0.0
      for inputs, labels in train_dataloader:
          outputs = mlp_model(inputs)
          loss = mlp_criterion(outputs, labels)

          mlp_optimizer.zero_grad()
          loss.backward()
          mlp_optimizer.step()

          running_loss += loss.item() * inputs.size(0)

      epoch_loss = running_loss / len(train_dataset)
      # if (epoch + 1) % 10 == 0:
      #     print(f'Epoch [{epoch+1}/{num_epochs_mlp}], Loss: {epoch_loss:.4f}')

  # Evaluate the MLP Classifier
  mlp_model.eval()
  y_pred_list = []
  with torch.no_grad():
      for inputs, _ in test_dataloader:
          outputs = mlp_model(inputs)
          y_pred = (outputs > 0.5).int() # Convert probabilities to binary predictions
          y_pred_list.extend(y_pred.tolist())

  y_test_list = y_test.tolist()
  y_pred_flat = [item for sublist in y_pred_list for item in sublist]
  y_test_flat = y_test_list
  plot_results_outlier(y_test_flat, y_pred_flat, model_name='Encoder Outlier Detection', name=name)




def predict_outlier_by_MLP(df,df_test,name):
  df_test['outlier'] = ((df_test['U 234'] > 3) | (df_test['U 238'] > 3)).astype(int)
  df['outlier'] = ((df['U 234'] > 3) | (df['U 238'] > 3)).astype(int)
  features_for_ae = [col for col in df.columns if col != 'outlier' and col != 'U 234' and col != 'U 238' and col != 'Ra 226 [mBq/dm^3]' and col != 'Ra 228']    
  y = df['outlier'] 
  X = df[features_for_ae]

  X_train = X
  y_train = y
  X_test = df_test[features_for_ae]
  y_test = df_test['outlier'] 


  X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
  X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

  y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) # Add dimension for binary classification
  y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

  # Create DataLoader
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

  train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  # Define the MLP Classification Model


  # Initialize the MLP Classifier model
  mlp_input_dim = X_train_tensor.shape[1]
  mlp_model = MLPClassifier(mlp_input_dim)

  # Define loss function and optimizer
  mlp_criterion = nn.BCELoss() # Binary Cross-Entropy Loss
  mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

  # Train the MLP Classifier
  num_epochs_mlp = 100
  for epoch in range(num_epochs_mlp):
      mlp_model.train()
      running_loss = 0.0
      for inputs, labels in train_dataloader:
          outputs = mlp_model(inputs)
          loss = mlp_criterion(outputs, labels)

          mlp_optimizer.zero_grad()
          loss.backward()
          mlp_optimizer.step()

          running_loss += loss.item() * inputs.size(0)

      epoch_loss = running_loss / len(train_dataset)
      # if (epoch + 1) % 10 == 0:
      #     print(f'Epoch [{epoch+1}/{num_epochs_mlp}], Loss: {epoch_loss:.4f}')

  # Evaluate the MLP Classifier
  mlp_model.eval()
  y_pred_list = []
  with torch.no_grad():
      for inputs, _ in test_dataloader:
          outputs = mlp_model(inputs)
          y_pred = (outputs > 0.5).int() # Convert probabilities to binary predictions
          y_pred_list.extend(y_pred.tolist())

  y_test_list = y_test.tolist()
  y_pred_flat = [item for sublist in y_pred_list for item in sublist]
  y_test_flat = y_test_list
  plot_results_outlier(y_test_flat, y_pred_flat, model_name='MLP Outlier Detection', name=name)



def train_model_MLP(df, df_test, name):
    
    features = [col for col in df.columns if col not in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228','CO2', 'Eh[mV)']]
    target = [col for col in df.columns if col in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]

    X = df[features]
    y = df[target]
    X_train = X
    y_train = y
    X_test = df_test[features]
    y_test = df_test[target]


    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    param_grid = {
        'batch_size': [32, 64],
        'units_input': [16, 32],
        'num_hidden_layers': [3, 5, 10],
        'units_hidden': [32, 64],
        'learning_rate': [0.001, 0.005]
    }

    # param_grid = {
    #     'batch_size': [32],
    #     'units_input': [16],
    #     'num_hidden_layers': [2],
    #     'units_hidden': [32],
    #     'learning_rate': [0.001]
    # }

    best_mse = float('inf')
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        model = MLP(
            input_size=X_train_tensor.shape[1],
            units_input=params['units_input'],
            num_hidden_layers=params['num_hidden_layers'],
            units_hidden=params['units_hidden'],
            output_size=len(target)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        loss_history = [] # Initialize list to store loss values
        best_epoch_loss = float('inf')
        patience = 10
        trigger_times = 0
        model.train()
        for epoch in range(200):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = F.mse_loss(preds, yb)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item()) # Append loss to history
                epoch_loss += loss.item()

            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    break  # Early stop

        model.eval()
        with torch.no_grad():
            preds_test = model(X_test_tensor).numpy()

        mse_mlp = mean_squared_error(y_test, preds_test, multioutput='raw_values')
        avg_mse = np.mean(mse_mlp)

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_params = params
            best_model = model
            best_preds = preds_test.copy()
            best_loss_history = loss_history # Store loss history for the best model


    mse_best = mean_squared_error(y_test, best_preds, multioutput='raw_values')
    mae_best = mean_absolute_error(y_test, best_preds, multioutput='raw_values')
    print(f"mean squared error: {mse_best}")
    rmse_best = np.sqrt(mse_best)
    results_data = []
    for i, (m, a) in enumerate(zip(mse_best, mae_best)):
        results_data.append((m, a))
    results_data.append(name)
    results_data.append('MLP')
    print(f"results_data: {results_data}")
    #save_metrics_to_excel_file_results(results_data, file_name='results.xlsx')

    # plot_results(y_test, best_preds, model_name='MLP',name=name)
    # plot_results_log(y_test, best_preds, model_name='MLP', name=name)
    plot_results_x_equals_y(y_test, best_preds, model_name='MLP', name=name)
    return best_model


def regression_by_linear(df,df_test, name):

    features = [col for col in df.columns if col not in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]
    target = [col for col in df.columns if col in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]

    X = df[features]
    y = df[target]
    X_train = X
    y_train = y
    X_test = df_test[features]
    y_test = df_test[target]



    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')


    results_data = []
    for i, (m, a) in enumerate(zip(mse, mae)):
        results_data.append((m, a))
    results_data.append(name)
    results_data.append('Linear Regression')
    # save_metrics_to_excel_file_results(results_data, file_name='results.xlsx')
    # plot_results(y_test, y_pred, model_name='Linear Regression',name=name)
    plot_results_x_equals_y(y_test, y_pred, model_name='Linear Regression', name=name)
    return model


def regression_by_polynomial(df,df_test,name):


    features = [col for col in df.columns if col not in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]
    target = [col for col in df.columns if col in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]

    X = df[features]
    y = df[target]
    X_train = X
    y_train = y
    X_test = df_test[features]
    y_test = df_test[target]

    # Create polynomial features
    poly = PolynomialFeatures(degree=4)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    # Fit linear regression model on polynomial features
    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    # Make predictions
    y_pred_poly = model.predict(X_poly_test)

    mse_poly = mean_squared_error(y_test, y_pred_poly, multioutput='raw_values')
    mae_poly = mean_absolute_error(y_test, y_pred_poly, multioutput='raw_values')

    results_data = []
    for i, (m, a) in enumerate(zip(mse_poly, mae_poly)):
        results_data.append((m, a))
    results_data.append(name)
    results_data.append('Polynomial Regression')
    save_metrics_to_excel_file_results(results_data, file_name='results.xlsx')

    plot_results(y_test, y_pred_poly, model_name='Polynomial Regression', name=name)



def regression_by_elastic_net(df,df_test,name ):
  features = [col for col in df.columns if col not in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]
  target = [col for col in df.columns if col in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]

  X = df[features]
  y = df[target]
  X_train = X
  y_train = y
  X_test = df_test[features]
  y_test = df_test[target]


  # Split data into training and testing sets
  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Define parameter grid for Elastic Net
  # ElasticNet has two main parameters: alpha (regularization strength) and l1_ratio (mix of L1 and L2)
  param_grid_enet = {
      'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
      'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 1.0] # l1_ratio=1.0 corresponds to Lasso, 0.0 to Ridge
  }

  best_mse_enet = float('inf')
  best_params_enet = None
  best_model_enet = None

  # Grid search for Elastic Net
  for params in ParameterGrid(param_grid_enet):
      # For multiple target variables, ElasticNet handles them independently.
      # We can train one ElasticNet model directly.
      enet_model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], random_state=42)

      # Train the model
      enet_model.fit(X_train, y_train)

      # Make predictions
      y_pred_enet = enet_model.predict(X_test)

      # Evaluate performance (Mean Squared Error across all outputs)
      mse_enet = mean_squared_error(y_test, y_pred_enet, multioutput='raw_values')
      avg_mse_enet = np.mean(mse_enet)

      # Check if this model is the best so far
      if avg_mse_enet < best_mse_enet:
          best_mse_enet = avg_mse_enet
          best_params_enet = params
          best_model_enet = enet_model



  # Evaluate the best model on the test set
  y_pred_best_enet = best_model_enet.predict(X_test)
  mse_best_enet = mean_squared_error(y_test, y_pred_best_enet, multioutput='raw_values')
  mae_best_enet = mean_absolute_error(y_test, y_pred_best_enet, multioutput='raw_values')
  rmse_best_enet = np.sqrt(mae_best_enet)

  results_data = []
  for i, (m, a) in enumerate(zip(mse_best_enet, mae_best_enet)):
    results_data.append((m, a))
  results_data.append(name)
  results_data.append('Elastic Net')
#   save_metrics_to_excel_file_results(results_data, file_name='results.xlsx')

#   plot_results(y_test, y_pred_best_enet, model_name='Elastic Net',name= name)
  plot_results_x_equals_y(y_test, y_pred_best_enet, model_name='Random Forest', name=name)   


def regression_by_elastic_net_with_PCA(df,df_test, name): 
  features = [col for col in df.columns if col not in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]
  target = [col for col in df.columns if col in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]
  pca = PCA(n_components=10)

  X = df[features]
  y = df[target]
  X_train = pca.fit_transform(X)
  y_train = y
  X_test = pca.transform(df_test[features])
  y_test = df_test[target]


  # Split data into training and testing sets
  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Define parameter grid for Elastic Net
  # ElasticNet has two main parameters: alpha (regularization strength) and l1_ratio (mix of L1 and L2)
  param_grid_enet = {
      'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
      'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 1.0] # l1_ratio=1.0 corresponds to Lasso, 0.0 to Ridge
  }

  best_mse_enet = float('inf')
  best_params_enet = None
  best_model_enet = None

  # Grid search for Elastic Net
  for params in ParameterGrid(param_grid_enet):
      # For multiple target variables, ElasticNet handles them independently.
      # We can train one ElasticNet model directly.
      enet_model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], random_state=42)

      # Train the model
      enet_model.fit(X_train, y_train)

      # Make predictions
      y_pred_enet = enet_model.predict(X_test)

      # Evaluate performance (Mean Squared Error across all outputs)
      mse_enet = mean_squared_error(y_test, y_pred_enet, multioutput='raw_values')
      avg_mse_enet = np.mean(mse_enet)

      # Check if this model is the best so far
      if avg_mse_enet < best_mse_enet:
          best_mse_enet = avg_mse_enet
          best_params_enet = params
          best_model_enet = enet_model


  # Evaluate the best model on the test set
  y_pred_best_enet = best_model_enet.predict(X_test)
  mse_best_enet = mean_squared_error(y_test, y_pred_best_enet, multioutput='raw_values')
  mae_best_enet = mean_absolute_error(y_test, y_pred_best_enet, multioutput='raw_values')
  rmse_best_enet = np.sqrt(mse_best_enet)
  results_data = []
  for i, (m, a) in enumerate(zip(mse_best_enet, mae_best_enet)):
    results_data.append((m, a))
  results_data.append(name)
  results_data.append('Elastic Net with PCA')
#   save_metrics_to_excel_file_results(results_data, file_name='results.xlsx')

#   plot_results(y_test, y_pred_best_enet, model_name='Elastic Net with PCA')
  plot_results_x_equals_y(y_test, y_pred_best_enet, model_name='Elastic Net with PCA', name=name)  



def regression_by_XGBoost(df, df_test, name):
    features = [col for col in df.columns if col not in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]
    target = [col for col in df.columns if col in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]

    # Usuń niedozwolone znaki z nazw kolumn
    clean_features = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in features]
    clean_target = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in target]

    X = df[features].copy()
    y = df[target].copy()
    X.columns = clean_features
    y.columns = clean_target
    X_train = X
    y_train = y
    X_test = df_test[features].copy()
    y_test = df_test[target].copy()
    X_test.columns = clean_features
    y_test.columns = clean_target

    # Define the model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

    results_data = []
    for i, (m, a) in enumerate(zip(mse, mae)):
        results_data.append((m, a))
    results_data.append(name)
    results_data.append('XGBoost')
    # save_metrics_to_excel_file_results(results_data, file_name='results.xlsx')
    # plot_results(y_test, y_pred, model_name='XGBoost', name=name)
    plot_results_x_equals_y(y_test, y_pred, model_name='XGBoost', name=name)

    return model

def regression_by_random_forest(df,df_test, name):


  features = [col for col in df.columns if col not in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228','CO2', 'Eh[mV)']]
  target = [col for col in df.columns if col in ['U 234', 'U 238', 'Ra 226 [mBq/dm^3]', 'Ra 228']]

  X = df[features]
  y = df[target]
  X_train = X
  y_train = y
  X_test = df_test[features]
  y_test = df_test[target]

  # Define the model

  model = RandomForestRegressor(n_estimators=1000, random_state=42)

  # Train the model
  model.fit(X_train, y_train)

  # Make predictions
  y_pred = model.predict(X_test)

  mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
  mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

  results_data = []
  for i, (m, a) in enumerate(zip(mse, mae)):
    results_data.append((m, a))
  results_data.append(name)
  results_data.append('Random Forest')
#   save_metrics_to_excel_file_results(results_data, file_name='results.xlsx')

#   plot_results(y_test, y_pred, model_name='Random Forest', name= name)
  plot_results_x_equals_y(y_test, y_pred, model_name='Random Forest', name=name)

  return model