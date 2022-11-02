# -*- coding: utf-8 -*-
"""model_process.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FUt7Sxbjy92J-KrAIuOT-AprpzGZYpLL
"""

!pip install arch
!pip install plotly
!pip install torchmetrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go 
from statsmodels.tsa.seasonal import seasonal_decompose
from arch.unitroot import ADF
from torchmetrics import SymmetricMeanAbsolutePercentageError
from google.colab import drive
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchmetrics.functional import symmetric_mean_absolute_percentage_error

pd.set_option('display.max_columns', None)
drive.mount('/content/gdrive')
path = "/content/gdrive/MyDrive/ML-Transportation-Rate/"

np.random.seed(114514)

class NN:
  '''
  Inital class for 2 hidden layers with ReLU as activation

  Steps:
  1. process = FCNN()
  2. process.clean_date(data, X_columns, dummies)
  3. process.modeling(epoch = 10)

  Attributes:
  --------------------------------------------------
  | self.device          | set cuda or cpu         |
  | self.cleaned_data    | store cleaned data      |
  | self.cost            | cost method             |
  | self.optimizer       | optimizer object        |
  | self.my_nn           | model object            |
  | self.losses          | loss of each iteration  |
  | self.loss            | loss of test set        |
  --------------------------------------------------

  Subclasses could overwrite set_cost(), set_optimizer() and model_initialize()
  to change the structure.
  '''

  def __init__(self, device_type='cuda'):
    # set device type to GPU
    self.device = torch.device(device_type)

  def clean_data(self, data, X_columns, dummies):
    # one hot code dummies
    first_data = data[X_columns + ['LINEHAUL COSTS']].copy()
    for col in dummies:
      temp_col = pd.get_dummies(first_data[col], prefix = col)
      first_data = pd.concat([first_data, temp_col],axis=1)
    first_data.drop(dummies, axis=1, inplace=True)
    first_data = first_data.astype('float64')
    self.cleaned_data = first_data.copy()
    del first_data

  # set cost function
  def set_cost(self, func = SymmetricMeanAbsolutePercentageError()):
    self.cost = func.to(self.device)

  # set optimizer
  def set_optimizer(self, optimizer=None):
    self.optimizer = torch.optim.Adam(self.my_nn.parameters(), lr = 0.001)

  # method for overwriting
  def model_initialize(self, hidden_size, x_size):
    # default to 2 hidden layers
    self.my_nn = torch.nn.Sequential(
          torch.nn.Linear(x_size, hidden_size),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_size, hidden_size),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_size, 1),
        ).to(self.device)
    
  def modeling(self, hidden_size=-1, epoch = 10):
    # default hiddden_size to data.shape[1]*2/3
    if hidden_size == -1:
      hidden_size = self.cleaned_data.shape[1]//3*2
    X_train, X_test, y_train, y_test = train_test_split(self.cleaned_data.drop('LINEHAUL COSTS',axis=1), self.cleaned_data['LINEHAUL COSTS'], test_size=0.2, random_state=42)
    
    train_features = torch.tensor(X_train.values, dtype=torch.float).to(self.device)

    train_labels = torch.tensor(y_train.to_numpy(), dtype=torch.float).to(self.device)

    test_features = torch.tensor(X_test.values, dtype=torch.float).to(self.device)
    test_labels = torch.tensor(y_test.to_numpy(), dtype=torch.float).to(self.device)

    print("train data size: ", train_features.shape)
    print("label data size: ", train_labels.shape)
    print("test data size: ", test_features.shape)
    print("test label size: ", test_labels.shape)

    # NN with 2 hidden layers
    self.model_initialize(hidden_size, train_features.shape[1])
    self.set_cost()
    self.set_optimizer()

    self.losses = []
    for i in range(epoch):
        xx = torch.tensor(train_features, dtype = torch.float, requires_grad = True).to(self.device)
        yy = torch.tensor(train_labels, dtype = torch.float, requires_grad = True).to(self.device)
        prediction = self.my_nn(xx).to(self.device)
        prediction = torch.reshape(prediction,(prediction.shape[0],)).to(self.device)
        loss = self.cost(prediction, yy)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.losses.append(loss.data.cpu().numpy())

        if i % (epoch//10)==0:
            print(i, self.losses[i-1])
    pred = self.my_nn(test_features).to(self.device)
    pred = torch.reshape(pred, (pred.shape[0], ))
    self.loss = self.cost(pred, test_labels)
    print("test:", self.loss)

ltl_process = NN()

ltl_data = pd.read_csv(path + "sample data/LTL_Cleaned_Sample_Data.csv")
X_columns = ['DEST ZIP','DEST_POSTAL_CODE', 'DEST CITY', 'ORIGIN ZIP', 'ORIGIN CITY', 'ORIG_POSTAL_CODE', 'CUSTOMER', 'ACTUAL CARRIER', 'DEST LOCATION ID','VOLUME', 'DEST STATE', 'DEST_STATE','WEIGHT', 'ORIGIN STATE', 'ORIG_STATE', 'DISTANCE', 'ORIGIN NAME', 'CONTRACT_AVG_LINEHAUL_RATE', 'CONTRACT_HIGH_LINEHAUL_RATE','CASES']
dummies = ['DEST ZIP','DEST_POSTAL_CODE', 'DEST CITY', 'ORIGIN ZIP', 'ORIGIN CITY', 'ORIG_POSTAL_CODE', 'CUSTOMER', 'ACTUAL CARRIER', 'DEST LOCATION ID', 'DEST STATE', 'DEST_STATE', 'ORIGIN STATE', 'ORIG_STATE', 'ORIGIN NAME']

ltl_process.clean_data(ltl_data, X_columns, dummies)

ltl_process.modeling(epoch = 1000)

class NN_tanh(NN):
  def model_initialize(self, hidden_size, x_size):
    self.my_nn = torch.nn.Sequential(
          torch.nn.Linear(x_size, hidden_size),
          torch.nn.Tanh(),
          torch.nn.Linear(hidden_size, hidden_size),
          torch.nn.Tanh(),
          torch.nn.Linear(hidden_size, 1),
        ).to(self.device)

ltl_tanh = NN_tanh()
ltl_tanh.clean_data(ltl_data, X_columns, dummies)
ltl_tanh.modeling(epoch = 1000)

class NN_3(NN):
  def model_initialize(self, hidden_size, x_size):
    self.my_nn = torch.nn.Sequential(
          torch.nn.Linear(x_size, hidden_size),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_size, hidden_size),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_size, hidden_size),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_size, 1)
        ).to(self.device)

ltl_3 = NN_3()
ltl_3.clean_data(ltl_data, X_columns, dummies)
ltl_3.modeling(epoch = 1000)

tl_data = pd.read_csv(path + "sample data/TL_Cleaned_Sample_Data.csv")
X_columns = ['DEST ZIP','DEST_POSTAL_CODE', 'DEST CITY', 'ORIGIN ZIP', 'ORIGIN CITY', 'ORIG_POSTAL_CODE', 'CUSTOMER', 'ACTUAL CARRIER', 'DEST LOCATION ID','VOLUME', 'DEST STATE', 'DEST_STATE','WEIGHT', 'ORIGIN STATE', 'ORIG_STATE', 'DISTANCE', 'ORIGIN NAME', 'CONTRACT_AVG_LINEHAUL_RATE', 'CONTRACT_HIGH_LINEHAUL_RATE','CASES']
dummies = ['DEST ZIP','DEST_POSTAL_CODE', 'DEST CITY', 'ORIGIN ZIP', 'ORIGIN CITY', 'ORIG_POSTAL_CODE', 'CUSTOMER', 'ACTUAL CARRIER', 'DEST LOCATION ID', 'DEST STATE', 'DEST_STATE', 'ORIGIN STATE', 'ORIG_STATE', 'ORIGIN NAME']

tl_p = NN()
tl_p.clean_data(tl_data, X_columns, dummies)
tl_p.modeling(epoch = 1000)

rail_data = pd.read_csv(path + "sample data/INTERMODAL_Cleaned_Sample_Data.csv")

rail_p = NN()
rail_p.clean_data(tl_data, X_columns, dummies)
rail_p.modeling(epoch = 1000)

class simple_NN(NN):
  def model_initialize(self, hidden_size, x_size):
    # default to 2 hidden layers
    self.my_nn = torch.nn.Sequential(
          torch.nn.Linear(x_size, hidden_size),
          torch.nn.Linear(hidden_size, hidden_size),
          torch.nn.Linear(hidden_size, 1),
        ).to(self.device)

ltl＿data.head()

ltl_simple = simple_NN()
ltl_simple.clean_data(ltl_data, X_columns, dummies)
ltl_simple.modeling(epoch = 1000)

class time_NN(NN):
  def clean_data(self, data, X_columns, dummies):
    # one hot code dummies
    first_data = data[X_columns + ['LINEHAUL COSTS']].copy()
    for col in dummies:
      temp_col = pd.get_dummies(first_data[col], prefix = col)
      first_data = pd.concat([first_data, temp_col],axis=1)
    first_data.drop(dummies, axis=1, inplace=True)
    for col in ['PU_APPT','DL_APPT']:
      temp_col = (first_data[col].astype('datetime64[D]') - first_data[col].astype('datetime64[Y]'))/ np.timedelta64(1, 'D')
      first_data.drop(col, axis=1, inplace=True)
      first_data = pd.concat([first_data, temp_col],axis=1)
    first_data = first_data.astype('float64')
    self.cleaned_data = first_data.copy()
    del first_data

ltl_time = time_NN()
ltl_time.clean_data(ltl_data, X_columns + ['PU_APPT','DL_APPT'], dummies)
ltl_time.modeling(epoch = 1000)

tl_time = time_NN()
tl_time.clean_data(tl_data, X_columns + ['PU_APPT','DL_APPT'], dummies)
tl_time.modeling(epoch = 1000)

rail_time = time_NN()
rail_time.clean_data(rail_data, X_columns + ['PU_APPT','DL_APPT'], dummies)
rail_time.modeling(epoch = 1000)