from utils import regression_dataframe_to_lstm, regression_dataframe_to_signature
from models import LSTMRegressor, RNNRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sktime.datasets import load_airline, load_lynx, load_macroeconomic
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from skorch import NeuralNetRegressor
import iisignature
import time
wall_start=time.time()
#set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#max iterations
max=20000
workers=2
windows_size=10 
        

loaders=[load_airline,load_lynx, load_macroeconomic]
names=['Airline', 'Lynx','Macroeconomic Data']
accuracy_lstm=[]
accuracy_rnn=[]
accuracy_mlp=[]
accuracy_gb=[]
running_time_lstm=[]
running_time_rnn=[]
running_time_mlp=[]
running_time_gb=[]
dataset=[]
for loader, name in list(zip(loaders,names))[2:]:
    print(name)
    dataset.append(name)
    if name=='Macroeconomic Data':
        y = loader()
        y_train, y_test = temporal_train_test_split(y, test_size=0.3)
        
        # train_input_lstm,_=regression_dataframe_to_lstm(y_train.drop('realgdp',axis=1))
        # train_input_lstm=torch.from_numpy(train_input_lstm.squeeze(2)).float().to(device)
        # train_target_lstm,_=regression_dataframe_to_lstm(y_train['realgdp'])
        # train_target_lstm=torch.from_numpy(train_target_lstm.squeeze(2)).float()
        # test_input_lstm,_=regression_dataframe_to_lstm(y_test.drop('realgdp',axis=1))
        # test_input_lstm=torch.from_numpy(test_input_lstm.squeeze(2)).float().to(device)
        # test_target_lstm,_=regression_dataframe_to_lstm(y_test['realgdp'])
        # test_target_lstm=torch.from_numpy(test_target_lstm.squeeze(2)).float()

        train_input_lstm,_=regression_dataframe_to_lstm(y_train)
        train_input_lstm=torch.from_numpy(train_input_lstm.squeeze(2)).float().to(device)
        train_target_lstm,_=regression_dataframe_to_lstm(y_train['realgdp'])
        train_target_lstm=torch.from_numpy(train_target_lstm.squeeze(2)).float()
        test_input_lstm,_=regression_dataframe_to_lstm(y_test)
        test_input_lstm=torch.from_numpy(test_input_lstm.squeeze(2)).float().to(device)
        test_target_lstm,_=regression_dataframe_to_lstm(y_test['realgdp'])
        test_target_lstm=torch.from_numpy(test_target_lstm.squeeze(2)).float()
        

        train_input_signature=regression_dataframe_to_signature(y_train)
        train_target_signature=np.expand_dims(np.array(y_train['realgdp'])[windows_size:], axis=1)
        test_input_signature=regression_dataframe_to_signature(y_test)
        test_target_signature=np.expand_dims(np.array(y_test['realgdp'])[windows_size:],axis=1)
    else:    
        y = loader()
        y_train, y_test = temporal_train_test_split(y, test_size=0.3)
        # plotting for visualization
    
        ###prepare data for recurrent models
        train_input_lstm, train_target_lstm=regression_dataframe_to_lstm(y_train,window_size=windows_size)
        train_input_lstm=torch.from_numpy(train_input_lstm).float()
        train_target_lstm=torch.from_numpy(train_target_lstm).squeeze(2).float()
        
        test_input_lstm, test_target_lstm=regression_dataframe_to_lstm(y_test,window_size=windows_size)
        test_input_lstm=torch.from_numpy(test_input_lstm).float()
        test_target_lstm=torch.from_numpy(test_target_lstm).squeeze(2).float()
        #prepare data for signature models
        train_input_signature=regression_dataframe_to_signature(y_train)
        train_target_signature=np.array(y_train)[windows_size:]
        test_input_signature=regression_dataframe_to_signature(y_test)
        test_target_signature=y_test[windows_size:]
    
    no_channels=train_input_lstm.shape[2]
    
    model_lstm = NeuralNetRegressor(
    LSTMRegressor,
    device=device,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    max_epochs=max,
    train_split=False,
    verbose=0,
    warm_start=False
    )

    params_lstm = {
    'optimizer__betas':[(0.9,0.999),(0.8,0.9),(0.8,0.999),(0.9,0.9)],
    'optimizer': [optim.Adam],
    'module__n_hidden': [20, 100],
    'module__input_size': [no_channels]
    }

    model_rnn = NeuralNetRegressor(
    RNNRegressor,
    device=device,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    max_epochs=max,
    train_split=False,
    verbose=0,
    warm_start=False
    )

    params_rnn = {
    'optimizer__betas':[(0.9,0.999),(0.8,0.9),(0.8,0.999),(0.9,0.9)],
    'optimizer': [optim.Adam],
    'module__n_hidden': [20, 100],
    'module__input_size': [no_channels]
    }
    #lstm training
    start=time.time()
    gs_lstm = GridSearchCV(model_lstm, params_lstm, cv=2, n_jobs=workers, error_score='raise',verbose=True)
    gs_lstm.fit(train_input_lstm,train_target_lstm)
    running_time=time.time()-start
    running_time_lstm.append(running_time)
    print(gs_lstm.best_params_)
    train_pred_lstm=gs_lstm.predict(train_input_lstm)[:,windows_size-1]
    test_pred_lstm=gs_lstm.predict(test_input_lstm)[:,windows_size-1]
    accuracy_lstm.append(mean_squared_error(test_target_lstm[:,windows_size-1],test_pred_lstm))
    #rnn training
    start=time.time()
    gs_rnn = GridSearchCV(model_rnn, params_rnn, cv=2, n_jobs=workers, verbose=True)
    gs_rnn.fit(train_input_lstm,train_target_lstm)
    running_time=time.time()-start
    running_time_rnn.append(running_time)
    print(gs_rnn.best_params_)
    train_pred_rnn=gs_rnn.predict(train_input_lstm)[:,windows_size-1]
    test_pred_rnn=gs_rnn.predict(test_input_lstm)[:,windows_size-1]
    accuracy_rnn.append(mean_squared_error(test_target_lstm[:,windows_size-1],test_pred_rnn))
    #mlp training
    model_sig=MLPRegressor(max_iter=max,
                       n_iter_no_change=40,
                       warm_start=False)

    params_sig={"hidden_layer_sizes":[(20,20,20),(100,100,100)], 
                "beta_1":[0.8,0.9],
                "beta_2":[0.9,0.999]}
    
    start=time.time()
    gs_signature = GridSearchCV(model_sig, params_sig, cv=2, n_jobs=workers,verbose=True)
    gs_signature.fit(train_input_signature,train_target_signature.ravel())
    running_time=time.time()-start
    running_time_mlp.append(running_time)
    print(gs_signature.best_params_)
    train_pred_signature=gs_signature.predict(train_input_signature)
    test_pred_signature=gs_signature.predict(test_input_signature)
    accuracy_mlp.append(mean_squared_error(test_target_signature,test_pred_signature))
    #gb training
    model_gb=GradientBoostingRegressor(max_features=1600)

    params_gb={"n_estimators":[200,100],
           "learning_rate":[0.1,1],
           "max_depth":[3,5]}
    start=time.time()
    gs_gb = GridSearchCV(model_gb, params_gb, cv=2, n_jobs=workers,verbose=True)
    gs_gb.fit(train_input_signature,train_target_signature.ravel())
    running_time=time.time()-start
    running_time_gb.append(running_time)
    print(gs_gb.best_params_)
    train_pred_gb=gs_gb.predict(train_input_signature)
    test_pred_gb=gs_gb.predict(test_input_signature)
    accuracy_gb.append(mean_squared_error(test_target_signature,test_pred_gb))

    pd.plotting.register_matplotlib_converters()
    if name=='Macroeconomic Data':
        #lstm
        df_train_lstm=y_train['realgdp'][windows_size:].to_frame()
        df_train_lstm.insert(1,'Train_Predictions LSTM', train_pred_lstm)
        df_test_lstm=y_test['realgdp'][windows_size:].to_frame()
        df_test_lstm.insert(1,'Predictions LSTM',test_pred_lstm)

        df_train_lstm.index=df_train_lstm.index.to_timestamp()
        df_test_lstm.index=df_test_lstm.index.to_timestamp()
        #rnn
        df_train_rnn=y_train['realgdp'][windows_size:].to_frame()
        df_train_rnn.insert(1,'Train_Predictions rnn', train_pred_rnn)
        df_test_rnn=y_test['realgdp'][windows_size:].to_frame()
        df_test_rnn.insert(1,'Predictions rnn',test_pred_rnn)

        df_train_rnn.index=df_train_rnn.index.to_timestamp()
        df_test_rnn.index=df_test_rnn.index.to_timestamp()
        #mlp
        df_train_signature=y_train['realgdp'][windows_size:].to_frame()
        df_train_signature.insert(1,'Train_Predictions MLP', train_pred_signature)
        df_test_signature=y_test['realgdp'][windows_size:].to_frame()
        df_test_signature.insert(1,'Predictions MLP',test_pred_signature)

        df_train_signature.index=df_train_signature.index.to_timestamp()
        df_test_signature.index=df_test_signature.index.to_timestamp()
        #gb
        df_train_gb=y_train['realgdp'][windows_size:].to_frame()
        df_train_gb.insert(1,'Train_Predictions MLP', train_pred_gb)
        df_test_gb=y_test['realgdp'][windows_size:].to_frame()
        df_test_gb.insert(1,'Predictions MLP',test_pred_gb)

        df_train_gb.index=df_train_gb.index.to_timestamp()
        df_test_gb.index=df_test_gb.index.to_timestamp()

    else:
        #lstm
        df_train_lstm=y_train[windows_size:].to_frame()
        df_train_lstm.insert(1,'Train_Predictions LSTM', train_pred_lstm)
        df_test_lstm=y_test[windows_size:].to_frame()
        df_test_lstm.insert(1,'Predictions LSTM',test_pred_lstm)

        df_train_lstm.index=df_train_lstm.index.to_timestamp()
        df_test_lstm.index=df_test_lstm.index.to_timestamp()
        #rnn
        df_train_rnn=y_train[windows_size:].to_frame()
        df_train_rnn.insert(1,'Train_Predictions rnn', train_pred_rnn)
        df_test_rnn=y_test[windows_size:].to_frame()
        df_test_rnn.insert(1,'Predictions rnn',test_pred_rnn)

        df_train_rnn.index=df_train_rnn.index.to_timestamp()
        df_test_rnn.index=df_test_rnn.index.to_timestamp()
        #mlp
        df_train_signature=y_train[windows_size:].to_frame()
        df_train_signature.insert(1,'Train_Predictions signature', train_pred_signature)
        df_test_signature=y_test[windows_size:].to_frame()
        df_test_signature.insert(1,'Predictions signature',test_pred_signature)

        df_train_signature.index=df_train_signature.index.to_timestamp()
        df_test_signature.index=df_test_signature.index.to_timestamp()
        #gb
        df_train_gb=y_train[windows_size:].to_frame()
        df_train_gb.insert(1,'Train_Predictions gb', train_pred_gb)
        df_test_gb=y_test[windows_size:].to_frame()
        df_test_gb.insert(1,'Predictions gb',test_pred_gb)

        df_train_gb.index=df_train_gb.index.to_timestamp()
        df_test_gb.index=df_test_gb.index.to_timestamp()
    
    plt.figure(figsize=(16,8))
    plt.title(f'{name}')
    plt.xlabel('Date')
    plt.plot(df_train_lstm)
    plt.plot(df_test_lstm)
    plt.legend(['Train', 'Train Predictions LSTM', 'Test', 'Test Predictions LSTM'], loc='upper left')
    plt.savefig(f"{name}_LSTM.pdf")
    plt.cla()

    plt.figure(figsize=(16,8))
    plt.title(f'{name}')
    plt.xlabel('Date')
    plt.plot(df_train_rnn)
    plt.plot(df_test_rnn)
    plt.legend(['Train', 'Train Predictions RNN', 'Test', 'Test Predictions RNN'], loc='upper left')
    plt.savefig(f"{name}_RNN.pdf")
    plt.cla()

    plt.figure(figsize=(16,8))
    plt.title(f'{name}')
    plt.xlabel('Date')
    plt.plot(df_train_signature)
    plt.plot(df_test_signature)
    plt.legend(['Train', 'Train Predictions MLP', 'Test', 'Test Predictions MLP'], loc='upper left')
    plt.savefig(f"{name}_MLP.pdf")
    plt.cla()

    plt.figure(figsize=(16,8))
    plt.title(f'{name}')
    plt.xlabel('Date')
    plt.plot(df_train_gb)
    plt.plot(df_test_gb)
    plt.legend(['Train', 'Train Predictions GB', 'Test', 'Test Predictions GB'], loc='upper left')
    plt.savefig(f"{name}_GB.pdf")
    plt.cla()
    

df_accuracy = pd.DataFrame(list(zip(accuracy_lstm,accuracy_rnn,accuracy_mlp,accuracy_gb)), index=dataset,
                                            columns =["LSTM","RNN","MLP","GB"])
mean_accuracy=df_accuracy.mean(axis=0)
df_accuracy.loc['Mean Loss']=mean_accuracy
print(df_accuracy)
ranks = df_accuracy.rank(axis=1,ascending=True, method='min')
mean_rank = ranks.mean(axis=0)
ranks.loc['Mean Rank'] = mean_rank
print(ranks)
#create dataframe out of running times and calculate ranks
df_time = pd.DataFrame(list(zip(running_time_lstm,running_time_rnn,running_time_mlp,running_time_gb)), index=dataset,
                                            columns =["LSTM","RNN","MLP","GB"])
print(df_time)
ranks_time=df_time.rank(axis=1,ascending=True, method='min')
mean_rank = ranks_time.mean(axis=0)
ranks_time.loc['Mean Rank'] = mean_rank
print(ranks_time)
#time end of script
wall_end=time.time()
print(f'The wall clock time amounts to {(wall_end-wall_start)/60} minutes')