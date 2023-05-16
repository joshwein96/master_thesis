#library imports
import pandas as pd
from utils import classification_dataframe_to_signature, classification_dataframe_to_lstm
from models import LSTMClassifier, RNNClassifier
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import iisignature
import time
from skorch import NeuralNetClassifier
wall_start=time.time()
#set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#set global variable to control max number of epochs
max_epochs=20000


#dataset names
names=['BasicMotions', 'JapaneseVowels', 'ArrowHead', 'GunPoint', 'OSULeaf',
       'ACSF1', 'ItalyPowerDemand', 'StandWalkJump']
#arrays to store the results in
accuracy_lstm=[]
accuracy_rnn=[]
accuracy_mlp=[]
accuracy_gb=[]
running_time_lstm=[]
running_time_rnn=[]
running_time_mlp=[]
running_time_gb=[]
dataset=[]
# iterate through all datasets
for name in names[0:2]:
    print(name)
    dataset.append(name)
    #load dataset and create test-train split
    X, y = load_UCR_UEA_dataset(name=name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69,stratify=y)
    #prepare the train data
    train_input_lstm=classification_dataframe_to_lstm(X_train)
    train_input_lstm=pad_sequence(train_input_lstm,batch_first=True).to(device)
    no_channels=train_input_lstm.shape[2]
    no_classes=len(np.unique(y))
    le=LabelEncoder()
    le.fit(y_train)
    train_target_lstm=torch.from_numpy(le.transform(y_train))
    #prepare test data
    test_target=le.transform(y_test)
    test_input_lstm=classification_dataframe_to_lstm(X_test)
    test_input_lstm=pad_sequence(test_input_lstm,batch_first=True).to(device)

    model_lstm = NeuralNetClassifier(
    LSTMClassifier,
    device=device,
    optimizer=optim.Adam,
    max_epochs=max_epochs,
    lr=0.001,
    train_split=False,
    verbose=0,
    warm_start=False
    )

    params_lstm = {
    'module__n_hidden': [20, 100],
    'optimizer__betas':[(0.9,0.999),(0.8,0.9),(0.8,0.999),(0.9,0.9)],
    'optimizer': [optim.Adam],
    'module__input_size': [no_channels],
    'module__no_classes':[no_classes]
    }
    
    model_rnn = NeuralNetClassifier(
    RNNClassifier,
    device=device,
    optimizer=optim.Adam,
    max_epochs=max_epochs,
    lr=0.001,
    train_split=False,
    verbose=0,
    warm_start=False
    )

    params_rnn = {
    'module__n_hidden': [20, 100],
    'optimizer__betas':[(0.9,0.999),(0.8,0.9),(0.8,0.999),(0.9,0.9)],
    'optimizer': [optim.Adam],
    'module__input_size': [no_channels],
    'module__no_classes':[no_classes]
    }
    #lstm training
    start=time.time()
    gs_lstm = GridSearchCV(model_lstm, params_lstm, cv=2, n_jobs=2,verbose=True)
    gs_lstm.fit(train_input_lstm,train_target_lstm)
    running_time=time.time()-start
    running_time_lstm.append(running_time)
    train_pred_lstm=gs_lstm.predict(train_input_lstm)
    test_pred_lstm=gs_lstm.predict(test_input_lstm)
    
    print(gs_lstm.best_params_)

    accuracy_lstm.append(accuracy_score(test_target,test_pred_lstm))
    

    #rnn training
    start=time.time()
    gs_rnn = GridSearchCV(model_rnn, params_rnn, cv=2, n_jobs=2,verbose=True)
    gs_rnn.fit(train_input_lstm,train_target_lstm)
    running_time=time.time()-start
    running_time_rnn.append(running_time)
    train_pred_rnn=gs_rnn.predict(train_input_lstm)
    test_pred_rnn=gs_rnn.predict(test_input_lstm)
    
    print(gs_rnn.best_params_)

    accuracy_rnn.append(accuracy_score(test_target,test_pred_rnn))

    ### Signature based methods
    train_input_signature=classification_dataframe_to_signature(X_train)
    test_input_signature=classification_dataframe_to_signature(X_test)
    train_target=le.transform(y_train)
    # # MLP Classifier
    mlp=MLPClassifier(max_iter=max_epochs)
    params_mlp={"hidden_layer_sizes":[(20,20,20),(100,100,100)],
            "beta_1":[0.8,0.9],
            "beta_2":[0.9,0.999]}
    start=time.time()
    gs_mlp=GridSearchCV(mlp,params_mlp,cv=2, n_jobs=2,verbose=True)
    gs_mlp.fit(train_input_signature,train_target)
    running_time=time.time()-start
    running_time_mlp.append(running_time)
    print(gs_mlp.best_params_)

    y_test_pred_signature=gs_mlp.predict(test_input_signature)
    accuracy_mlp.append(accuracy_score(y_test_pred_signature,test_target))



    # # Decision tree Classifier
    gb=GradientBoostingClassifier(max_features=1600)
    params_gb={"n_estimators":[200,100],
           "learning_rate":[0.1,1],
           "max_depth":[3,5]}
    start=time.time()
    gs_gb=GridSearchCV(gb,params_gb,cv=2, n_jobs=2,verbose=True)
    gs_gb.fit(train_input_signature,train_target)
    running_time=time.time()-start
    running_time_gb.append(running_time)
    print(gs_gb.best_params_)

    y_test_pred_signature=gs_gb.predict(test_input_signature)
    accuracy_gb.append(accuracy_score(y_test_pred_signature,test_target))
    # # Plotting



#create dataframe out of accuracy scores and calculate ranks
df_accuracy = pd.DataFrame(list(zip(accuracy_lstm,accuracy_rnn,accuracy_mlp,accuracy_gb)), index=dataset,
                                            columns =["LSTM","RNN","MLP","GB"])
mean_accuracy=df_accuracy.mean(axis=0)
df_accuracy.loc['Mean Accuracy']=mean_accuracy
print(df_accuracy)
ranks = df_accuracy.rank(axis=1,ascending=False, method='min')
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

