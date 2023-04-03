import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.datasets import load_airline, load_lynx, load_macroeconomic
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from skorch import NeuralNetRegressor
import iisignature
import time
start=time.time()
#max iterations
max=1500
workers=2
def dataframe_to_lstm(dataframe):
    tensors=torch.from_numpy(dataframe.values)
    if len(tensors.shape)==1:
        tensors=tensors.unsqueeze(1)
    return tensors.float()

def augment(path):
    len=path.shape[-2]
    time=np.arange(0,1,1/len)
    time=np.resize(time,(path.shape[0],path.shape[1],1))
    augmented=np.concatenate((path,time),axis=2)
    return augmented


def dataframe_to_signature(dataframe, window_size=10):
    array_of_signatures=[]
    length=dataframe.shape[0]
    vectors=np.array(dataframe)
    if len(vectors.shape)==2:
        vectors=np.expand_dims(vectors, axis=0)
    else:
        vectors=np.expand_dims(vectors, axis=(0,2))
    vectors=augment(vectors)
    for i in range(length-window_size):
        path=vectors[:,i:i+window_size,:]
        signature=iisignature.sig(path,3)
        array_of_signatures.append(signature)
    array_of_signatures=np.stack(array_of_signatures,axis=0)
    array_of_signatures=np.squeeze(array_of_signatures,axis=1)
    return array_of_signatures    
        


class LSTMPredictor(nn.Module):
    def __init__(self,n_hidden=51,channels=1):
        super(LSTMPredictor, self).__init__()
        self.n_hidden=n_hidden
        self.lstm1=nn.LSTMCell(channels,self.n_hidden)
        self.lstm2=nn.LSTMCell(self.n_hidden,self.n_hidden)
        self.linear = nn.Linear(self.n_hidden,1)

    def forward(self,x):
        outputs=[]
        n_samples = 1

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        for input_t in x.split(1,dim=0):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)



        outputs = torch.cat(outputs, dim=0)
        return outputs







loaders=[load_airline,load_lynx, load_macroeconomic]
names=['Airline', 'Lynx', 'Macroeconomic Data']

for loader, name in list(zip(loaders,names))[2:3]:
    print(name)
    if name=='Macroeconomic Data':
        y = loader()
        y_train, y_test = temporal_train_test_split(y, test_size=0.3)
        
        train_input_lstm=dataframe_to_lstm(y_train.drop('realgdp',axis=1))
        train_target_lstm=torch.from_numpy(np.array(y_train['realgdp'])).unsqueeze(1).float()
        test_input_lstm=dataframe_to_lstm(y_test.drop('realgdp',axis=1))
        test_target_lstm=torch.from_numpy(np.array(y_test['realgdp'])).unsqueeze(1).float()

        train_input_signature=dataframe_to_signature(y_train.drop('realgdp',axis=1))
        train_target_signature=np.expand_dims(np.array(y_train['realgdp'])[10:], axis=1)
        test_input_signature=dataframe_to_signature(y_test.drop('realgdp',axis=1))
        test_target_signature=np.expand_dims(np.array(y_test['realgdp'])[10:],axis=1)
    else:    
        y = loader()
        y_train, y_test = temporal_train_test_split(y, test_size=0.3)
        # plotting for visualization
    
        ###LSTM Training
        train_input_lstm=dataframe_to_lstm(y_train)[:-1]
        train_target_lstm=dataframe_to_lstm(y_train)[1:]
        test_input_lstm=dataframe_to_lstm(y_test)[:-1]
        test_target_lstm=dataframe_to_lstm(y_test)[1:]

        train_input_signature=dataframe_to_signature(y_train)
        train_target_signature=np.array(y_train)[10:]
        test_input_signature=dataframe_to_signature(y_test)
    
    no_channels=train_input_lstm.shape[1]
    model_lstm = NeuralNetRegressor(
    LSTMPredictor,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    max_epochs=max,
    lr=0.01,
    train_split=False,
    verbose=0,
    warm_start=True
    )

    params_lstm = {
    'lr': [0.001, 0.01],
    'module__n_hidden': [50, 200],
    'module__channels': [no_channels]
    }
    gs_lstm = GridSearchCV(model_lstm, params_lstm, cv=2, n_jobs=workers)
    gs_lstm.fit(train_input_lstm,train_target_lstm)
    print(gs_lstm.best_params_)
    train_pred_lstm=gs_lstm.predict(train_input_lstm)
    test_pred_lstm=gs_lstm.predict(test_input_lstm)
    ###Sig Training
   
    model_sig=MLPRegressor(max_iter=max,
                       n_iter_no_change=40,
                       warm_start=True)

    params_sig={'hidden_layer_sizes':[(50,50),(200,200)], 
                'learning_rate_init': [0.001, 0.01]}
    
    gs_signature = GridSearchCV(model_sig, params_sig, cv=2, n_jobs=workers)
    gs_signature.fit(train_input_signature,train_target_signature.ravel())
    print(gs_signature.best_params_)
    train_pred_signature=gs_signature.predict(train_input_signature)
    test_pred_signature=gs_signature.predict(test_input_signature)


    pd.plotting.register_matplotlib_converters()
    if name=='Macroeconomic Data':
        df_train_lstm=y_train['realgdp'].to_frame()
        df_train_lstm.insert(1,'Train_Predictions LSTM', train_pred_lstm)
        df_test_lstm=y_test['realgdp'].to_frame()
        df_test_lstm.insert(1,'Predictions LSTM',test_pred_lstm)

        df_train_lstm.index=df_train_lstm.index.to_timestamp()
        df_test_lstm.index=df_test_lstm.index.to_timestamp()

        df_train_signature=y_train['realgdp'][10:].to_frame()
        df_train_signature.insert(1,'Train_Predictions signature', train_pred_signature)
        df_test_signature=y_test['realgdp'][10:].to_frame()
        df_test_signature.insert(1,'Predictions signature',test_pred_signature)

        df_train_signature.index=df_train_signature.index.to_timestamp()
        df_test_signature.index=df_test_signature.index.to_timestamp()

    else:
        df_train_lstm=y_train[1:].to_frame()
        df_train_lstm.insert(1,'Train_Predictions LSTM', train_pred_lstm)
        df_test_lstm=y_test[1:].to_frame()
        df_test_lstm.insert(1,'Predictions LSTM',test_pred_lstm)

        df_train_lstm.index=df_train_lstm.index.to_timestamp()
        df_test_lstm.index=df_test_lstm.index.to_timestamp()

        df_train_signature=y_train[10:].to_frame()
        df_train_signature.insert(1,'Train_Predictions signature', train_pred_signature)
        df_test_signature=y_test[10:].to_frame()
        df_test_signature.insert(1,'Predictions signature',test_pred_signature)

        df_train_signature.index=df_train_signature.index.to_timestamp()
        df_test_signature.index=df_test_signature.index.to_timestamp()


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
    plt.plot(df_train_signature)
    plt.plot(df_test_signature)
    plt.legend(['Train', 'Train Predictions Signature', 'Test', 'Test Predictions Signature'], loc='upper left')
    plt.savefig(f"{name}_Signature.pdf")
    plt.cla()

    print((time.time()-start)/60)