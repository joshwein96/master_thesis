


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.datasets import load_basic_motions, load_japanese_vowels, load_arrow_head, load_gunpoint, load_osuleaf, load_acsf1, load_italy_power_demand
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import iisignature
import time
from skorch import NeuralNetClassifier
start=time.time()
def dataframe_to_lstm(dataframe):
    array_of_signatures=[]
    for i in range(dataframe.shape[0]):
        vectors=np.array(dataframe.iloc[i].to_list(), dtype=np.float32)
        tensors=torch.transpose(torch.from_numpy(vectors),1,0)
        #tensors=tensors.unsqueeze(0)
        array_of_signatures.append(tensors)
    #array_of_signatures=torch.stack(array_of_signatures,dim=0)
    #array_of_signatures=array_of_signatures.squeeze(1)
    return array_of_signatures

class LSTMPredictor(nn.Module):
    def __init__(self,input_size,n_hidden=51):
        super(LSTMPredictor, self).__init__()
        self.n_hidden=n_hidden
        self.lstm=nn.LSTM(input_size,n_hidden,1,batch_first=True)
        self.linear = nn.Linear(self.n_hidden,no_classes)
        self.softmax=torch.nn.Softmax(dim=2)

    def forward(self,x):
        
        output, (hn, cn) = self.lstm(x)
        output=self.softmax(self.linear(hn))

        return output.squeeze(0)

def augment(path):
    len=path.shape[-2]
    time=np.arange(0,1,1/len)
    time=np.resize(time,(path.shape[0],path.shape[1],1))
    augmented=np.concatenate((path,time),axis=2)
    return augmented


# def dataframe_to_signature(dataframe):
#     array_of_signatures=[]
#     for i in range(dataframe.shape[0]):
#         vectors=np.array(dataframe.iloc[i].to_list(), dtype=np.float64)
#         tensors=torch.transpose(torch.from_numpy(vectors),1,0)
#         tensors=tensors.unsqueeze(0)
#         augment=signatory.Augment(no_channels,layer_sizes=(), kernel_size=1, include_original=True, include_time=True)
#         tensors=augment(tensors)
#         signature=signatory.signature(tensors,3)
#         array_of_signatures.append(signature)
#     array_of_signatures=torch.stack(array_of_signatures,dim=0)
#     array_of_signatures=array_of_signatures.squeeze(1)
#     return array_of_signatures
def dataframe_to_signature(dataframe):
    array_of_signatures=[]
    for i in range(dataframe.shape[0]):
        vectors=np.array(dataframe.iloc[i].to_list(), dtype=np.float64)
        vectors=np.transpose(vectors,(1,0))
        vectors=np.expand_dims(vectors, axis=0)
        vectors=augment(vectors)
        signature=iisignature.sig(vectors,3)
        array_of_signatures.append(signature)
    array_of_signatures=np.stack(array_of_signatures,axis=0)
    array_of_signatures=np.squeeze(array_of_signatures,axis=1)
    return array_of_signatures

#loader functions
loader=[load_basic_motions, load_japanese_vowels, load_arrow_head, load_gunpoint, load_osuleaf,
        load_acsf1,load_italy_power_demand]
#corresponding names
names=['Basic motions', 'Japanese vowels', 'Arrow head', 'Gunpoint', 'Osuleaf',
       'Power consumption', 'Italy Power Demand']

# loads both train and test together
for function, name in list(zip(loader,names))[4:]:
    print(name)
    X, y = function()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69,stratify=y)

    train_input_lstm=dataframe_to_lstm(X_train)
    train_input_lstm=pad_sequence(train_input_lstm,batch_first=True)
    no_channels=train_input_lstm.shape[2]
    no_classes=len(np.unique(y))
    le=LabelEncoder()
    le.fit(y_train)
    train_target_lstm=torch.from_numpy(le.transform(y_train))

    test_input_lstm=dataframe_to_lstm(X_test)
    test_input_lstm=pad_sequence(test_input_lstm,batch_first=True)

    model_lstm = NeuralNetClassifier(
    LSTMPredictor,
    optimizer=optim.Adam,
    max_epochs=4000,
    lr=0.001,
    train_split=False,
    verbose=0,
    warm_start=True
    )

    params_lstm = {
    'lr': [0.001],
    'module__n_hidden': [20, 100],
    'optimizer__weight_decay':[0],
    'optimizer': [optim.Adam],
    'module__input_size': [no_channels]
    }
    gs_lstm = GridSearchCV(model_lstm, params_lstm, cv=2, n_jobs=-1)
    gs_lstm.fit(train_input_lstm,train_target_lstm)
    train_pred_lstm=gs_lstm.predict(train_input_lstm)
    test_pred_lstm=gs_lstm.predict(test_input_lstm)
    ###Sig Training
    print(gs_lstm.best_params_)

    test_target=le.transform(y_test)

    
    precision_lstm=precision_score(test_target, test_pred_lstm,average='weighted')
    accuracy_lstm=accuracy_score(test_target,test_pred_lstm)
    f1_lstm=f1_score(test_target, test_pred_lstm,  average='weighted')
    recall_lstm=recall_score(test_target, test_pred_lstm, average='weighted')
    results_lstm=np.array([precision_lstm,accuracy_lstm,f1_lstm,recall_lstm])

    ### Signature based methods
    train_input_signature=dataframe_to_signature(X_train)
    test_input_signature=dataframe_to_signature(X_test)
    train_target=le.transform(y_train)
    # # MLP Classifier


    mlp=MLPClassifier(hidden_layer_sizes=(25,25,25), max_iter=4000)
    mlp.fit(train_input_signature,train_target)

    y_test_pred_signature=mlp.predict(test_input_signature)
    precision_mlp=precision_score(y_test_pred_signature,test_target,average='weighted')
    accuracy_mlp=accuracy_score(y_test_pred_signature,test_target)
    f1_mlp=f1_score(y_test_pred_signature,test_target, average='weighted')
    recall_mlp=recall_score(y_test_pred_signature,test_target, average='weighted')
    results_mlp=np.array([precision_mlp,accuracy_mlp,f1_mlp,recall_mlp])


    # # Decision tree Classifier
    dt=DecisionTreeClassifier()
    dt.fit(train_input_signature,train_target)
    y_test_pred_signature=dt.predict(test_input_signature)
    precision_dt=precision_score(y_test_pred_signature,test_target,average='weighted')
    accuracy_dt=accuracy_score(y_test_pred_signature,test_target)
    f1_dt=f1_score(y_test_pred_signature,test_target, average='weighted')
    recall_dt=recall_score(y_test_pred_signature,test_target, average='weighted')
    results_dt=np.array([precision_dt,accuracy_dt,f1_dt,recall_dt])



    rfc=RandomForestClassifier()
    rfc.fit(train_input_signature,train_target)
    y_test_pred_signature=rfc.predict(test_input_signature)
    precision_rfc=precision_score(y_test_pred_signature,test_target,average='weighted')
    accuracy_rfc=accuracy_score(y_test_pred_signature,test_target)
    f1_rfc=f1_score(y_test_pred_signature,test_target, average='weighted')
    recall_rfc=recall_score(y_test_pred_signature,test_target, average='weighted')
    results_rfc=np.array([precision_rfc,accuracy_rfc,f1_rfc,recall_rfc])


    # # Plotting



    #create dataframe out of scores 
    df=pd.DataFrame(np.array([results_lstm,results_mlp,results_dt,results_rfc]),
                    columns=['Precision','Accuracy', 'F1-Score','Recall'],
                index=['LSTM','MLP','Decision Tree', 'Random Forest Classifier'])
    df=df.rename_axis(index='Models', columns="Scores")
    df.to_csv(f'{name}_results')
    sns.scatterplot(data=df).set_title(name)
    plt.savefig(f'{name}.pdf')
    plt.cla()

print(time.time()-start)



