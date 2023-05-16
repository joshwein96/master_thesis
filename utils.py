import torch
import numpy as np
import iisignature

#transform the data such that it can be used for the rnn based models
def classification_dataframe_to_lstm(dataframe):
    array_of_signatures=[]
    for i in range(dataframe.shape[0]):
        vectors=np.array(dataframe.iloc[i].to_list(), dtype=np.float32)
        tensors=torch.transpose(torch.from_numpy(vectors),1,0)
        #tensors=tensors.unsqueeze(0)
        array_of_signatures.append(tensors)
    #array_of_signatures=torch.stack(array_of_signatures,dim=0)
    #array_of_signatures=array_of_signatures.squeeze(1)
    return array_of_signatures
#time augmentation
def augment(path):
    len=path.shape[-2]
    time=np.arange(0,1,1/len)
    time=np.resize(time,(path.shape[0],path.shape[1],1))
    augmented=np.concatenate((path,time),axis=2)
    return augmented
#turn the input data stream into signature features up to level 4
def classification_dataframe_to_signature(dataframe):
    array_of_signatures=[]
    for i in range(dataframe.shape[0]):
        vectors=np.array(dataframe.iloc[i].to_list(), dtype=np.float64)
        vectors=np.transpose(vectors,(1,0))
        vectors=np.expand_dims(vectors, axis=0)
        vectors=augment(vectors)
        signature=iisignature.sig(vectors,4)
        array_of_signatures.append(signature)
    array_of_signatures=np.stack(array_of_signatures,axis=0)
    array_of_signatures=np.squeeze(array_of_signatures,axis=1)
    return array_of_signatures

def regression_dataframe_to_lstm(dataframe,window_size=10):
    values=dataframe.values
    X,y=[],[]
    for i in range(len(values)-window_size):
        feature=values[i:i+window_size]
        target=values[i+1:i+1+window_size]
        X.append(feature)
        y.append(target)
    
    return np.expand_dims(X,2), np.expand_dims(X,2)




def regression_dataframe_to_signature(dataframe, window_size=10):
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
        signature=iisignature.sig(path,4)
        array_of_signatures.append(signature)
    array_of_signatures=np.stack(array_of_signatures,axis=0)
    array_of_signatures=np.squeeze(array_of_signatures,axis=1)
    return array_of_signatures    
        