import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self,input_size,no_classes,n_hidden=51):
        super(LSTMClassifier, self).__init__()
        self.n_hidden=n_hidden
        self.lstm=nn.LSTM(input_size,n_hidden,1,batch_first=True)
        self.linear = nn.Linear(self.n_hidden,no_classes)
        self.softmax=torch.nn.Softmax(dim=2)

    def forward(self,x):
        
        output, (hn, cn) = self.lstm(x)
        output=self.softmax(self.linear(hn))

        return output.squeeze(0)

class RNNClassifier(nn.Module):
    def __init__(self,input_size,no_classes,n_hidden=51):
        super(RNNClassifier, self).__init__()
        self.n_hidden=n_hidden
        self.rnn=nn.RNN(input_size,n_hidden,1,batch_first=True)
        self.linear = nn.Linear(self.n_hidden,no_classes)
        self.softmax=torch.nn.Softmax(dim=2)

    def forward(self,x):
        
        output, hn = self.rnn(x)
        output=self.softmax(self.linear(hn))

        return output.squeeze(0)
    
# class LSTMRegressor(nn.Module):
#     def __init__(self,n_hidden=51,channels=1):
#         super(LSTMRegressor, self).__init__()
#         self.n_hidden=n_hidden
#         self.lstm1=nn.LSTMCell(channels,self.n_hidden)
#         self.lstm2=nn.LSTMCell(self.n_hidden,self.n_hidden)
#         self.linear = nn.Linear(self.n_hidden,1)

#     def forward(self,x):
#         outputs=[]
#         n_samples = 1

#         h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(x.device)
#         c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(x.device)
#         h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(x.device)
#         c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(x.device)

#         for input_t in x.split(1,dim=0):
#             h_t, c_t = self.lstm1(input_t, (h_t, c_t))
#             h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
#             output = self.linear(h_t2)
#             outputs.append(output)



#         outputs = torch.cat(outputs, dim=0)
#         return outputs    

class LSTMRegressor(nn.Module):
    def __init__(self,n_hidden=51,input_size=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, n_hidden, num_layers=1, batch_first=True)
        self.linear = nn.Linear(n_hidden, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x=x.squeeze(2)
        return x 
    
class RNNRegressor(nn.Module):
    def __init__(self,n_hidden=51,input_size=1):
        super(RNNRegressor, self).__init__()
        self.RNN = nn.RNN(input_size, n_hidden, num_layers=1, batch_first=True)
        self.linear = nn.Linear(n_hidden, 1)
    def forward(self, x):
        x, _ = self.RNN(x)
        x = self.linear(x)
        x=x.squeeze(2)
        return x
