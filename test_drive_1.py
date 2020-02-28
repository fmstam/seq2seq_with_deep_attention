
import numpy as np
import torch
import torch.utils.data as Data


EPOCH = 500
BATCH_SIZE = 250
DATA_SIZE = 10000
INPUT_SIZE = 1
HIDDEN_SIZE = 512
WEIGHT_SIZE = 256
LR = 0.001


import torch
import torch.nn as nn
import torch.nn.functional as F

def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

class PointerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, weight_size, is_GRU=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_size = weight_size
        self.is_GRU = is_GRU

        if self.is_GRU:
            RNN = nn.GRU
            RNNCell = nn.GRUCell
        else:
            RNN = nn.LSTM
            RNNCell = nn.LSTMCell

        self.encoder = RNN(input_size, hidden_size, batch_first=True)
        self.decoder = RNNCell(input_size, hidden_size)
        
        self.W1 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.vt = nn.Linear(weight_size, 1, bias=False)

    def forward(self, input):
        batch_size = input.shape[0]
        decoder_seq_len = input.shape[1]

        encoder_output, hc = self.encoder(input) 

        # Decoding states initialization
        hidden = encoder_output[:, -1, :] #hidden state for decoder is the last timestep's output of encoder 
        if not self.is_GRU: #For LSTM, cell state is the sencond state output
            cell = hc[1][-1, :, :]
        decoder_input = to_cuda(torch.rand(batch_size, self.input_size))  
        
        # Decoding with attention             
        probs = []
        encoder_output = encoder_output.transpose(1, 0) #Transpose the matrix for mm
        for i in range(decoder_seq_len):  
            if self.is_GRU:
                hidden = self.decoder(decoder_input, hidden) 
            else:
                hidden, decoder_hc = self.decoder(decoder_input, (hidden, cell)) 
            # Compute attention
            sum = torch.tanh(self.W1(encoder_output) + self.W2(hidden))    
            out = self.vt(sum).squeeze()        
            out = F.log_softmax(out.transpose(0, 1).contiguous(), -1)  
            probs.append(out)

        probs = torch.stack(probs, dim=1)           
        return probs



def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

def getdata(experiment=1, data_size=DATA_SIZE):
    if experiment == 1:
        high = 100
        senlen = 5
        x = np.array([np.random.choice(range(high), senlen, replace=False)
                      for _ in range(data_size)])
        y = np.argsort(x)
    elif experiment == 2:
        high = 100
        senlen = 10
        x = np.array([np.random.choice(range(high), senlen, replace=False)
                      for _ in range(data_size)])
        y = np.argsort(x)
    elif experiment == 3:
        senlen = 5
        x = np.array([np.random.random(senlen) for _ in range(data_size)])
        y = np.argsort(x)
    elif experiment == 4:
        senlen = 10
        x = np.array([np.random.random(senlen) for _ in range(data_size)])
        y = np.argsort(x)
    return x, y

def evaluate(model, X, Y):
    probs = model(X) 
    prob, indices = torch.max(probs, 2) 
    equal_cnt = sum([1 if torch.equal(index.detach(), y.detach()) else 0 for index, y in zip(indices, Y)])
    accuracy = equal_cnt/len(X)
    print('Acc: {:.2f}%'.format(accuracy*100))

#Get Dataset
x, y = getdata(experiment=2, data_size = DATA_SIZE)
x = to_cuda(torch.FloatTensor(x).unsqueeze(2))     
y = to_cuda(torch.LongTensor(y)) 
#Split Dataset
train_size = (int)(DATA_SIZE * 0.9)
train_X = x[:train_size]
train_Y = y[:train_size]
test_X = x[train_size:]
test_Y = y[train_size:]
#Build DataLoader
train_data = Data.TensorDataset(train_X, train_Y)
data_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True,
)


#Define the Model
model = PointerNetwork(INPUT_SIZE, HIDDEN_SIZE, WEIGHT_SIZE, is_GRU=False)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fun = torch.nn.CrossEntropyLoss()


#Training...
print('Training... ')
for epoch in range(EPOCH):
    for (batch_x, batch_y) in data_loader:
        probs = model(batch_x)         
        outputs = probs.view(-1, batch_x.shape[1])
        batch_y = batch_y.view(-1) 
        loss = loss_fun(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
        evaluate(model, train_X, train_Y)
#Test...    
print('Test...')
evaluate(model, test_X, test_Y)