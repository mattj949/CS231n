import sys
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


NUM_INPUT_DAYS = 20 # Days on which to base prediction
batch_size = 50
DAYS_AHEAD_FORECAST = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add along S dimension
        # x should be (S, N, E)
        #print(x.shape)
        #print(self.pe[:x.size(0), :].shape)
        return x + self.pe[:x.size(0), :]
          

class TransAm(nn.Module):
    def __init__(self,feature_size,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=11, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size, 1) # Output being size of embedding dimension?
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



def generate_input_output(data, vix, dates, num_deltas, num_tenors):
    L = dates.shape[0]
    inout_sequences = []
    
    target_dates = []

    last_vix = 0
    for di in range(NUM_INPUT_DAYS, L + 1 - DAYS_AHEAD_FORECAST):
        input_dates = dates[di-NUM_INPUT_DAYS:di]
        target_date = dates[di - 1 + DAYS_AHEAD_FORECAST]
        
        # Build out input data array
        # SHAPE: (NUM_INPUT_DAYS, num_deltas * num_tenors)
        input_data = np.zeros((NUM_INPUT_DAYS, num_deltas * num_tenors))
        for i, date in enumerate(input_dates):
            raw_data = data.loc[date].to_numpy()
            input_data[i,:] = raw_data.flatten()
        # Not sure how aligned the dates are going to be. If missing, then use last value.
        try:
            target_value = np.asarray(vix.loc[target_date].Close)
            last_vix = target_value
        except:
            target_value = last_vix

        # Turn into pytorch tensor
        input_data = torch.from_numpy(input_data)  
        target_data = torch.from_numpy(target_value)

        # Set type as float32
        input_data = input_data.type(torch.float32)
        target_data = target_data.type(torch.float32)

        inout_sequences.append((input_data, target_data))

        target_dates.append(target_date)
        
    return inout_sequences, target_dates



def get_data():

    # PROCESS VOL SURFACE DATA
    data = pd.read_csv('data/SPX_vol_surface_FULL.csv')
    # Fill nans
    data = data.fillna(0)
    max_ivol = data.impl_volatility.max()

    unique_dates = data.date.unique()
    unique_deltas = data.delta.unique()
    unique_tenors = data.days.unique()
    num_dates = unique_dates.shape[0]
    num_deltas = unique_deltas.shape[0]
    num_tenors = unique_tenors.shape[0]

    # Shape: (num_dates, num_tenors, num_deltas)
    vol_surface_matrix = data.set_index(
        ["date", "days", "delta"]).impl_volatility.unstack().values.reshape(
        num_dates, num_tenors, num_deltas)

    vol_surface_matrix /= (255/max_ivol)
    dates = pd.to_datetime(unique_dates, format='%Y%m%d')

    midx = pd.MultiIndex.from_product([dates, unique_deltas])
    data = pd.DataFrame(index = midx, columns = unique_tenors, dtype='float')
    for i in range(len(dates)):
        data.loc[dates[i]] = vol_surface_matrix.squeeze()[i].T

    # PROCESS VIX DATA
    vix = pd.read_csv('data/VIX.csv', header=2)
    vix['Date'] = pd.to_datetime(vix['Date'])
    vix = vix.set_index('Date')

    #split = 6040
    split = 5788
    print("Start of test dataset: ", dates[split])
    train_dates = dates[:split]
    test_dates = dates[split:]
    print("Train dates shape: ", train_dates.shape)
    print("Test dates shape:" , test_dates.shape)

    # Creates a list of tuples of input data (size INPUT_DAYS) and target data (size 1)
    train_tuples, train_target_dates = generate_input_output(data, vix, train_dates, num_deltas, num_tenors)
    test_tuples, test_target_dates = generate_input_output(data, vix, test_dates, num_deltas, num_tenors)

    return train_tuples, test_tuples, num_dates, num_deltas, num_tenors, test_target_dates



def get_batch(list_of_tuples, batch_size):
    # Get a random batch of the data
    L = len(list_of_tuples)
    indices = np.random.randint(low=0, high=L, size=batch_size)
    batch_tuples = [list_of_tuples[i] for i in indices]  
    input_data = torch.stack([item[0] for item in batch_tuples], dim=0) # Get input data
    target = torch.stack([item[1] for item in batch_tuples], dim=0) # Get targets

    # N = batch size
    # S = input_window
    # T = output_window

    # Shape (N, S, E) where E is the embedding dimension == num_deltas * num_tenors
    input_data = torch.reshape(input_data, (batch_size, NUM_INPUT_DAYS, -1))

    # Shape (N, T, E) where E is the last embedding dimension == 1
    target = torch.reshape(target, (batch_size, 1, 1))

    input_data = torch.transpose(input_data, 0, 1) # Now shape (S, N, E)
    target = torch.transpose(target, 0, 1) # Now shape (T, N, E)
    return input_data, target


def train(train_data):
    model.train() # Turn on the train mode
    total_loss = 0.0
    start_time = time.time()

    L = len(train_data)
    # Determine num batches to cover whole trainin set, in probability/expectation
    num_batches = int(L / batch_size)

    for i in range(num_batches):
        data, targets = get_batch(train_data, batch_size)

        data = data.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output[-1:], targets)

        if torch.isnan(loss):
          print(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()

        # Print information 5 times per EPOCH
        interval = int(num_batches / 5)
        if i % interval == 0 and i > 0:
            cur_loss = total_loss / interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(
                    epoch, i, num_batches, scheduler.get_lr()[0],
                    elapsed * 1000 / interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, tuple_of_test_data):
    eval_model.eval() 
    with torch.no_grad():
      
      # WARNING: ONLY A SINGLE TUPLE!! ONLY A SINGLE WEEK PREDICTION!!
      data, target = tuple_of_test_data
      # Shape: (S, N, E)
      data = torch.reshape(data, (NUM_INPUT_DAYS, 1, -1))
      # Shape: (T, N, E)
      target = torch.reshape(target, (1, 1, 1))
      
      data = data.to(device)
      target = target.to(device)

      #print("Input shape: ", data.shape)
      output = eval_model(data)
      #print("Output shape: ", output.shape)
      output = output[-1:]            
      #loss = criterion(output, target).item()

      squared_error = torch.square(output.squeeze() - target.squeeze())
      absolute_error = torch.abs(output.squeeze() - target.squeeze())

      # return loss, output[-1:]
      return squared_error, absolute_error, output



train_data, val_data, num_dates, num_deltas, num_tenors, test_dates = get_data()
model = TransAm(num_deltas * num_tenors).to(device)

criterion = nn.MSELoss()
lr = 0.0005
epochs = 500
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)

# Just first out of sample train and target tuple...
val_data_CHOSEN = val_data[0]
val_losses = []

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    
    # THIS IS A VERY IMPERFECT METRIC!!!
    val_loss, absolute_error, output = evaluate(model, val_data_CHOSEN)
    
    # Calculate val loss across ENTIRE validation set
    if epoch % 8 == 0:
      num_in_val_set = len(val_data)

      squared_errors = torch.zeros(num_in_val_set)
      absolute_errors = torch.zeros(num_in_val_set)
      outputs = torch.zeros(num_in_val_set)
      truths = torch.zeros(num_in_val_set)

      for i in range(num_in_val_set):
        squared_error, absolute_error, output = evaluate(model, val_data[i])
        squared_errors[i] = squared_error
        absolute_errors[i] = absolute_error
        outputs[i] = output
        truths[i] = val_data[i][1]

      MSE = torch.mean(squared_errors)
      MAE = torch.mean(absolute_errors)
      AESTD = torch.std(absolute_errors, unbiased=True)

      R_squared = 1 - (torch.sum(torch.square(truths-outputs)) / torch.sum(torch.square(truths - torch.mean(truths))))

      val_losses.append(MSE)
      pyplot.figure(figsize=(16,10))
      pyplot.plot(val_losses)
      pyplot.title("Validation MSE loss across entire validation set")
      pyplot.show()

      print("R-squared: ", R_squared)
      print("MSE: ", MSE)
      print("MAE: ", MAE)
      print("AE STD: ", AESTD)

      residuals = outputs - truths
      pyplot.figure(figsize=(16,10))
      pyplot.title("Histogram of residuals")
      pyplot.hist(residuals, bins=20)
      pyplot.show()
    
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    scheduler.step() 


# FINAL OUTPUT
   
num_in_val_set = len(val_data)

squared_errors = torch.zeros(num_in_val_set)
absolute_errors = torch.zeros(num_in_val_set)
outputs = torch.zeros(num_in_val_set)
truths = torch.zeros(num_in_val_set)

for i in range(num_in_val_set):
  squared_error, absolute_error, output = evaluate(model, val_data[i])
  squared_errors[i] = squared_error
  absolute_errors[i] = absolute_error
  outputs[i] = output
  truths[i] = val_data[i][1]

MSE = torch.mean(squared_errors)
MAE = torch.mean(absolute_errors)
AESTD = torch.std(absolute_errors, unbiased=True)

R_squared = 1 - (torch.sum(torch.square(truths-outputs)) / torch.sum(torch.square(truths - torch.mean(truths))))
pyplot.figure(figsize=(16,10))
pyplot.plot(test_dates, truths)
pyplot.plot(test_dates, outputs)
pyplot.legend(["Realized", "Model"])
pyplot.show()


