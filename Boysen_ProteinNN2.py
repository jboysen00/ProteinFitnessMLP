# Protein Fitness Prediction with Language Model-Based Deep Neural Network
# Joanne Boysen
# Luo Lab

# Imports
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.optim import SGD
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

#from sklearn.model_selection import train_test_split
import torch
import esm
import os
import argparse
import random
import pandas as pd
import numpy as np

def load_tensors():
    mutations = []
    with open('mutations_out.txt', 'r') as f:
        for line in f:
            vals = line.rstrip()
            mutations.append(vals)

    dms_scores = []
    with open('dms_out.txt', 'r') as f2:
        for line in f2:
            vals = float(line.rstrip())
            dms_scores.append(vals)

    #mutation_tensors = torch.empty(size=(len(mutations), 320))
    # Load in tensors from pickles.t
    tensor_list = []
    for file in os.listdir('Pickles/'):
        path = 'Pickles/' + file
        tensor_batch = torch.load(path)
        tensor_list.append(tensor_batch)
    mutation_tensors = torch.cat(tensor_list, dim=0)

    return mutation_tensors, mutations, dms_scores

mutation_tensors, mutations, dms_scores = load_tensors()



class ESMDataset(Dataset):
  '''
  #Prepare the ESM dataset for regression
  '''

  def __init__(self, X, y):
    #if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      # scale_data=True
    #if scale_data:
     #   X = StandardScaler().fit_transform(X)
    self.X = X
    # unsqueeze so that input and target have same dimensions
    self.y = torch.Tensor(y).unsqueeze(1)
    

  def __len__(self):
      return len(self.X)    # self.X.size(dim=0)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

#import pdb; pdb.set_trace()
dataset = ESMDataset(X=mutation_tensors, y=dms_scores[0:mutation_tensors.size()[0]])
train, test = random_split(dataset, [0.8, 0.2])

# create data loader
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=32, shuffle=False)


# model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.fc1 = nn.Linear(self.n_inputs, 200)
        self.fc2 = nn.Linear(200, 75)
        self.fc3 = nn.Linear(75, 10)
        self.output = nn.Linear(10,1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
 
    # forward propagate input
    def forward(self, X):
        # Input to first hidden layer
        hidden1 = self.fc1(X)
        relu1 = self.relu1(hidden1)

        # Second hidden layer
        hidden2 = self.fc2(relu1)
        relu2 = self.relu2(hidden2)

        # Third hidden layer to output
        hidden3 = self.fc3(relu2)
        relu3 = self.relu3(hidden3)

        output = self.output(relu3)
        return output

print('Training model...')
# Train Model
model = MLP(mutation_tensors.size()[1])
criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.007, momentum=0.95)

train_predictions = []
train_actuals = []
# enumerate epochs
for epoch in range(100):
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_dl):

        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        # append to list to keep track of Training MSE
        if epoch == 99:
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            actual = actual.reshape((len(actual), 1))
            train_predictions.append(yhat)
            train_actuals.append(actual)
train_predictions, train_actuals = np.vstack(train_predictions), np.vstack(train_actuals)



print('Evaluating model...')
test_predictions = []
test_actuals = []
for i, (inputs, targets) in enumerate(test_dl):
    # evaluate the model on the test set
    yhat = model(inputs)
     # retrieve numpy array
    yhat = yhat.detach().numpy()
    actual = targets.numpy()
    actual = actual.reshape((len(actual), 1))
    # store
    test_predictions.append(yhat)
    test_actuals.append(actual)
test_predictions, test_actuals = np.vstack(test_predictions), np.vstack(test_actuals)
# calculate mse
mse = mean_squared_error(test_actuals, test_predictions)



print('Test MSE: %.3f, Test RMSE: %.3f' % (mse, np.sqrt(mse)))
print(f'Train MSE: {mean_squared_error(train_actuals, train_predictions)}')
print('Test set coefficient of determination: %.2f' % r2_score(test_actuals, test_predictions))
print(f'Test Predicted values: \n{test_predictions[0:10]} \n{test_predictions[100:110]}')
print(f'Test Actual values: \n{test_actuals[0:10]} \n{test_actuals[100:110]}')
