# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="436" height="698" alt="image" src="https://github.com/user-attachments/assets/98ed159c-47e9-40e6-81d2-b0138e0568b5" />



## DESIGN STEPS

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.

### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.

### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).

### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.




## PROGRAM

### Name: VARNIKA P
### Register Number: 212223240170
```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
```
```python
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information

<img width="1169" height="236" alt="image" src="https://github.com/user-attachments/assets/7ca1d86c-a024-4a66-904c-7f2e7fd9371a" />


## OUTPUT

### Confusion Matrix

<img width="693" height="550" alt="image" src="https://github.com/user-attachments/assets/ad49fbd8-7ee6-4d46-8493-2865fb17c162" />


### Classification Report

<img width="583" height="400" alt="image" src="https://github.com/user-attachments/assets/7caf1394-389d-4e63-9fd3-fed963dfc6f8" />


### New Sample Data Prediction

<img width="892" height="328" alt="image" src="https://github.com/user-attachments/assets/e7227e3e-ecbf-49b9-867a-868ac18237b7" />


## RESULT

Thus a neural network classification model for the given dataset is executed successfully.

