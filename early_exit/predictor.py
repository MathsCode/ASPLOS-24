import os
import torch.nn as nn
class MLPModel(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,num_classes=1):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        # self.linear3 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = self.relu(x)
        # out = self.linear3(out)
        x = self.sigmoid(x)
        return x
import torch
model_name = 'Llama-7B'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = torch.load('/home/xujiaming/xujiaming/research/ASPLOS-24/results/'+model_name+'_feature.pt').to(device).to(torch.float)
print(train_data.shape)
train_label = torch.load('/home/xujiaming/xujiaming/research/ASPLOS-24/results/'+model_name+'_label.pt').reshape(-1).to(train_data.device).to(train_data.dtype)




from torch.utils.data import TensorDataset,DataLoader,random_split
dataset = TensorDataset(train_data,train_label)

epochs = 100

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset,val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(device)
model = MLPModel(train_data.shape[-1],4096).to(device).to(torch.float)
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# training
for epoch in range(epochs):
    for X,Y in train_loader:
        # print(X.shape)
        outputs = model(X).squeeze()
        l = loss(outputs, Y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
    # model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            val_loss += loss(outputs, labels).item()
            preds = (outputs > 0.5).float()
            val_correct += (preds == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = val_correct / len(val_dataset)
    print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

test_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze()
        preds = (outputs > 0.5).float()
        test_correct += (preds == labels).sum().item()
        
    test_accuracy = test_correct / len(test_dataset)
    print(test_accuracy)
