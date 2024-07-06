import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch.nn as nn
class MLPModel(nn.Module):
    
    def __init__(self,input_dim,num_classes=1):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, num_classes)
        self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(32, 32)
        # self.linear3 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear1(x)
        # out = self.relu(out)
        # out = self.linear2(out)
        # out = self.relu(out)
        # out = self.linear3(out)
        out = self.sigmoid(out)
        return out
import torch
train_data = torch.load('/home/xujiaming/xujiaming/research/ASPLOS-24/results/train_data.pt')
train_lable = torch.load('/home/xujiaming/xujiaming/research/ASPLOS-24/results/train_lable.pt')
train_data = train_data.to(torch.float)
train_lable = train_lable.reshape(-1).to(torch.float).to(train_data.device)
from torch.utils.data import TensorDataset,DataLoader,random_split
dataset = TensorDataset(train_data,train_lable)

epochs = 100

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset,val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = MLPModel(27).to(device)
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# training
loss_list = []
for epoch in range(epochs):
    f_l = 0
    for X,Y in train_loader:
        outputs = model(X).squeeze()
        l = loss(outputs, Y)
        f_l += l.item()
        loss_list.append(l.item())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {f_l:.4f}')
    model.eval()
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
