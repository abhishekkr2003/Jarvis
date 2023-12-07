import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from NeuralNetwork import bag_of_words,stem,tokenize
from Brain import NeuralNet
with open('intents.json','r') as f:
    intents = json.load(f)
all_words = []
tags = []
xy = []
epoch_list = []
accuracy_list = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
       
ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
x_train = []
y_train = []
for(pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
x_train = np.array(x_train)
y_train = np.array(y_train)

num_epochs = 2500
batch_size = 8
learning_rate = 0.0001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size,x_train.shape,y_train.shape,"learning the model ....")

class chatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples
dataset = chatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

print(train_loader)

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch+1)%100==0:
        print(f'epoch {epoch+1}/{num_epochs},loss={loss.item():.6f}')
        with torch.no_grad():
            total = 0
            correct = 0
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)
                outputs = model(words)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f'Training Accuracy: {accuracy * 100:.6f}%')

            # Save accuracy values for plotting
            epoch_list.append(epoch + 1)
            accuracy_list.append(accuracy)
print(f'final loss,loss={loss.item():.6f}')
data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}
FILE = "TrainData.pth"
plt.plot(epoch_list, accuracy_list)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.show()
torch.save(data,FILE)
print(f"training complete. file saved to {FILE}")