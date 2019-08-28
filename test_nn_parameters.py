import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
# models.AlexNet


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        # self.fc2 = nn.Linear(3, 1)


    def forward(self, x):
        self.fc1 = nn.Linear(3, 2)

        # print("fc1.weight:", self.fc1.weight)
        # print("fc1.bias:", self.fc1.bias)
        # print(self.fc1.parameters())
        # print(self.parameters())
        x = self.fc1(x)
        return x


input_data = torch.randn(2,3)
print("input_data:", input_data)
net = TestModel()
cirterion = nn.CrossEntropyLoss()
# net.parameters.add([])

# print(list(net.parameters()))
# print(net)
y = net(input_data)


# print(y)

# print(y.size)
print("fc1.weight behind opt:", net.fc1.weight)
target = torch.tensor([1, 0])



print("pred::", y)




loss = cirterion(y, target)
loss.backward()

optimizer = optim.SGD(net.fc1.parameters(), lr=1e-5)
optimizer.step()

print(net.fc1.weight)