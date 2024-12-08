"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import math
import time


# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    std = 1.0 / math.sqrt(shape[0])
    r = minitorch.rand(shape) * std
    return minitorch.Parameter(r)


#  three linears (2-> Hidden (relu), Hidden -> Hidden (relu), Hidden -> Output (sigmoid)).
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        x = self.layer1.forward(x).relu()
        x = self.layer2.forward(x).relu()
        x = self.layer3.forward(x).sigmoid()
        return x


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        # print("Input shape: ", x.shape, "weights shape: ", self.weights.value.shape)
        output = x.view(x.shape[0], x.shape[1], 1) * self.weights.value
        # print("bias shape ", self.bias.value.shape)
        output = output.sum(1) + self.bias.value
        return output.view(x.shape[0], self.out_size)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        start_time = time.time()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        print(self.model.parameters())
        X = minitorch.tensor(data.X)
        print("input data has size ", X.shape)
        print("N =  ", data.N)

        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 10
    RATE = 0.2
    data = minitorch.datasets["Xor"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
