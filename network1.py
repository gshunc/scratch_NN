import random

data = [
    ([5.1, 3.5, 1.4, 0.2], 0), ([4.9, 3.0, 1.4, 0.2], 0), ([4.7, 3.2, 1.3, 0.2], 0),
    ([4.6, 3.1, 1.5, 0.2], 0), ([5.0, 3.6, 1.4, 0.2], 0), ([5.4, 3.9, 1.7, 0.4], 0),
    ([4.6, 3.4, 1.4, 0.3], 0), ([5.0, 3.4, 1.5, 0.2], 0), ([4.4, 2.9, 1.4, 0.2], 0),
    ([4.9, 3.1, 1.5, 0.1], 0), ([5.4, 3.7, 1.5, 0.2], 0), ([4.8, 3.4, 1.6, 0.2], 0),
    ([4.8, 3.0, 1.4, 0.1], 0), ([4.3, 3.0, 1.1, 0.1], 0), ([5.8, 4.0, 1.2, 0.2], 0),
    ([5.7, 4.4, 1.5, 0.4], 0), ([5.4, 3.9, 1.3, 0.4], 0), ([5.1, 3.5, 1.4, 0.3], 0),
    ([5.7, 3.8, 1.7, 0.3], 0), ([5.1, 3.8, 1.5, 0.3], 0), ([5.4, 3.4, 1.7, 0.2], 0),
    ([5.1, 3.7, 1.5, 0.4], 0), ([4.6, 3.6, 1.0, 0.2], 0), ([5.1, 3.3, 1.7, 0.5], 0),
    ([4.8, 3.4, 1.9, 0.2], 0),
    ([7.0, 3.2, 4.7, 1.4], 1), ([6.4, 3.2, 4.5, 1.5], 1), ([6.9, 3.1, 4.9, 1.5], 1),
    ([5.5, 2.3, 4.0, 1.3], 1), ([6.5, 2.8, 4.6, 1.5], 1), ([5.7, 2.8, 4.5, 1.3], 1),
    ([6.3, 3.3, 4.7, 1.6], 1), ([4.9, 2.4, 3.3, 1.0], 1), ([6.6, 2.9, 4.6, 1.3], 1),
    ([5.2, 2.7, 3.9, 1.4], 1), ([5.0, 2.0, 3.5, 1.0], 1), ([5.9, 3.0, 4.2, 1.5], 1),
    ([6.0, 2.2, 4.0, 1.0], 1), ([6.1, 2.9, 4.7, 1.4], 1), ([5.6, 2.9, 3.6, 1.3], 1),
    ([6.7, 3.1, 4.4, 1.4], 1), ([5.6, 3.0, 4.5, 1.5], 1), ([5.8, 2.7, 4.1, 1.0], 1),
    ([6.2, 2.2, 4.5, 1.5], 1), ([5.6, 2.5, 3.9, 1.1], 1), ([5.9, 3.2, 4.8, 1.8], 1),
    ([6.1, 2.8, 4.0, 1.3], 1), ([6.3, 2.5, 4.9, 1.5], 1), ([6.1, 2.8, 4.7, 1.2], 1),
    ([6.4, 2.9, 4.3, 1.3], 1)
]

random.shuffle(data)

class Network:
    def __init__(self, input_size, learning_rate = 0.1):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate
    
    def relu(self, num):
        return 0 if num < 0 else num
    
    def relu_derivative(self, num):
        return 0 if num < 0 else 1
    
    def classify(self, x):
        return 1 if x >= 0.5 else 0

    def forward(self, inputs):
        self.last_input = inputs
        self.last_total = (sum(w * x for w, x in zip(self.weights, inputs)) + self.bias)
        self.last_activated = self.relu(self.last_total)
        return self.last_activated
    
    def compute_gradients(self, loss):
        d_activated = loss * self.relu_derivative(self.last_total)
        d_weights = [d_activated * w for w in self.weights]
        d_bias = d_activated
        return d_weights, d_bias
    
    def update_weights(self, gradients):
        d_weights, d_bias = gradients
        self.weights = [w - d_w*self.learning_rate for w, d_w in zip(self.weights, d_weights)]
        self.bias -= self.learning_rate * d_bias

    def train(self, inputs, target):
        prediction = self.forward(inputs)
        error = target - self.classify(prediction)
        loss = error ** 2
        gradients = self.compute_gradients(error)
        self.update_weights(gradients)
        return loss
    
def train_network(network, data, labels, epochs = 1000):
    for epoch in range(epochs):
        total_loss = 0
        for inputs, target in zip(data, labels):
            error = network.train(inputs, target)
            total_loss += error
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(data)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

X = [item[0] for item in data]
y = [item[1] for item in data]

def train_test_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

X_train, X_test, y_train, y_test = train_test_split(X, y)

network = Network(input_size=4)

train_network(network, X_train, y_train)