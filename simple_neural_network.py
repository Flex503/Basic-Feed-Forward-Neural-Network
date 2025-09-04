import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, output_size):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(1. / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, hidden3_size) * np.sqrt(1. / hidden2_size)
        self.b3 = np.zeros((1, hidden3_size))
        self.W4 = np.random.randn(hidden3_size, hidden4_size) * np.sqrt(1. / hidden3_size)
        self.b4 = np.zeros((1, hidden4_size))
        self.W5 = np.random.randn(hidden4_size, output_size) * np.sqrt(1. / hidden4_size)
        self.b5 = np.zeros((1, output_size))

    # Added: Static methods for sigmoid and its derivative
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward(self, x):
        self.hidden1_input = np.dot(x, self.W1) + self.b1
        self.hidden1_output = self.sigmoid(self.hidden1_input)

        self.hidden2_input = np.dot(self.hidden1_output, self.W2) + self.b2
        self.hidden2_output = self.sigmoid(self.hidden2_input)

        self.hidden3_input = np.dot(self.hidden2_output, self.W3) + self.b3
        self.hidden3_output = self.sigmoid(self.hidden3_input)

        self.hidden4_input = np.dot(self.hidden3_output, self.W4) + self.b4
        self.hidden4_output = self.sigmoid(self.hidden4_input)

        self.output = np.dot(self.hidden4_output, self.W5) + self.b5
        return self.output

    def backward(self, X, y, lr=0.1):
        err = y - self.output
        d_output = err * self.sigmoid_derivative(self.output)

        d_hidden4 = d_output.dot(self.W5.T) * self.sigmoid_derivative(self.hidden4_output)
        d_hidden3 = d_hidden4.dot(self.W4.T) * self.sigmoid_derivative(self.hidden3_output)
        d_hidden2 = d_hidden3.dot(self.W3.T) * self.sigmoid_derivative(self.hidden2_output)
        d_hidden1 = d_hidden2.dot(self.W2.T) * self.sigmoid_derivative(self.hidden1_output)

        self.W5 -= self.hidden4_output.T.dot(d_output) * lr
        self.b5 -= np.sum(d_output, axis=0, keepdims=True) * lr
        self.W4 -= self.hidden3_output.T.dot(d_hidden4) * lr
        self.b4 -= np.sum(d_hidden4, axis=0, keepdims=True) * lr
        self.W3 -= self.hidden2_output.T.dot(d_hidden3) * lr
        self.b3 -= np.sum(d_hidden3, axis=0, keepdims=True) * lr
        self.W2 -= self.hidden1_output.T.dot(d_hidden2) * lr
        self.b2 -= np.sum(d_hidden2, axis=0, keepdims=True) * lr
        self.W1 -= X.T.dot(d_hidden1) * lr
        self.b1 -= np.sum(d_hidden1, axis=0, keepdims=True) * lr
        return np.mean(np.abs(err))

    # Added: Train method
    def train(self, X, y, epochs, lr=0.1):
        print("Starting training...")
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.backward(X, y, lr)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

if __name__ == "__main__":
    # Added: Main block to run the network
    # XOR problem data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize network
    nn = SimpleNeuralNetwork(input_size=2, hidden1_size=4, hidden2_size=4, hidden3_size=4, hidden4_size=4, output_size=1)

    # Train network
    nn.train(X, y, epochs=1000, lr=0.01)

    # Test and print predictions
    predictions = nn.forward(X)
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"Input: {X[i]}, Predicted: {pred[0]:.4f}, Actual: {y[i][0]}")
