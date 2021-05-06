import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

if __name__ == '__main__':
    X = np.array(
        [[0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1]]
    )
    Y = np.array([[0],[1],[1],[1],[1],[1],[1],[1]])
    epochs = 100
    lr = .1

    w1 = np.full((3, 16), 1, dtype=float)
    w2 = np.full((16, 1), 1, dtype=float)

    b1 = np.full((16), -1, dtype=float)
    b2 = np.full((1), -1, dtype=float)
    truth = []

    # use 2 layers + loop over each (x, y) pairs from the dataset (ineficient)
    for _ in range(epochs):
        for (x, y) in zip(X, Y):
            x1 = sigmoid(np.dot(w1.T, x) + b1)
            x2 = sigmoid(np.dot(x1, w2) + b2)
        
            loss = y - x2
            dy = loss * sigmoid_derivative(x2)

            out = 0 if x2[0] < 0.5 else 1
            print('✅' if y[0] == out else '❌', f'y={y[0]}; out={out}; loss: %0.3f' % loss[0])
            truth.append(y[0] == out)

            dw2 = np.dot(x2, dy)
            db2 = np.sum(dy)
            dx2 = np.dot(dy, w2.T)
            dx2_s = dx2 * sigmoid_derivative(dx2)

            dw1 = np.dot(x1, dx2_s)
            db1 = np.sum(dx2_s)
            dx1 = np.dot(dx2_s, w1.T)
            dx1_s = dx1 * sigmoid_derivative(dx1)

            w1 += lr * dw1
            w2 += lr * dw2
            b1 += lr * db1
            b2 += lr * db2

    plt.plot(truth)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()