import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# use 1 layer + no additionnal loop, only matrix mul
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
    Y = np.array([0,1,1,1,1,1,1,1]).reshape((-1, 1))
    epochs = 100
    lr = 1

    w1 = np.random.random((3, 1))
    b1 = np.random.random((1))
    truth = []

    def log_info(x1, Y, loss):
        out = np.array([0 if x[0] < 0.5 else 1 for x in x1])
        is_correct = (Y.reshape(-1) == out).all()
        print('✅' if is_correct else '❌', f'Y={Y.reshape(-1)}; out={out}; loss: %0.3f' % loss[0])
        truth.append(is_correct)

    for _ in range(epochs):
        x1 = sigmoid(np.dot(X, w1) + b1)
    
        loss = Y - x1
        dy = loss * sigmoid_derivative(x1)
        log_info(x1, Y, loss)

        dw1 = np.dot(x1.T, dy)
        db1 = np.sum(dy)
        w1 += lr * dw1
        b1 += lr * db1

    plt.plot(truth)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()