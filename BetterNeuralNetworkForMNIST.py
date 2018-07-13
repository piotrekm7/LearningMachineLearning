# Importing MNIST dataset
from sklearn.datasets import load_digits
digits = load_digits()

# Scaling data
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)

# Creating test and training datasets
from sklearn.model_selection import train_test_split
Y = digits.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

# Setting up the output layer
import numpy as np


def convert_y_to_vect(Y):
    Y_vect = np.zeros((len(Y), 10))
    for i in range(len(Y)):
        Y_vect[i, Y[i]] = 1
    return Y_vect


Y_V_train = convert_y_to_vect(Y_train)
Y_V_test = convert_y_to_vect(Y_test)

# Creating the neural network
nn_structure = [64, 30, 10]


def f(x):
    return 1/(1+np.exp(-x))


def f_deriv(x):
    return f(x) * (1-f(x))


import numpy.random as random


def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = random.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = random.random_sample((nn_structure[l],))
    return W, b


def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W)+1):
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l]
        h[l+1] = f(z[l+1])
    return h, z


def calculate_out_layer_delta(y, h_out, z_out):
    return -(y-h_out)*f_deriv(z_out)


def calculate_hidden_layer_delta(delta_plus_1, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_plus_1)*f_deriv(z_l)


def get_mini_batches(X, y, batch_size):
    random_idxs = random.choice(len(y), len(y), replace=False)
    X_shuffled = X[random_idxs, :]
    y_shuffled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size, :], y_shuffled[i:i+batch_size])
                    for i in range(0, len(y), batch_size)]
    return mini_batches


def train_nn(nn_structure, X, y, bs=100, iter_num=1000, alpha=0.25, lamb=0.001):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print("Starting gradient descent for {} iterations".format(iter_num))
    while cnt < iter_num:
        if cnt % 50 == 0:
            print("Iteration {} of {}".format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        mini_batches = get_mini_batches(X, y, bs)
        for mb in mini_batches:
            X_mb = mb[0]
            y_mb = mb[1]
            for i in range(len(y_mb)):
                delta = {}
                h, z = feed_forward(X_mb[i, :],  W, b)
                for l in range(len(nn_structure), 0, -1):
                    if l == len(nn_structure):
                        delta[l] = calculate_out_layer_delta(
                            y_mb[i, :], h[l], z[l])
                        avg_cost += np.linalg.norm((y_mb[i, :]-h[l]))
                    else:
                        if l > 1:
                            delta[l] = calculate_hidden_layer_delta(
                                delta[l+1], W[l], z[l])
                        tri_W[l] += np.dot(delta[l+1][:, np.newaxis],
                                           np.transpose(h[l][:, np.newaxis]))
                        tri_b[l] += delta[l+1]

            for l in range(len(nn_structure) - 1, 0, -1):
                W[l] += -alpha * (1.0/bs*tri_W[l] + lamb*W[l])
                b[l] += -alpha * (1.0/bs*tri_b[l])

        avg_cost = 1.0/m*avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


W, b, avg_cost_func = train_nn(nn_structure, X_train, Y_V_train)
import matplotlib.pyplot as plt
plt.plot(avg_cost_func)
plt.ylabel("Average cost")
plt.ylabel("iteration number")
plt.show()


def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y


from sklearn.metrics import accuracy_score
y_pred = predict_y(W, b, X_test, 3)
accuracy = accuracy_score(Y_test, y_pred)*100
print("Prediction accuracy is {}%".format(accuracy))
