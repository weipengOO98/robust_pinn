import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import random
from poisson_gen_data import get_noise_data, get_truth
from parser_pinn import get_parser
import pathlib
import pickle
from logger import logger
import os
import sys
import silence_tensorflow.auto
import time

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

parser_PINN = get_parser()
args = parser_PINN.parse_args()
path = pathlib.Path(args.save_path)
path.mkdir(exist_ok=True, parents=True)
for key, val in vars(args).items():
    print(f"{key} = {val}")
with open(path.joinpath('config'), 'wt') as f:
    f.writelines([f"{key} = {val}\n" for key, val in vars(args).items()])
adam_iter: int = int(args.adam_iter)
bfgs_iter: int = int(args.bfgs_iter)
verbose: bool = bool(args.verbose)
repeat: int = int(args.repeat)
start_epoch: int = int(args.start_epoch)
Nf: int = int(args.Nf)
num_neurons: int = int(args.num_neurons)
num_layers: int = int(args.num_layers)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, X_f, u, layers, ExistModel=0, uvDir='', loss_type='square'):
        self.count = 0

        self.lb = -np.pi
        self.ub = np.pi

        self.xd = X
        self.ud = u
        self.x_f = X_f[:, 0:1]

        # Initialize NNs
        self.layers = layers

        self.loss_rec = []
        self.loss_data_rec = []
        self.loss_pde_rec = []
        self.error_u_rec = []

        if ExistModel == 0:
            self.weights, self.biases = self.initialize_NN(self.layers)
        else:
            print("Loading uv NN ...")
            self.weights, self.biases = self.load_NN(uvDir, self.layers)

        # tf Placeholders

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.xd_tf = tf.placeholder(tf.float32, shape=[None, self.xd.shape[1]])
        self.ud_tf = tf.placeholder(tf.float32, shape=[None, self.ud.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.weight_tf = tf.placeholder(tf.float32, shape=[1])

        # tf Graphs
        self.ud_pred = self.net_u(self.xd_tf)
        self.u_pred = self.net_u(self.x_f_tf)
        self.f_u_pred = self.net_f_u(self.x_f_tf)
        self.loss_pde = tf.reduce_mean(tf.square(self.f_u_pred))

        # Loss
        if loss_type == 'square':
            self.loss_data = tf.reduce_mean(tf.square(self.ud_pred - self.ud_tf))
        elif loss_type == 'l1':
            self.loss_data = tf.reduce_mean(tf.abs(self.ud_pred - self.ud_tf))
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented.')
        self.loss = self.weight_tf * self.loss_pde + self.loss_data
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': bfgs_iter,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def save_NN(self, fileDir):

        weights = self.sess.run(self.weights)
        biases = self.sess.run(self.biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([weights, biases], f)
            print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            weights, biases = pickle.load(f)
            assert num_layers == (len(weights) + 1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(weights[num], dtype=tf.float32)
                b = tf.Variable(biases[num], dtype=tf.float32)
                weights.append(W)
                biases.append(b)
                print(" - Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        # H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x):
        X = tf.concat([x, ], 1)
        u = self.neural_net(X, self.weights, self.biases)

        return u

    def net_f_u(self, x):
        u = self.net_u(x)

        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        f_u = u_xx + tf.sin(4 * x) * 4 ** 2

        return f_u

    def callback(self, loss_value, loss_data, loss_pde):
        self.count = self.count + 1
        self.loss_rec.append(loss_value)
        self.loss_data_rec.append(loss_data)
        self.loss_pde_rec.append(loss_pde)
        if self.count % 10 == 0:
            error_u = self.predict_error()
            self.error_u_rec.append(error_u)
            print('It: %d, Loss: %.3e, Loss pde: %.3e,  Loss d: %.3e' % (self.count, loss_value, loss_pde, loss_data))

    def train(self, adam_iter, learning_rate, weight):

        tf_dict = {self.xd_tf: self.xd,
                   self.ud_tf: self.ud,
                   self.x_f_tf: self.x_f,
                   self.learning_rate: learning_rate,
                   self.weight_tf: weight}

        for it in range(adam_iter):
            loss_value, loss_data, loss_pde, _ = self.sess.run(
                [self.loss, self.loss_data, self.loss_pde, self.train_op_Adam], tf_dict)
            self.callback(loss_value, loss_data, loss_pde)
        self.train_bfgs(weight=np.array(weight))
        tf_dict[self.learning_rate] = 1e-20

        for it in range(100):
            loss_value, loss_data, loss_pde, _ = self.sess.run(
                [self.loss, self.loss_data, self.loss_pde, self.train_op_Adam], tf_dict)
            self.callback(loss_value, loss_data, loss_pde)
        return self.error_u_rec

    def train_bfgs(self, weight):

        tf_dict = {self.xd_tf: self.xd,
                   self.ud_tf: self.ud,
                   self.x_f_tf: self.x_f,
                   self.weight_tf: weight}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss_data, self.loss_pde],
                                loss_callback=self.callback)

    def predict(self, X_star):

        tf_dict = {self.x_f_tf: X_star[:, 0:1]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        tf_dict = {self.x_f_tf: X_star[:, 0:1]}
        return u_star

    def predict_error(self):
        x, u = get_truth(10000, full=True)
        u_pred = self.predict(x)
        error_u = np.linalg.norm(u - u_pred, 2) / np.linalg.norm(u, 2)
        return error_u
    #
    def plot_result(self, filename):
        X_full, u_full = get_truth(N=1000, full=True)
        u_pred = self.predict(X_full)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
        fig.set_tight_layout(True)
        ax.plot(X_full.ravel(), u_pred.ravel())
        ax.plot(X_full.ravel(), u_full.ravel(), '--')
        ax.scatter(self.xd, self.ud, s=4)
        ax.set_ylim([-1., 3])
        plt.savefig(filename)
        plt.close()

def run_experiment(epoch_num, noise_type, noise, loss_type, N, weight, _data=[], abnormal_size=0):
    layers = [1] + [num_neurons] * num_layers + [1]
    # Doman bounds
    lb = np.array([-np.pi, ])
    rb = np.array([np.pi, ])
    ###########################

    if len(_data) == 0:
        X_u_train, u_train = get_noise_data(N, noise_type=noise_type, sigma=noise, size=abnormal_size)
        if Nf == 0:
            X_f_train = X_u_train
        else:
            X_f_train = lb + (rb - lb) * lhs(1, Nf)
            print(X_u_train.shape, X_f_train.shape)
            X_f_train = np.concatenate([X_u_train, X_f_train], axis=0)
        _data.append((X_u_train, X_f_train, u_train))
    X_u_train, X_f_train, u_train = _data[0]

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        model = PhysicsInformedNN(X_u_train, X_f_train, u_train, layers,
                                  loss_type=loss_type)
        model.train(adam_iter=adam_iter, learning_rate=1e-3, weight=weight)
        model.save_NN(
            path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.pickle'))
        error_u = model.predict_error()
        model.plot_result(
            path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.png'))
        with open(path.joinpath('result.csv'), 'a+') as f:
            f.write(f"{epoch_num},{loss_type},{N},{noise_type},{noise},{abnormal_size},{weight},{error_u}\n")
        logger.info(f"Eu: {error_u * 100:.3f}%")


def get_last_idx(filename):
    if not os.path.exists(filename):
        return -1
    with open(filename, "r") as f1:
        last_idx = int(f1.readlines()[-1].strip().split(',')[0])
        return last_idx
