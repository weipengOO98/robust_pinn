import tensorflow as tf
import numpy as np
from parser_pinn import get_parser
from pyDOE import lhs
import pathlib
from wave_gen_data import get_noise_data, get_truth
import matplotlib.pyplot as plt
from logger import logger
import pickle
import os
import silence_tensorflow.auto
import sys
import time
import random

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
Nf: int = int(args.Nf)
verbose: bool = bool(args.verbose)
repeat: int = int(args.repeat)
num_neurons: int = int(args.num_neurons)
num_layers: int = int(args.num_layers)
start_epoch: int = int(args.start_epoch)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, Xf, u, layers, lb, ub, ExistModel=0, uvDir='', loss_type='square'):

        self.count = 0

        self.lb = lb
        self.ub = ub

        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.xf = Xf[:, 0:1]
        self.tf = Xf[:, 1:2]
        self.u = u

        self.uv_layers = layers

        self.loss_rec = []
        self.loss_data_rec = []
        self.loss_pde_rec = []
        self.error_u_rec = []
        self.error_c_rec = []

        # Initialize NNs
        if ExistModel == 0:
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
            self.c = tf.Variable([1.0], dtype=tf.float32)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # Initialize parameters

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.xf_tf = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]])
        self.tf_tf = tf.placeholder(tf.float32, shape=[None, self.tf.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.xf_tf, self.tf_tf)

        self.weight_tf = tf.placeholder(tf.float32, shape=[1])

        self.loss_pde = tf.reduce_mean(tf.square(self.f_pred))
        if loss_type == 'square':
            self.loss_data = tf.reduce_mean(tf.square(self.u_pred - self.u_tf))
        elif loss_type == 'l1':
            self.loss_data = tf.reduce_mean(tf.abs(self.u_pred - self.u_tf))
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented.')

        self.loss = self.weight_tf * self.loss_pde + self.loss_data
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': bfgs_iter,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, use_locking=True)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

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

        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)
        c = self.sess.run(self.c)

        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases, c], f)
            print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases, c = pickle.load(f)
            assert num_layers == (len(uv_weights) + 1)
            self.c = tf.Variable(c, dtype=tf.float32)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float32)
                b = tf.Variable(uv_biases[num], dtype=tf.float32)
                weights.append(W)
                biases.append(b)
                print(" - Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.uv_weights, self.uv_biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_tt - self.c * u_xx
        return f

    def callback(self, loss_value, loss_data, loss_pde):
        self.count = self.count + 1
        self.loss_rec.append(loss_value)
        self.loss_data_rec.append(loss_data)
        self.loss_pde_rec.append(loss_pde)
        if self.count % 10 == 0:
            error_u, error_c = self.predict_error()
            self.error_u_rec.append(error_u)
            self.error_c_rec.append(error_c)
            print('It: %d, Loss: %.3e, Loss pde: %.3e,  Loss d: %.3e, error c: %.3e' % (
            self.count, loss_value, loss_pde, loss_data, error_c))

    def train(self, adam_iter, learning_rate, weight):
        tf_dict = {self.x_tf: self.x,
                   self.t_tf: self.t,
                   self.u_tf: self.u,
                   self.xf_tf: self.xf,
                   self.tf_tf: self.tf,
                   self.learning_rate: learning_rate,
                   self.weight_tf: weight}

        for it in range(adam_iter):
            loss_value, loss_data, loss_pde, _ = self.sess.run(
                [self.loss, self.loss_data, self.loss_pde, self.train_op_Adam], tf_dict)
            self.callback(loss_value, loss_data, loss_pde)
        self.train_bfgs(weight=np.array(weight))
        tf_dict[self.learning_rate] = 1e-10

        # for plotting
        for it in range(100):
            loss_value, loss_data, loss_pde, _ = self.sess.run(
                [self.loss, self.loss_data, self.loss_pde, self.train_op_Adam], tf_dict)
            self.callback(loss_value, loss_data, loss_pde)
        return self.error_u_rec, self.error_c_rec

    def train_bfgs(self, weight):
        tf_dict = {self.x_tf: self.x,
                   self.t_tf: self.t,
                   self.u_tf: self.u,
                   self.xf_tf: self.xf,
                   self.tf_tf: self.tf,
                   self.weight_tf: weight}
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss_data, self.loss_pde],
                                loss_callback=self.callback)

    def predict(self, X_star):
        tf_dict = {self.x_tf: X_star[:, 0:1],
                   self.t_tf: X_star[:, 1:2],
                   self.xf_tf: X_star[:, 0:1],
                   self.tf_tf: X_star[:, 1:2], }

        u_star = self.sess.run(self.u_pred, tf_dict)

        return u_star

    def predict_error(self):
        x, u = get_truth()
        u_pred = self.predict(x)
        error_u = np.linalg.norm(u - u_pred, 2) / np.linalg.norm(u, 2)
        lambda_c = self.sess.run(self.c)
        error_c = (np.abs(lambda_c - 1.56 ** 2) / 1.56 ** 2)[0]
        return error_u, error_c

    def plot_result(self, filename):
        _shape = (401, 201)
        X_star, u_star = get_truth()
        u_pred = self.predict(X_star)
        x = X_star[:, 0].reshape(*_shape)
        t = X_star[:, 1].reshape(*_shape)
        print(X_star.shape, '?')
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), dpi=200)
        fig.set_tight_layout(True)
        ax[0].pcolormesh(t.ravel().reshape(*_shape), x.ravel().reshape(*_shape), u_pred.ravel().reshape(*_shape),
                         vmin=-1, vmax=1.)
        ax[1].scatter(self.t, self.x, zorder=100)
        ax[1].pcolormesh(t.ravel().reshape(*_shape), x.ravel().reshape(*_shape), u_star.ravel().reshape(*_shape),
                         vmin=-1, vmax=1.)
        ax[2].pcolormesh(t.ravel().reshape(*_shape), x.ravel().reshape(*_shape),
                         u_pred.ravel().reshape(*_shape) - u_star.ravel().reshape(*_shape), vmin=-0.1, vmax=0.1,
                         cmap='bwr')
        plt.savefig(filename)
        plt.close()


def run_experiment(epoch_num, noise_type, noise, loss_type, N, weight, _data=[], abnormal_size=0):
    layers = np.concatenate([[2], num_neurons * np.ones(num_layers), [1]]).astype(int).tolist()
    lb = np.array([0, 0])
    ub = np.array([1., 2.]) * np.pi

    if len(_data) == 0:
        X_u_train, u_train = get_noise_data(N, noise_type, noise, size=abnormal_size)
        if Nf == 0:
            X_f_train = X_u_train
        else:
            X_f_train = lb + (ub - lb) * lhs(1, Nf)
            X_f_train = np.concatenate([X_u_train, X_f_train], axis=0)
        _data.append((X_u_train, X_f_train, u_train))
    X_u_train, X_f_train, u_train = _data[0]

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        model = PhysicsInformedNN(X_u_train, X_f_train, u_train, layers, lb, ub,
                                  loss_type=loss_type)
        model.train(adam_iter=adam_iter, learning_rate=1e-3, weight=weight)
        model.save_NN(
            path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.pickle'))
        error_u, error_c = model.predict_error()
        model.plot_result(
            path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.png'))
        with open(path.joinpath('result.csv'), 'a+') as f:
            f.write(f"{epoch_num},{loss_type},{N},{noise_type},{noise},{abnormal_size},{weight},{error_u},{error_c}\n")
        logger.info(f"EC: {error_c * 100:.3f}%")


def get_last_idx(filename):
    if not os.path.exists(filename):
        return -1
    with open(filename, "r") as f1:
        last_idx = int(f1.readlines()[-1].strip().split(',')[0])
        return last_idx
