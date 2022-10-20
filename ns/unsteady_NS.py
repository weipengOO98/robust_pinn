"""
@author: Maziar Raissi
"""

import sys

sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gen_ns_data import get_noise_data, get_truth
import random
from parser_pinn import get_parser
import pathlib
import pickle
import os
import time
import silence_tensorflow.auto

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

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
num_neurons: int = int(args.num_neurons)
num_layers: int = int(args.num_layers)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, t, u, v, layers, ExistModel=0, uvDir='', loss_type='square'):
        self.count = 0
        X = np.concatenate([x, y, t], 1)

        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X = X

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.t = X[:, 2:3]

        self.u = u
        self.v = v

        self.loss_rec = []
        self.loss_pde_rec = []
        self.loss_data_rec = []
        self.error_rec = []
        self.error_max = []
        self.error_p = []
        self.error_lambda_1 = []
        self.error_lambda_2 = []

        self.layers = layers
        self.weight_tf = tf.placeholder(tf.float32, shape=[1])
        # Initialize NN
        if ExistModel == 0:
            self.uv_weights, self.uv_biases = self.initialize_NN(layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])

        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf,
                                                                                          self.t_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_u_pred)) + \
                      tf.reduce_mean(tf.square(self.f_v_pred))

        if loss_type == 'square':
            self.loss_DATA = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) \
                             + tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
        elif loss_type == 'l1':
            self.loss_DATA = tf.reduce_mean(tf.abs(self.u_tf - self.u_pred)) \
                             + tf.reduce_mean(tf.abs(self.v_tf - self.v_pred))
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented.')
        self.loss = self.weight_tf * self.loss_f + self.loss_DATA

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': bfgs_iter,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
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

        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights) + 1)

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

    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        psi_and_p = self.neural_net(tf.concat([x, y, t], 1), self.uv_weights, self.uv_biases)
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def callback(self, loss_total, loss_data, loss_pde, lambda_1, lambda_2):
        self.count = self.count + 1
        self.loss_rec.append(loss_total)
        self.loss_data_rec.append(loss_data)
        self.loss_pde_rec.append(loss_pde)
        self.error_lambda_1.append(np.abs((lambda_1 - 1.)))
        self.error_lambda_2.append(np.abs((lambda_2 - 0.01) / 0.01))
        if self.count % 10 == 0:
            _, _, error_rel, error_max, error_p = self.predict_error()
            self.error_rec.append(error_rel)
            self.error_max.append(error_max)
            self.error_p.append(error_p)
            print('It: %d, Loss: %.3e, Loss pde: %.3e, Loss d: %.3e, l1: %.5e, l2: %.5e' % (
                self.count, loss_total, loss_pde, loss_data, lambda_1, lambda_2))

    def train(self, adam_iter, learning_rate, weight):

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v,
                   self.learning_rate: learning_rate,
                   self.weight_tf: weight}

        for it in range(adam_iter):
            loss_total, loss_data, loss_pde, lambda_1, lambda_2, _ = self.sess.run(
                [self.loss, self.loss_DATA, self.loss_f, self.lambda_1, self.lambda_2, self.train_op_Adam], tf_dict)
            self.callback(loss_total, loss_data, loss_pde, lambda_1, lambda_2)
        self.train_bfgs(weight=np.array(weight))

        tf_dict[self.learning_rate] = 1e-20
        for it in range(100):
            loss_total, loss_data, loss_pde, lambda_1, lambda_2, _ = self.sess.run(
                [self.loss, self.loss_DATA, self.loss_f, self.lambda_1, self.lambda_2, self.train_op_Adam], tf_dict)
            self.callback(loss_total, loss_data, loss_pde, lambda_1, lambda_2)

    def train_bfgs(self, weight):
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v,
                   self.weight_tf: weight}
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss_DATA, self.loss_f, self.lambda_1, self.lambda_2],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star, t_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, p_star

    def predict_error(self):
        x, y, t, u, v, p = get_truth()
        u_pred, v_pred, p_pred = self.predict(x, y, t)
        error_u = np.linalg.norm(u_pred - u, 2) / np.linalg.norm(u, 2)
        error_v = np.linalg.norm(v_pred - v, 2) / np.linalg.norm(v, 2)
        error_vel = np.sqrt(np.sum((u_pred - u) ** 2 + (v_pred - v) ** 2)) / np.sqrt(np.sum(u ** 2 + v ** 2))
        error_max = np.max(np.sqrt((u_pred - u) ** 2 + (v_pred - v) ** 2))
        error_p = np.linalg.norm(p_pred - p, 2) / np.linalg.norm(p, 2)
        return error_u, error_v, error_vel, error_max, error_p

    def plot_result(self, filename):
        x, y, t, u, v, p = get_truth()
        _shape = (50, 100)
        t_p = 100
        x = x[t_p::200, 0]
        y = y[t_p::200, 0]
        t = t[t_p::200, 0]
        u = u[t_p::200, 0]
        v = v[t_p::200, 0]
        p = p[t_p::200, 0]

        u_pred, _, p_pred = self.predict(x[:, None], y[:, None], t[:, None])
        fig, ax = plt.subplots(6, 1, figsize=(10, 20), dpi=200)
        fig.set_tight_layout(True)
        im = ax[0].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape), u_pred.ravel().reshape(*_shape), )
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[1].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape), u.ravel().reshape(*_shape), )
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[2].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape),
                              u_pred.ravel().reshape(*_shape) - u.ravel().reshape(*_shape), cmap='bwr')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[3].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape), p_pred.ravel().reshape(*_shape), )
        divider = make_axes_locatable(ax[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[4].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape), p.ravel().reshape(*_shape), )
        divider = make_axes_locatable(ax[4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[5].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape),
                              p_pred.ravel().reshape(*_shape) - p.ravel().reshape(*_shape), cmap='bwr')
        divider = make_axes_locatable(ax[5])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax[5].set_title(f'mse: {np.mean(np.abs(p_pred - p)):.4f}')

        plt.colorbar(im, cax=cax)
        plt.savefig(filename)


def run_experiment(epoch_num, noise_type, noise, loss_type, weight, N=5000, _data=[], l_size=-1, abnormal_size=0):
    # Domain bounds
    lb = np.array([1, -2, 0])
    ub = np.array([8, 2, 20])

    # Network configuration
    uv_layers = [3] + 8 * [40] + [2]

    if len(_data) == 0:
        x_train, y_train, t_train, u_train, v_train, p_train = get_noise_data(N=N, noise_type=noise_type, sigma=noise,
                                                                              size=abnormal_size)
        _data.append((x_train, y_train, t_train, u_train, v_train))
    x_train, y_train, t_train, u_train, v_train = _data[0]
    DATA = np.concatenate(_data[0], axis=1)

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(_data[0][0].flatten(), _data[0][1].flatten(), marker='o', alpha=0.2, color='blue')
    plt.savefig('collocation.png')

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Load trained neural network
        model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers=uv_layers, ExistModel=0,
                                  uvDir='uvNN.pickle', loss_type=loss_type)
        start_time = time.time()
        model.train(adam_iter=adam_iter, learning_rate=1e-3, weight=weight)
        model.save_NN(
            path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.pickle'))

        error_u, error_v, error_vel, error_max, error_p = model.predict_error()
        lambda_1_error = model.error_lambda_1[-1]
        lambda_2_error = model.error_lambda_2[-1]
        model.plot_result(
            path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.png'))
        with open(path.joinpath('result.csv'), 'a+') as f:
            f.write(
                f"{epoch_num},{loss_type},{N},{noise_type},{noise},{abnormal_size},{weight},{error_u},{error_v},{error_vel},{error_p},{lambda_1_error}, {lambda_2_error}\n")

        print("--- %s seconds ---" % (time.time() - start_time))


def get_last_idx(filename):
    if not os.path.exists(filename):
        return -1
    with open(filename, "r") as f1:
        last_idx = int(f1.readlines()[-1].strip().split(',')[0])
        return last_idx
