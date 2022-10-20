"""
while true;do python steady_NS_two_stage.py --save_path='./data/test_two_stage';done
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
# matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from gen_data_piv import get_noise_data, get_truth
import random
from parser_pinn import get_parser
import pathlib
import glob
import sys
import time
import silence_tensorflow.auto

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf
import math
from scipy.special import erfinv

import os

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
Nf: int = int(args.Nf)
num_neurons: int = int(args.num_neurons)
num_layers: int = int(args.num_layers)


class PINN_laminar_flow_robust:
    # Initialize the class
    def __init__(self, Collo, OUTLET, WALL, DATA, uv_layers, lb, ub, ExistModel=0, uvDir='', loss_type='square'):

        # Count for callback function
        self.count = 0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        self.rho = 1.0
        self.mu = 0.02

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]

        self.x_train = DATA[:, 0:1]
        self.y_train = DATA[:, 1:2]
        self.u_train = DATA[:, 2:3]
        self.v_train = DATA[:, 3:4]

        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]

        self.x_WALL = WALL[:, 0:1]
        self.y_WALL = WALL[:, 1:2]

        # Define layers
        self.uv_layers = uv_layers

        self.loss_rec = []
        self.loss_pde = []
        self.loss_data = []
        self.loss_wall = []
        self.error_rec = []
        self.error_max = []
        self.error_p = []
        self.weight_tf = tf.placeholder(tf.float32, shape=[1])
        self.switch_tf = tf.placeholder(tf.float32, shape=[1])
        # Initialize NNs
        if ExistModel == 0:
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_train_tf = tf.placeholder(tf.float32, shape=[None, self.x_train.shape[1]])
        self.y_train_tf = tf.placeholder(tf.float32, shape=[None, self.y_train.shape[1]])
        self.u_train_tf = tf.placeholder(tf.float32, shape=[None, self.u_train.shape[1]])
        self.v_train_tf = tf.placeholder(tf.float32, shape=[None, self.v_train.shape[1]])

        self.x_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL.shape[1]])
        self.y_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL.shape[1]])

        self.x_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET.shape[1]])
        self.y_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.p_pred, _, _, _ = self.net_uv(self.x_tf, self.y_tf)
        self.u_pred_train, self.v_pred_train, _, _, _, _ = self.net_uv(self.x_train_tf, self.y_train_tf)

        self.f_pred_u, self.f_pred_v, self.f_pred_s11, self.f_pred_s22, self.f_pred_s12, \
        self.f_pred_p = self.net_f(self.x_c_tf, self.y_c_tf)
        self.u_WALL_pred, self.v_WALL_pred, _, _, _, _ = self.net_uv(self.x_WALL_tf, self.y_WALL_tf)
        _, _, self.p_OUTLET_pred, _, _, _ = self.net_uv(self.x_OUTLET_tf, self.y_OUTLET_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_pred_u)) \
                      + tf.reduce_mean(tf.square(self.f_pred_v)) \
                      + tf.reduce_mean(tf.square(self.f_pred_s11)) \
                      + tf.reduce_mean(tf.square(self.f_pred_s22)) \
                      + tf.reduce_mean(tf.square(self.f_pred_s12)) \
                      + tf.reduce_mean(tf.square(self.f_pred_p))
        self.loss_WALL = tf.reduce_mean(tf.square(self.u_WALL_pred)) \
                         + tf.reduce_mean(tf.square(self.v_WALL_pred))
        self.data_error = tf.sqrt(
            tf.square(self.u_pred_train - self.u_train_tf) + tf.square(self.v_pred_train - self.v_train_tf))

        self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred))
        self.loss_DATA_sq = tf.reduce_mean(tf.square(self.u_pred_train - self.u_train_tf)) \
                            + tf.reduce_mean(tf.square(self.v_pred_train - self.v_train_tf))
        self.loss_DATA = self.loss_DATA_sq
        self.loss_DATA_l1 = tf.reduce_mean(tf.abs(self.u_pred_train - self.u_train_tf)) \
                            + tf.reduce_mean(tf.abs(self.v_pred_train - self.v_train_tf))
        self.loss = self.weight_tf * self.loss_f + self.loss_WALL \
                    + self.loss_OUTLET + (1 - self.switch_tf) * self.loss_DATA_l1 + self.switch_tf * self.loss_DATA_sq

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': bfgs_iter,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)
        self.reset_optimizer_op = tf.variables_initializer(self.optimizer_Adam.variables())

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
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32),
                           dtype=tf.float32)

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
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, y):
        psips = self.neural_net(tf.concat([x, y], 1), self.uv_weights, self.uv_biases)
        psi = psips[:, 0:1]
        p = psips[:, 1:2]
        s11 = psips[:, 2:3]
        s22 = psips[:, 3:4]
        s12 = psips[:, 4:5]
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]
        return u, v, p, s11, s22, s12

    def net_f(self, x, y):

        rho = self.rho
        mu = self.mu
        u, v, p, s11, s22, s12 = self.net_uv(x, y)

        s11_1 = tf.gradients(s11, x)[0]
        s12_2 = tf.gradients(s12, y)[0]
        s22_2 = tf.gradients(s22, y)[0]
        s12_1 = tf.gradients(s12, x)[0]

        # Plane stress problem
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]

        # f_u:=Sxx_x+Sxy_y
        f_u = rho * (u * u_x + v * u_y) - s11_1 - s12_2
        f_v = rho * (u * v_x + v * v_y) - s12_1 - s22_2

        # f_mass = u_x+v_y

        f_s11 = -p + 2 * mu * u_x - s11
        f_s22 = -p + 2 * mu * v_y - s22
        f_s12 = mu * (u_y + v_x) - s12

        f_p = p + (s11 + s22) / 2

        return f_u, f_v, f_s11, f_s22, f_s12, f_p

    def callback(self, loss_value, loss_wall, loss_data, loss_pde):
        self.count = self.count + 1
        self.loss_rec.append(loss_value)
        self.loss_data.append(loss_data)
        self.loss_pde.append(loss_pde)
        self.loss_wall.append(loss_wall)
        if self.count % 10 == 0:
            _, _, error_rel, error_max, error_p = self.predict_error()
            self.error_rec.append(error_rel)
            self.error_max.append(error_max)
            self.error_p.append(error_p)
            print('It: %d, Loss: %.3e, Loss pde: %.3e, Loss w: %.3e, Loss d: %.3e' % (
                self.count, loss_value, loss_pde, loss_wall, loss_data))

    def train(self, adam_iter, learning_rate, weight, switch):

        tf_dict = {self.x_c_tf: self.x_c,
                   self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL,
                   self.y_WALL_tf: self.y_WALL,
                   self.x_OUTLET_tf: self.x_OUTLET,
                   self.y_OUTLET_tf: self.y_OUTLET,
                   self.x_train_tf: self.x_train,
                   self.y_train_tf: self.y_train,
                   self.u_train_tf: self.u_train,
                   self.v_train_tf: self.v_train,
                   self.learning_rate: learning_rate,
                   self.weight_tf: weight,
                   self.switch_tf: switch}

        print("train:", self.x_train.shape, self.y_train.shape, self.u_train.shape, self.v_train.shape, )

        for it in range(adam_iter):
            loss_value, loss_wall, loss_data, loss_pde, _ = self.sess.run(
                [self.loss, self.loss_WALL, self.loss_DATA, self.loss_f, self.train_op_Adam], tf_dict)
            self.callback(loss_value, loss_wall, loss_data, loss_pde)
        self.train_bfgs(weight=np.array(weight), switch=switch)
        tf_dict[self.learning_rate] = 1e-20

        for it in range(100):
            loss_value, loss_wall, loss_data, loss_pde, _ = self.sess.run(
                [self.loss, self.loss_WALL, self.loss_DATA, self.loss_f, self.train_op_Adam], tf_dict)
            self.callback(loss_value, loss_wall, loss_data, loss_pde)

    def train_bfgs(self, weight, switch):

        tf_dict = {self.x_c_tf: self.x_c,
                   self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL,
                   self.y_WALL_tf: self.y_WALL,
                   self.x_OUTLET_tf: self.x_OUTLET,
                   self.y_OUTLET_tf: self.y_OUTLET,
                   self.x_train_tf: self.x_train,
                   self.y_train_tf: self.y_train,
                   self.u_train_tf: self.u_train,
                   self.v_train_tf: self.v_train,
                   self.weight_tf: weight,
                   self.switch_tf: switch}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss_WALL, self.loss_DATA, self.loss_f],
                                loss_callback=self.callback)

    def sieve_obs(self, ratio):
        tf_dict = {self.x_train_tf: self.x_train,
                   self.y_train_tf: self.y_train,
                   self.u_train_tf: self.u_train,
                   self.v_train_tf: self.v_train, }

        error = self.sess.run(self.data_error, tf_dict).ravel()
        print(error)
        execluded_number = int(len(error) * ratio)
        print(f'{execluded_number} observations have been removed.')
        ind = np.argpartition(error, -execluded_number)[:-execluded_number]

        self.x_train = self.x_train[ind, :]
        self.y_train = self.y_train[ind, :]
        self.u_train = self.u_train[ind, :]
        self.v_train = self.v_train[ind, :]

        error = self.sess.run(self.data_error, tf_dict).ravel()
        print("sieve:", self.x_train.shape, self.y_train.shape, self.u_train.shape, self.v_train.shape, )

        tf_dict = {self.x_c_tf: self.x_c,
                   self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL,
                   self.y_WALL_tf: self.y_WALL,
                   self.x_OUTLET_tf: self.x_OUTLET,
                   self.y_OUTLET_tf: self.y_OUTLET,
                   self.x_train_tf: self.x_train,
                   self.y_train_tf: self.y_train,
                   self.u_train_tf: self.u_train,
                   self.v_train_tf: self.v_train,
                   self.learning_rate: 1e-3,
                   self.weight_tf: np.array([1.0]),
                   self.switch_tf: np.array([1.0])}
        return error

    def sieve_obs_sigma(self, k=2):
        tf_dict = {self.x_train_tf: self.x_train,
                   self.y_train_tf: self.y_train,
                   self.u_train_tf: self.u_train,
                   self.v_train_tf: self.v_train, }
        error = self.sess.run(self.data_error, tf_dict).ravel()
        sigma = np.median(np.abs(error)) / math.sqrt(2) / erfinv(0.5)
        ind = (np.abs(error) <= k * sigma)
        print(error)
        print()
        original_N = len(error)

        self.x_train = self.x_train[ind, :]
        self.y_train = self.y_train[ind, :]
        self.u_train = self.u_train[ind, :]
        self.v_train = self.v_train[ind, :]

        new_N = len(self.x_train)
        print(f'{original_N - new_N} observations have been removed.')

        error = self.sess.run(self.data_error, tf_dict).ravel()
        print("sieve:", self.x_train.shape, self.y_train.shape, self.u_train.shape, self.v_train.shape, )

    def predict(self, x_star, y_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star})
        p_star = self.sess.run(self.p_pred, {self.x_tf: x_star, self.y_tf: y_star})
        return u_star, v_star, p_star

    def predict_error(self):
        x, y, u, v, p = get_truth(pressure=True)
        u_pred, v_pred, p_pred = self.predict(x, y)
        error_u = np.linalg.norm(u_pred - u, 2) / np.linalg.norm(u, 2)
        error_v = np.linalg.norm(v_pred - v, 2) / np.linalg.norm(v, 2)
        error_vel = np.sqrt(np.sum((u_pred - u) ** 2 + (v_pred - v) ** 2)) / np.sqrt(np.sum(u ** 2 + v ** 2))
        error_max = np.max(np.sqrt((u_pred - u) ** 2 + (v_pred - v) ** 2))
        error_p = np.linalg.norm(p_pred - p, 2) / np.linalg.norm(p, 2)
        return error_u, error_v, error_vel, error_max, error_p

    def plot_result(self, filename):
        model = self
        x_train, y_train, u_train, v_train, p_train = get_truth(pressure=True)
        u_pred_train, v_pred_train, p_pred_train = model.predict(x_train, y_train)
        fig, ax = plt.subplots(7, 1, figsize=(10, 23))
        fig.set_tight_layout(True)
        im = ax[0].scatter(x_train, y_train, c=np.hypot(u_train, v_train))
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[1].scatter(x_train, y_train, c=np.hypot(u_pred_train, v_pred_train))
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[2].scatter(x_train, y_train, c=np.hypot(u_train - u_pred_train, v_train - v_pred_train), vmin=0.,
                           vmax=0.5)
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[2].set_title(f'mse: {np.mean(np.hypot(u_train - u_pred_train, v_train - v_pred_train) ** 2):.4f}')

        im = ax[3].scatter(x_train, y_train, c=p_train)
        divider = make_axes_locatable(ax[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[4].scatter(x_train, y_train, c=p_pred_train, vmin=0, vmax=3.6)
        divider = make_axes_locatable(ax[4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[5].scatter(x_train, y_train, c=np.abs(p_train - p_pred_train), vmin=0, vmax=3.6)
        divider = make_axes_locatable(ax[5])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax[5].set_title(f'mse: {np.mean(np.abs(p_pred_train - p_train)):.4f}')
        plt.colorbar(im, cax=cax)

        im = ax[6].scatter(self.x_train.ravel(), self.y_train.ravel(), s=4)

        plt.savefig(filename)


def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst > r, :]


def run_experiment(epoch_num, noise_type, noise, loss_type, N, weight, _data=[], l_size=-1, abnormal_size=0,
                   sieve_sigma=0, sieve_ratio=0):
    # Domain bounds
    lb = np.array([0, 0])
    ub = np.array([1.1, 0.41])

    # Network configuration
    uv_layers = [2] + 8 * [40] + [5]

    wall_up = [0.0, 0.41] + [1.1, 0.0] * lhs(2, 101)
    wall_lw = [0.0, 0.00] + [1.1, 0.0] * lhs(2, 101)
    OUTLET = [1.1, 0.0] + [0.0, 0.41] * lhs(2, 201)

    # Cylinder surface
    r = 0.05
    theta = [0.0] + [2 * np.pi] * lhs(1, 360)
    x_CYLD = np.multiply(r, np.cos(theta)) + 0.2
    y_CYLD = np.multiply(r, np.sin(theta)) + 0.2
    CYLD = np.concatenate((x_CYLD, y_CYLD), 1)
    WALL = np.concatenate((CYLD, wall_up, wall_lw), 0)

    # Collocation point for equation residual
    XY_c = lb + (ub - lb) * lhs(2, 40000)
    XY_c_refine = [0.1, 0.1] + [0.2, 0.2] * lhs(2, 10000)
    XY_c = np.concatenate((XY_c, XY_c_refine), 0)
    XY_c = DelCylPT(XY_c, xc=0.2, yc=0.2, r=0.05)

    # np.random.seed(0)
    if len(_data) == 0:
        x_train, y_train, u_train, v_train = get_noise_data(N=N, noise_type=noise_type, sigma=noise, size=abnormal_size)
        _data.append((x_train, y_train, u_train, v_train))
    DATA = np.concatenate(_data[0], axis=1)

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Load trained neural network
        if (sieve_sigma == 0) and (sieve_ratio == 0):
            model = PINN_laminar_flow_robust(XY_c, OUTLET, WALL, DATA, uv_layers, lb, ub,
                                             ExistModel=0,
                                             uvDir='uvNN.pickle',
                                             loss_type=loss_type)
            if loss_type == 'l1':
                model.train(adam_iter=adam_iter, learning_rate=1e-3, weight=weight, switch=np.array([0.]))
            elif loss_type == 'square':
                model.train(adam_iter=adam_iter, learning_rate=1e-3, weight=weight, switch=np.array([1.]))
        else:
            model = PINN_laminar_flow_robust(XY_c, OUTLET, WALL, DATA, uv_layers, lb, ub,
                                             ExistModel=1,
                                             uvDir=glob.glob(str(
                                                 path) + '/' + f"*_l1_{N}_{noise_type}_{noise}_{abnormal_size}_*_0_0.pickle")[
                                                 0],
                                             loss_type=loss_type)
            if sieve_sigma > 0 and sieve_ratio == 0:
                model.sieve_obs_sigma(k=sieve_sigma)
            elif sieve_sigma == 0 and sieve_ratio > 0:
                model.sieve_obs(ratio=sieve_ratio)
            else:
                print(sieve_sigma, sieve_ratio)
                raise Exception('not good')
            assert loss_type == 'square'
            model.train(adam_iter=1000, learning_rate=1e-3, weight=weight, switch=np.array([1.]))

        model.save_NN(path.joinpath(
            f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}_{sieve_ratio}.pickle'))
        error_u, error_v, error_vel, error_max, error_p = model.predict_error()
        model.plot_result(path.joinpath(
            f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}_{sieve_ratio}.png'))
        with open(path.joinpath('result.csv'), 'a+') as f:
            f.write(
                f"{epoch_num},{loss_type},{N},{noise_type},{noise},{abnormal_size},{weight},{error_u},{error_v},{error_vel}, {error_p},{sieve_sigma},{sieve_ratio},{len(model.x_train)}\n")


def get_last_idx(filename):
    if not os.path.exists(filename):
        return -1
    with open(filename, "r") as f1:
        last_idx = int(f1.readlines()[-1].strip().split(',')[0])
        return last_idx


if __name__ == "__main__":
    idx = 0
    last_idx = get_last_idx(path.joinpath('result.csv'))
    executed_flag = False

    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise, abnormal_ratio in zip([0.20], [0.20]):
                for N in [500]:
                    for loss_type, sieve_sigma, sieve_ratio in zip(
                            ['l1', 'square', 'square', 'square', 'square', 'square', 'square', 'square'],
                            [0, 0, 0, 0, 0, 2, 2.5, 3],
                            [0, 0, 0.1, 0.2, 0.3, 0, 0, 0]):
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1E-0]:
                            if idx > last_idx:
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                               loss_type=loss_type, abnormal_size=abnormal_size,
                                               sieve_sigma=sieve_sigma, sieve_ratio=sieve_ratio)
                                executed_flag = True
                            idx += 1
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['contamined', 'normal', 't1', 'none']:
            for noise in [0.20]:
                for abnormal_ratio in [0]:
                    for N in [500]:
                        # _data= []
                        for loss_type, sieve_sigma, sieve_ratio in zip(
                                ['l1', 'square', 'square', 'square', 'square', 'square', 'square', 'square', ],
                                [0, 0, 0, 0, 0, 2, 2.5, 3, ],
                                [0, 0, 0.1, 0.2, 0.3, 0, 0, 0, ]):
                            abnormal_size = int(N * abnormal_ratio)
                            if executed_flag:
                                sys.exit()
                            for weight in [1E-0]:
                                if idx > last_idx:
                                    run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                                   loss_type=loss_type, abnormal_size=abnormal_size,
                                                   sieve_sigma=sieve_sigma, sieve_ratio=sieve_ratio)
                                    executed_flag = True
                                idx += 1
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise in [0.0]:
                for abnormal_ratio in [0.20]:
                    for N in [500]:
                        for loss_type, sieve_sigma, sieve_ratio in zip(
                                ['l1', 'square', 'square', 'square', 'square', 'square', 'square', 'square'],
                                [0, 0, 0, 0, 0, 2, 2.5, 3],
                                [0, 0, 0.1, 0.2, 0.3, 0, 0, 0]):
                            abnormal_size = int(N * abnormal_ratio)
                            if executed_flag:
                                sys.exit()
                            for weight in [1E-0]:
                                if idx > last_idx:
                                    run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                                   loss_type=loss_type, abnormal_size=abnormal_size,
                                                   sieve_sigma=sieve_sigma, sieve_ratio=sieve_ratio)
                                    executed_flag = True
                                idx += 1
    if executed_flag:
        sys.exit()

    time.sleep(5)
