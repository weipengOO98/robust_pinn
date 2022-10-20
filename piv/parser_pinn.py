import argparse
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--adam_iter', default=10000, type=int
    )
    parser.add_argument(
        '--bfgs_iter', default=100000, type=int
    )
    parser.add_argument(
        '--Nf', default=8000, type=int
    )
    parser.add_argument(
        '--N_increase', default=1, type=int
    )
    parser.add_argument(
        '--save_path', default=f'./data/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    )
    parser.add_argument(
        '--verbose', default=False
    )
    parser.add_argument(
        '--repeat', default=1
    )
    parser.add_argument(
        '--num_layers', default=8
    )
    parser.add_argument(
        '--num_neurons', default=40
    )
    parser.add_argument(
        '--start_epoch', default=0
    )
    return parser
