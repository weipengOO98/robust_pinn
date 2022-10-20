import argparse
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--adam_iter', default=15000, type=int
    )
    parser.add_argument(
        '--bfgs_iter', default=1000, type=int
    )
    parser.add_argument(
        '--Nf', default=2000, type=int
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
        '--start_epoch', default=0
    )
    parser.add_argument(
        '--num_layers', default=4
    )
    parser.add_argument(
        '--num_neurons', default=50
    )
    return parser
