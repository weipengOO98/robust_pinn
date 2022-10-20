"""
while true; do python steady_NS_noise.py --save_path='./data/test_noise';done
"""

from steady_NS import *

if __name__ == "__main__":
    idx = 0
    last_idx = get_last_idx(path.joinpath('result.csv'))
    executed_flag = False
    for r in range(repeat):
        for noise_type in ['contamined', 'normal', 't1']:
            for noise in [0.30, 0.25, 0.20, 0.15, 0.10]:
                for abnormal_ratio in [0]:
                    for N in [1000]:
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1E-0]:
                            if idx > last_idx:
                                tf.reset_default_graph()
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                               loss_type='l1', _data=_data, abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1
                            if idx > last_idx:
                                tf.reset_default_graph()
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                               loss_type='square', _data=_data, abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise in [0.0]:
                for abnormal_ratio in [0.30, 0.25, 0.20, 0.15, 0.10]:
                    for N in [1000]:
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1E-0]:
                            if idx > last_idx:
                                tf.reset_default_graph()
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                               loss_type='l1', _data=_data, abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1
                            if idx > last_idx:
                                tf.reset_default_graph()
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                               loss_type='square', _data=_data, abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise, abnormal_ratio in zip([0.30, 0.25, 0.20, 0.15, 0.10], [0.30, 0.25, 0.20, 0.15, 0.10]):
                for N in [1000]:
                    _data = []
                    abnormal_size = int(N * abnormal_ratio)
                    if executed_flag:
                        sys.exit()
                    for weight in [1E-0]:
                        if idx > last_idx:
                            tf.reset_default_graph()
                            run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                           loss_type='l1', _data=_data, abnormal_size=abnormal_size)
                            executed_flag = True
                        idx += 1
                        if idx > last_idx:
                            tf.reset_default_graph()
                            run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                           loss_type='square', _data=_data, abnormal_size=abnormal_size)
                            executed_flag = True
                        idx += 1
    if executed_flag:
        sys.exit()

    time.sleep(1)
