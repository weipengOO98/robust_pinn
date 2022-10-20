"""
while true; do python unsteady_NS_noise.py --adam_iter=10000 --save_path='./data/test_noise';done
"""

from unsteady_NS import *

if __name__ == "__main__":
    idx = 0
    last_idx = get_last_idx(path.joinpath('result.csv'))
    executed_flag = False
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise, abnormal_ratio in zip([0.01, 0.05, 0.10, 0.20], [0.01, 0.05, 0.10, 0.20]):
                for N in [5000]:
                    _data = []
                    abnormal_size = int(N * abnormal_ratio)
                    if executed_flag:
                        sys.exit()
                    for weight in [1E-0]:
                        if idx > last_idx:
                            run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                           loss_type='l1', _data=_data, abnormal_size=abnormal_size)
                            executed_flag = True
                        idx += 1
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise in [0.0]:
                for abnormal_ratio in [0.01, 0.05, 0.10, 0.20]:
                    for N in [5000]:
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1E-0]:
                            if idx > last_idx:
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                               loss_type='l1', _data=_data, abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1

    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['t1', 'contamined', 'normal']:
            for noise in [0.01, 0.05, 0.10, 0.20]:
                for abnormal_ratio in [0]:
                    for N in [5000]:
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1E-0]:
                            if idx > last_idx:
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                               loss_type='l1', _data=_data, abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1

    if executed_flag:
        sys.exit()

    for r in range(repeat):
        for noise_type in ['none']:
            for noise in [0.0]:
                for abnormal_ratio in [0]:
                    for N in [5000]:
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1E-0]:
                            if idx > last_idx:
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=[weight],
                                               loss_type='l1', _data=_data, abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1

    time.sleep(1)
