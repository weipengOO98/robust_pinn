"""_summary_
while true; do python poisson_eq_weight.py --save_path='./data/test_weight' --adam_iter=15000; done
"""
from poisson_eq import *

if __name__ == "__main__":
    idx = 0
    last_idx = get_last_idx(path.joinpath('result.csv'))
    executed_flag = False

    for r in range(repeat):
        for noise_type in ['normal']:
            for noise in [0.20]:
                for abnormal_ratio in [0]:
                    for N in [400]:
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [0, 1E-4, 1E-3, 1E-2, 1E-1, 1E-0, 1E1, 1E2, 1E3]:
                            if idx > last_idx:
                                run_experiment(idx, noise_type, noise, 'square', N, [weight], _data,
                                               abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1

                            if idx > last_idx:
                                run_experiment(idx, noise_type, noise, 'l1', N, [weight], _data,
                                               abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1

    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise in [0]:
                for abnormal_ratio in [0.2]:
                    for N in [400]:
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        for weight in [0, 1E-4, 1E-3, 1E-2, 1E-1, 1E-0, 1E1, 1E2, 1E3]:
                            if idx > last_idx:
                                run_experiment(idx, noise_type, noise, 'square', N, [weight], _data,
                                               abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1

                            if idx > last_idx:
                                run_experiment(idx, noise_type, noise, 'l1', N, [weight], _data,
                                               abnormal_size=abnormal_size)
                                executed_flag = True
                            idx += 1
    if executed_flag:
        sys.exit()

    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise, abnormal_ratio in zip([0.10], [0.10]):
                for N in [400]:
                    _data = []
                    abnormal_size = int(N * abnormal_ratio)
                    for weight in [0, 1E-4, 1E-3, 1E-2, 1E-1, 1E-0, 1E1, 1E2, 1E3]:
                        if idx > last_idx:
                            run_experiment(idx, noise_type, noise, 'square', N, [weight], _data,
                                           abnormal_size=abnormal_size)
                            executed_flag = True
                        idx += 1

                        if idx > last_idx:
                            run_experiment(idx, noise_type, noise, 'l1', N, [weight], _data,
                                           abnormal_size=abnormal_size)
                            executed_flag = True
                        idx += 1
        if executed_flag:
            sys.exit()

    for r in range(repeat):
        for noise_type in ['none']:
            for noise, abnormal_ratio in zip([0.0], [0.0]):
                for N in [400]:
                    _data = []
                    abnormal_size = int(N * abnormal_ratio)
                    for weight in [0, 1E-4, 1E-3, 1E-2, 1E-1, 1E-0, 1E1, 1E2, 1E3]:
                        if idx > last_idx:
                            run_experiment(idx, noise_type, noise, 'square', N, [weight], _data,
                                           abnormal_size=abnormal_size)
                            executed_flag = True
                        idx += 1

                        if idx > last_idx:
                            run_experiment(idx, noise_type, noise, 'l1', N, [weight], _data,
                                           abnormal_size=abnormal_size)
                            executed_flag = True
                        idx += 1
    time.sleep(1)
