"""
while true; do python wave_eq_size.py --save_path='./data/test_size';done
"""
from wave_eq import *

if __name__ == "__main__":
    idx = 0
    last_idx = get_last_idx(path.joinpath('result.csv'))
    executed_flag = False

    for r in range(repeat):
        for noise_type in ['normal', 't1', 'contamined', 'none']:
            for noise in [0.10]:
                for abnormal_ratio in [0]:
                    for N in [1000, 800, 600, 400, 200]:
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1.0]:
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
                for abnormal_ratio in [0.10]:
                    for N in [1000, 800, 600, 400, 200]:
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1.0]:
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
                for N in [1000, 800, 600, 400, 200]:
                    _data = []
                    abnormal_size = int(N * abnormal_ratio)
                    for weight in [1.0]:
                        if executed_flag:
                            sys.exit()
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
