import numpy as np

base = list(range(99))
sample_freq = 5
n_sample = 16
availible_n_samples = len(base) // sample_freq
assert availible_n_samples >= n_sample, 'not enough availible samples, reduce sample frequency !'
print(availible_n_samples)

samples = list()
cur_n_samples = 0
while cur_n_samples <= n_sample:
    start_i = np.random.randint(0, len(base) - n_sample * sample_freq)
    for i in range(len(base)):
        if (i - start_i) % sample_freq == 0:
            samples.append(base[i])
            cur_n_samples += 1

print(samples)

