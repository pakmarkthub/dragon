import csv

import matplotlib.pyplot as plt
import numpy as np
import math

num_sys_pagefaults_read_array = dict()
num_sys_pagefaults_write_array = dict()
num_sys_pagefaults_flush_array = dict()
num_gpu_pagefaults_read_array = dict()
num_gpu_pagefaults_write_array = dict()

with open('../matrixMul-nvmgpu/results/result.data', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        size_in_gib = int(row['size_in_gib'])
        num_sys_pagefaults_read = int(row['num_host_stage_pages_read'])
        num_sys_pagefaults_write = int(row['num_host_stage_pages_write'])
        num_sys_pagefaults_flush = int(row['num_host_stage_pages_flush'])
        num_gpu_pagefaults_read = int(row['num_gpu_pagefaults_read'])
        num_gpu_pagefaults_write = int(row['num_gpu_pagefaults_write'])

        if size_in_gib not in num_sys_pagefaults_read_array:
            num_sys_pagefaults_read_array[size_in_gib] = list()
            num_sys_pagefaults_write_array[size_in_gib] = list()
            num_sys_pagefaults_flush_array[size_in_gib] = list()
            num_gpu_pagefaults_read_array[size_in_gib] = list()
            num_gpu_pagefaults_write_array[size_in_gib] = list()
        
        num_sys_pagefaults_read_array[size_in_gib].append(num_sys_pagefaults_read)
        num_sys_pagefaults_write_array[size_in_gib].append(num_sys_pagefaults_write)
        num_sys_pagefaults_flush_array[size_in_gib].append(num_sys_pagefaults_flush)
        num_gpu_pagefaults_read_array[size_in_gib].append(num_gpu_pagefaults_read)
        num_gpu_pagefaults_write_array[size_in_gib].append(num_gpu_pagefaults_write)

x_array, num_sys_pagefaults_read_array = zip(*sorted(num_sys_pagefaults_read_array.items(), key = lambda item: item[0]))
num_sys_pagefaults_write_array = list(zip(*sorted(num_sys_pagefaults_write_array.items(), key = lambda item: item[0])))[1]
num_sys_pagefaults_flush_array = list(zip(*sorted(num_sys_pagefaults_flush_array.items(), key = lambda item: item[0])))[1]
num_gpu_pagefaults_read_array = list(zip(*sorted(num_gpu_pagefaults_read_array.items(), key = lambda item: item[0])))[1]
num_gpu_pagefaults_write_array = list(zip(*sorted(num_gpu_pagefaults_write_array.items(), key = lambda item: item[0])))[1]

fig, ax1 = plt.subplots()
legend_array = list()
        
legend_array.append((
    ax1.errorbar(x_array, [np.mean(item) for item in num_sys_pagefaults_read_array], yerr = [np.std(item) for item in num_sys_pagefaults_read_array])[0],
    'Sys page faults read (measured)',
))
legend_array.append((
    ax1.errorbar(x_array, [np.mean(item) for item in num_sys_pagefaults_write_array], yerr = [np.std(item) for item in num_sys_pagefaults_write_array])[0],
    'Sys page faults write (measured)',
))
legend_array.append((
    ax1.errorbar(x_array, [np.mean(item) for item in num_sys_pagefaults_flush_array], yerr = [np.std(item) for item in num_sys_pagefaults_flush_array])[0],
    'Sys page faults flush (measured)',
))

ax1.set_ylabel("Number of system page faults")

ax2 = ax1.twinx()

legend_array.append((
    ax2.errorbar(x_array, [np.mean(item) for item in num_gpu_pagefaults_read_array], yerr = [np.std(item) for item in num_gpu_pagefaults_read_array], fmt = '--')[0],
    'GPU page faults read (measured)',
))
legend_array.append((
    ax2.errorbar(x_array, [np.mean(item) for item in num_gpu_pagefaults_write_array], yerr = [np.std(item) for item in num_gpu_pagefaults_write_array], fmt = '--')[0],
    'GPU page faults write (measured)',
))

ax2.set_ylabel("Number of GPU page faults")

ax1.set_xlabel("Matrix size (GB)")

objs, labels = list(zip(*legend_array))
plt.legend(objs, labels)

plt.title('matrixMul')
plt.show()
