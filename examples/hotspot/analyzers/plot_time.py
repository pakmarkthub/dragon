import csv

import matplotlib.pyplot as plt
import numpy as np

kernel_time_array = dict() 
readfile_time_array = dict()
writefile_time_array = dict()
flushfile_time_array = dict()
make_resident_time_array = dict()

with open('../matrixMul-nvmgpu/results/result.data', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        size_in_gib = int(row['size_in_gib'])
        kernel_time = float(row['kernel_time (ms)']) / 1000.0
        readfile_time = float(row['readfile (s)'])
        writefile_time = float(row['writefile (s)'])
        flushfile_time = float(row['flushfile (s)'])
        make_resident_time = float(row['make_resident (s)'])

        if size_in_gib not in kernel_time_array:
            kernel_time_array[size_in_gib] = list()
            readfile_time_array[size_in_gib] = list()
            writefile_time_array[size_in_gib] = list()
            flushfile_time_array[size_in_gib] = list()
            make_resident_time_array[size_in_gib] = list()
        
        kernel_time_array[size_in_gib].append(kernel_time)
        readfile_time_array[size_in_gib].append(readfile_time)
        writefile_time_array[size_in_gib].append(writefile_time)
        flushfile_time_array[size_in_gib].append(flushfile_time)
        make_resident_time_array[size_in_gib].append(make_resident_time)

x_array, kernel_time_array = zip(*sorted(kernel_time_array.items(), key = lambda item: item[0]))
readfile_time_array = list(zip(*sorted(readfile_time_array.items(), key = lambda item: item[0])))[1]
writefile_time_array = list(zip(*sorted(writefile_time_array.items(), key = lambda item: item[0])))[1]
flushfile_time_array = list(zip(*sorted(flushfile_time_array.items(), key = lambda item: item[0])))[1]
make_resident_time_array = list(zip(*sorted(make_resident_time_array.items(), key = lambda item: item[0])))[1]

kernel_time_means = np.asarray([np.mean(item) for item in kernel_time_array])
kernel_time_std = np.asarray([np.std(item) for item in kernel_time_array])
readfile_time_means = np.asarray([np.mean(item) for item in readfile_time_array])
readfile_time_std = np.asarray([np.std(item) for item in readfile_time_array])
writefile_time_means = np.asarray([np.mean(item) for item in writefile_time_array])
writefile_time_std = np.asarray([np.std(item) for item in writefile_time_array])
flushfile_time_means = np.asarray([np.mean(item) for item in flushfile_time_array])
flushfile_time_std = np.asarray([np.std(item) for item in flushfile_time_array])
make_resident_time_means = np.asarray([np.mean(item) for item in make_resident_time_array])
make_resident_time_std = np.asarray([np.std(item) for item in make_resident_time_array])

plt.errorbar(
    x_array,
    kernel_time_means,
    yerr = kernel_time_std,
    label = 'kernel'
)

plt.errorbar(
    x_array,
    readfile_time_means,
    yerr = readfile_time_std,
    label = 'readfile'
)

plt.errorbar(
    x_array,
    writefile_time_means,
    yerr = writefile_time_std,
    label = 'writefile'
)

plt.errorbar(
    x_array,
    flushfile_time_means,
    yerr = flushfile_time_std,
    label = 'flushfile'
)

plt.errorbar(
    x_array,
    make_resident_time_means,
    yerr = make_resident_time_std,
    label = 'make_resident'
)

plt.xlabel("Matrix size (GB)")
plt.ylabel("Time (s)")
plt.title("matrixMul")
plt.legend()
plt.show()

