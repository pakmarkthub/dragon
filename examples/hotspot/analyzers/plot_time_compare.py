import csv

import matplotlib.pyplot as plt
import numpy as np

width = 0.5

COLORS = {
    'write': 'r',
    'exec': 'b',
    'd2h_memcpy': 'g',
    'h2d_memcpy': 'y',
    'read': (0.7, 0.2, 0,),
    'flush': (0.5, 0.0, 0.5,),
    'make_resident': 'c',
}

folder_size_map = {
    16: 1.0 * 3,
    20: 1.6 * 3,
    24: 2.3 * 3,
    28: 3.1 * 3,
    32: 4.0 * 3,
    40: 6.3 * 3,
    48: 9.0 * 3,
    56: 13 * 3,
    64: 16 * 3,
}

fig, ax = plt.subplots()


exec_time_array = dict()
writefile_time_array = dict()
readfile_time_array = dict()
h2d_memcpy_time_array = dict()
d2h_memcpy_time_array = dict()

with open('../hotspot-cudamemcpy/results/result.data', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        folder = folder_size_map[int(row['folder'][:-1])]
        exec_time = float(row['kernel_time (ms)']) / 1000.0
        writefile_time = float(row['writefile_time (ms)']) / 1000.0
        readfile_time = float(row['readfile_time (ms)']) / 1000.0
        h2d_memcpy_time = float(row['h2d_memcpy_time (ms)']) / 1000.0
        d2h_memcpy_time = float(row['d2h_memcpy_time (ms)']) / 1000.0

        if folder not in exec_time_array:
            exec_time_array[folder] = list()
            writefile_time_array[folder] = list()
            readfile_time_array[folder] = list()
            h2d_memcpy_time_array[folder] = list()
            d2h_memcpy_time_array[folder] = list()
        
        exec_time_array[folder].append(exec_time)
        writefile_time_array[folder].append(writefile_time)
        readfile_time_array[folder].append(readfile_time)
        h2d_memcpy_time_array[folder].append(h2d_memcpy_time)
        d2h_memcpy_time_array[folder].append(d2h_memcpy_time)

x_array, exec_time_array = zip(*sorted(exec_time_array.items(), key = lambda item: item[0]))
writefile_time_array = list(zip(*sorted(writefile_time_array.items(), key = lambda item: item[0])))[1]
readfile_time_array = list(zip(*sorted(readfile_time_array.items(), key = lambda item: item[0])))[1]
h2d_memcpy_time_array = list(zip(*sorted(h2d_memcpy_time_array.items(), key = lambda item: item[0])))[1]
d2h_memcpy_time_array = list(zip(*sorted(d2h_memcpy_time_array.items(), key = lambda item: item[0])))[1]

x_array = np.asarray(x_array) - width
writefile_time_means = np.asarray([np.mean(item) for item in writefile_time_array])
readfile_time_means = np.asarray([np.mean(item) for item in readfile_time_array])
exec_time_means = np.asarray([np.mean(item) for item in exec_time_array])
h2d_memcpy_time_means = np.asarray([np.mean(item) for item in h2d_memcpy_time_array])
d2h_memcpy_time_means = np.asarray([np.mean(item) for item in d2h_memcpy_time_array])

plt.bar(
    x_array,
    readfile_time_means,
    width,
    yerr = [np.std(item) for item in readfile_time_array],
    label = 'read',
    color = COLORS['read'],
    edgecolor = 'k'
)
bottom = readfile_time_means

plt.bar(
    x_array,
    writefile_time_means,
    width,
    yerr = [np.std(item) for item in writefile_time_array],
    label = 'write',
    color = COLORS['write'],
    bottom = bottom,
    edgecolor = 'k'
)
bottom += writefile_time_means

plt.bar(
    x_array,
    h2d_memcpy_time_means,
    width,
    yerr = [np.std(item) for item in h2d_memcpy_time_array],
    label = 'memcpy (h2d)',
    bottom = bottom,
    color = COLORS['h2d_memcpy'],
    edgecolor = 'k'
)
bottom += h2d_memcpy_time_means

plt.bar(
    x_array,
    d2h_memcpy_time_means,
    width,
    yerr = [np.std(item) for item in d2h_memcpy_time_array],
    label = 'memcpy (d2h)',
    bottom = bottom, 
    color = COLORS['d2h_memcpy'],
    edgecolor = 'k'
)
bottom += d2h_memcpy_time_means

plt.bar(
    x_array,
    exec_time_means,
    width,
    yerr = [np.std(item) for item in exec_time_array],
    label = 'exec',
    bottom = bottom, 
    color = COLORS['exec'],
    edgecolor = 'k'
)
bottom += exec_time_means



exec_time_array = dict()
writefile_time_array = dict()
readfile_time_array = dict()

with open('../hotspot-uvm/results/result.data', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        folder = folder_size_map[int(row['folder'][:-1])]
        exec_time = float(row['kernel_time (ms)']) / 1000.0
        writefile_time = float(row['writefile_time (ms)']) / 1000.0
        readfile_time = float(row['readfile_time (ms)']) / 1000.0

        if folder not in exec_time_array:
            exec_time_array[folder] = list()
            writefile_time_array[folder] = list()
            readfile_time_array[folder] = list()
        
        exec_time_array[folder].append(exec_time)
        writefile_time_array[folder].append(writefile_time)
        readfile_time_array[folder].append(readfile_time)

x_array, exec_time_array = zip(*sorted(exec_time_array.items(), key = lambda item: item[0]))
writefile_time_array = list(zip(*sorted(writefile_time_array.items(), key = lambda item: item[0])))[1]
readfile_time_array = list(zip(*sorted(readfile_time_array.items(), key = lambda item: item[0])))[1]

x_array = np.asarray(x_array)
writefile_time_means = np.asarray([np.mean(item) for item in writefile_time_array])
readfile_time_means = np.asarray([np.mean(item) for item in readfile_time_array])
exec_time_means = np.asarray([np.mean(item) for item in exec_time_array])

plt.bar(
    x_array,
    readfile_time_means,
    width,
    yerr = [np.std(item) for item in readfile_time_array],
    color = COLORS['read'],
    edgecolor = 'k'
)
bottom = readfile_time_means

plt.bar(
    x_array,
    writefile_time_means,
    width,
    yerr = [np.std(item) for item in writefile_time_array],
    bottom = bottom,
    color = COLORS['write'],
    edgecolor = 'k'
)
bottom += writefile_time_means

plt.bar(
    x_array,
    exec_time_means,
    width,
    yerr = [np.std(item) for item in exec_time_array],
    bottom = bottom,
    color = COLORS['exec'],
    edgecolor = 'k'
)
bottom += exec_time_means



exec_time_array = dict()
readfile_time_array = dict()
flushfile_time_array = dict()
make_resident_time_array = dict()
h2d_time_array = dict()
d2h_time_array = dict()

with open('../hotspot-nvmgpu/results/result.data', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        folder = folder_size_map[int(row['folder'][:-1])]
        kernel_time = float(row['kernel_time (ms)']) / 1000.0
        free_time = float(row['free_time (ms)']) / 1000.0
        readfile_time = float(row['readfile (s)'])
        flushfile_time = float(row['flushfile (s)'])
        make_resident_time = float(row['make_resident (s)'])
        h2d_time = float(row['h2d (s)'])
        d2h_time = float(row['d2h (s)'])

        exec_time = kernel_time + free_time - (readfile_time + flushfile_time + make_resident_time + h2d_time + d2h_time)

        if folder not in exec_time_array:
            exec_time_array[folder] = list()
            readfile_time_array[folder] = list()
            flushfile_time_array[folder] = list()
            make_resident_time_array[folder] = list()
            h2d_time_array[folder] = list()
            d2h_time_array[folder] = list()
        
        exec_time_array[folder].append(exec_time)
        readfile_time_array[folder].append(readfile_time)
        flushfile_time_array[folder].append(flushfile_time)
        make_resident_time_array[folder].append(make_resident_time)
        h2d_time_array[folder].append(h2d_time)
        d2h_time_array[folder].append(d2h_time)

x_array, exec_time_array = zip(*sorted(exec_time_array.items(), key = lambda item: item[0]))
readfile_time_array = list(zip(*sorted(readfile_time_array.items(), key = lambda item: item[0])))[1]
flushfile_time_array = list(zip(*sorted(flushfile_time_array.items(), key = lambda item: item[0])))[1]
make_resident_time_array = list(zip(*sorted(make_resident_time_array.items(), key = lambda item: item[0])))[1]
h2d_time_array = list(zip(*sorted(h2d_time_array.items(), key = lambda item: item[0])))[1]
d2h_time_array = list(zip(*sorted(d2h_time_array.items(), key = lambda item: item[0])))[1]

x_array = np.asarray(x_array) + width
exec_time_means = np.asarray([np.mean(item) for item in exec_time_array])
readfile_time_means = np.asarray([np.mean(item) for item in readfile_time_array])
flushfile_time_means = np.asarray([np.mean(item) for item in flushfile_time_array])
make_resident_time_means = np.asarray([np.mean(item) for item in make_resident_time_array])
h2d_time_means = np.asarray([np.mean(item) for item in h2d_time_array])
d2h_time_means = np.asarray([np.mean(item) for item in d2h_time_array])

plt.bar(
    x_array,
    readfile_time_means,
    width,
    yerr = [np.std(item) for item in readfile_time_array],
    color = COLORS['read'],
    edgecolor = 'k'
)
bottom = readfile_time_means

plt.bar(
    x_array,
    make_resident_time_means,
    width,
    yerr = [np.std(item) for item in make_resident_time_array],
    label = 'create pages',
    bottom = bottom,
    color = COLORS['make_resident'],
    edgecolor = 'k'
)
bottom += make_resident_time_means

plt.bar(
    x_array,
    h2d_time_means,
    width,
    yerr = [np.std(item) for item in h2d_time_array],
    bottom = bottom,
    color = COLORS['h2d_memcpy'],
    edgecolor = 'k'
)
bottom += h2d_time_means

plt.bar(
    x_array,
    d2h_time_means,
    width,
    yerr = [np.std(item) for item in d2h_time_array],
    bottom = bottom,
    color = COLORS['d2h_memcpy'],
    edgecolor = 'k'
)
bottom += d2h_time_means

plt.bar(
    x_array,
    flushfile_time_means,
    width,
    yerr = [np.std(item) for item in flushfile_time_array],
    label = 'flush',
    bottom = bottom,
    color = COLORS['flush'],
    edgecolor = 'k'
)
bottom += flushfile_time_means

plt.bar(
    x_array,
    exec_time_means,
    width,
    yerr = [np.std(item) for item in exec_time_array],
    bottom = bottom,
    color = COLORS['exec'],
    edgecolor = 'k'
)
bottom += exec_time_means


plt.xlabel("Memory footprint (GB)")
plt.ylabel("Time (s)")
plt.legend()

desc = """
    Bar #1: cudamemcpy
    Bar #2: uvm
    Bar #3: nvmgpu
    """

fig.text(0, 0, desc)

plt.title("hotspot")
plt.show()

