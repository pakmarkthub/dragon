import csv
import argparse
import math
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter

import numpy as np
import scipy.stats

mpl.rcParams['hatch.linewidth'] = 3.0
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

width = 0.23

BAR_NUM_FONTSIZE = 30

HATCHES = {
    'nvm-read': None,
    'nvm-write': '//',
    'map': 'x',
    'free': '\\\\',
    'gpu-trans': None,
    'exec': None,
}

COLORS = {
    'nvm-read': (0.8, 0.8, 0.8,),
    'nvm-write': (0.8, 0.8, 0.8,),
    'map': (0.6, 0.6, 0.6,),
    'free': (0.6, 0.6, 0.6,),
    'gpu-trans': (0.4, 0.4, 0.4,),
    'exec': (0.0, 0.0, 0.0),
}

ABBR_MAP = {
    'cudamemcpy': 'df',
    'uvm': 'um',
    'nvmgpu': 'dg',
}

def parseargs():
    parser = argparse.ArgumentParser(
        description = 'Benchmark result iteration plotter'
    )

    parser.add_argument(
        '--save',
        help = 'Output filename'
    )

    parser.add_argument(
        'folder',
        help = 'Path to the folder that contains the results'
    )

    return parser.parse_args()

def plot_data(args, size, ax, use_logx = False):
    data_raw = {
        'cudamemcpy': {
            'nvm-read': dict(),
            'nvm-write': dict(),
            'gpu-trans': dict(),
            'exec': dict(),
        },
        'uvm': {
            'nvm-read': dict(),
            'nvm-write': dict(),
            'exec': dict(),
        },
        'nvmgpu': {
            'nvm-read': dict(),
            'nvm-write': dict(),
            'gpu-trans': dict(),
            'exec': dict(),
        }
    }

    readfile_time_array = data_raw['cudamemcpy']['nvm-read']
    writefile_time_array = data_raw['cudamemcpy']['nvm-write']
    gputrans_time_array = data_raw['cudamemcpy']['gpu-trans']
    exec_time_array = data_raw['cudamemcpy']['exec']

    filepath = os.path.join(args.folder, 'result-cudamemcpy.data')
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder != size:
                continue

            total_iters = int(row['total_iters'])
            exec_time = float(row['kernel_time (ms)']) / 1000.0
            writefile_time = float(row['writefile_time (ms)']) / 1000.0
            readfile_time = float(row['readfile_time (ms)']) / 1000.0
            h2d_memcpy_time = float(row['h2d_memcpy_time (ms)']) / 1000.0
            d2h_memcpy_time = float(row['d2h_memcpy_time (ms)']) / 1000.0

            exec_time_array[total_iters] = exec_time
            writefile_time_array[total_iters] = writefile_time
            readfile_time_array[total_iters] = readfile_time
            gputrans_time_array[total_iters] = h2d_memcpy_time + d2h_memcpy_time


    readfile_time_array = data_raw['uvm']['nvm-read']
    writefile_time_array = data_raw['uvm']['nvm-write']
    exec_time_array = data_raw['uvm']['exec']

    filepath = os.path.join(args.folder, 'result-uvm.data')
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder != size:
                continue

            total_iters = int(row['total_iters'])
            exec_time = float(row['kernel_time (ms)']) / 1000.0
            writefile_time = float(row['writefile_time (ms)']) / 1000.0
            readfile_time = float(row['readfile_time (ms)']) / 1000.0

            exec_time_array[total_iters] = exec_time
            writefile_time_array[total_iters] = writefile_time
            readfile_time_array[total_iters] = readfile_time


    readfile_time_array = data_raw['nvmgpu']['nvm-read']
    writefile_time_array = data_raw['nvmgpu']['nvm-write']
    gputrans_time_array = data_raw['nvmgpu']['gpu-trans']
    exec_time_array = data_raw['nvmgpu']['exec']

    filepath = os.path.join(args.folder, 'result-nvmgpu.data')
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder != size:
                continue

            total_iters = int(row['total_iters'])
            kernel_time = float(row['kernel_time (ms)']) / 1000.0
            map_time = float(row['map_time (ms)']) / 1000.0
            free_time = float(row['free_time (ms)']) / 1000.0
            readfile_time = float(row['readfile (s)'])
            flushfile_time = float(row['flushfile (s)'])
            evictfile_time = float(row.get('evictfile (s)', 0))
            aggrwrite_time = float(row.get('aggrwrite (s)', 0))
            make_resident_time = float(row['make_resident (s)'])
            h2d_time = float(row['h2d (s)'])
            d2h_time = float(row['d2h (s)'])

            d2h_time += make_resident_time - aggrwrite_time
            writefile_time = flushfile_time + evictfile_time + aggrwrite_time

            exec_time = kernel_time + map_time + free_time - (readfile_time + writefile_time + h2d_time + d2h_time)

            assert exec_time >= 0, "%s: %d %f" % (name, folder, exec_time,)

            exec_time_array[total_iters] = exec_time
            readfile_time_array[total_iters] = readfile_time
            writefile_time_array[total_iters] = writefile_time
            gputrans_time_array[total_iters] = h2d_time + d2h_time


    data = dict()
    for prog, items in data_raw.items():
        data[prog] = dict()
        for key, item in items.items():
            if len(data[prog]) == 0:
                if len(item) > 0:
                    iter_array, time_array = zip(*sorted(item.items(), key = lambda t: t[0]))
                else:
                    iter_array = list()
                data[prog]['iters'] = iter_array
            if len(item) > 0:
                data[prog][key] = np.asarray([time for time in list(zip(*sorted(item.items(), key = lambda t: t[0])))[1]])
            else:
                data[prog][key] = list()

    sorted_time_types = ['nvm-read', 'nvm-write', 'gpu-trans', 'exec',]

    nvmgpu_total_time_array = np.asarray([0.0,] * len(data['nvmgpu']['iters']))
    for time_type in sorted_time_types:
        if time_type in data['nvmgpu']:
            nvmgpu_total_time_array += data['nvmgpu'][time_type]

    total_time_array = np.asarray([0.0,] * len(data['uvm']['iters']))
    for time_type in sorted_time_types:
        if time_type in data['uvm']:
            total_time_array += data['uvm'][time_type]

    legends = dict()

    sorted_progs = ['cudamemcpy', 'uvm', 'nvmgpu',]
    num_progs = len(sorted_progs)
    i = 0
    for prog in sorted_progs:
        prog_data = data[prog]
        x_array = np.arange(len(prog_data['iters'])) + (i - (float(num_progs) / 2.0 - 0.5)) * width
        bottom = np.asarray([0.0,] * len(prog_data['iters']))

        for time_type in sorted_time_types:
            if time_type not in prog_data:
                continue
            y_array = np.asarray([float(y) / float(t) for y, t in zip(prog_data[time_type], total_time_array)])

            b = ax.bar(
                x_array,
                y_array,
                width * 0.8,
                bottom = bottom,
                label = time_type,
                hatch = HATCHES[time_type],
                color = COLORS[time_type],
                edgecolor = 'k'
            )

            bottom += y_array

            if time_type not in legends:
                legends[time_type] = b

        for x, y in zip(x_array, bottom):
            y_pos = y + 0.02
            ax.text(x, y_pos, '{}-{}'.format(i + 1, ABBR_MAP[sorted_progs[i]]), 
                fontdict = {
                    'size': BAR_NUM_FONTSIZE,
                    'weight': 'bold',
                },
                ha = 'center',
                rotation = 'vertical',
                va = 'bottom'
            )
        
        i += 1

    if use_logx:
        labels = ["$\\mathbf{10^{%d}}$" % (math.log(x),) for x in data['nvmgpu']['iters']]
    else:
        labels = data['nvmgpu']['iters']

    ax.set_xticks(np.arange(len(data['nvmgpu']['iters'])))
    ax.set_xticklabels(
        labels,
        fontdict = {
            'weight': 'bold',
            'size': 35,
        }
    )

    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(35)

    ax.set_ylim(top = ax.get_ylim()[1] * 1.2)

    ax.set_xlabel("Num Iterations", size = 40, weight = 'bold')
    ax.set_ylabel("Normalized time", size = 40, weight = 'bold')

    ax.set_title("{} GiB".format(size), size = 40, weight = 'bold')

    return legends

def main(args):
    fig, axes = plt.subplots(1, 2)

    legends = plot_data(args, 8, axes[0], True)
    plot_data(args, 64, axes[1])

    sorted_time_types = ['nvm-read', 'nvm-write', 'gpu-trans', 'exec',]
    sorted_legend_objs = list()
    for time_types in sorted_time_types:
        sorted_legend_objs.append(legends[time_types])

    plt.legend(
        sorted_legend_objs,
        sorted_time_types,
        loc = 'upper center', 
        bbox_to_anchor = (-0.15, 1.2),
        ncol = len(legends),
        prop = {
            'size': 45,
            'weight': 'bold',
        },
        edgecolor = 'k',
        fancybox = True
    )

    if args.save:
        fig.set_size_inches(30, 15)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())

