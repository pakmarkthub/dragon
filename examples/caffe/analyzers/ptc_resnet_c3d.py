import csv
import argparse
import os.path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import numpy as np
import scipy.stats

mpl.rcParams['hatch.linewidth'] = 3.0

width = 0.17

BAR_NUM_FONTSIZE = 23

LENTH_MEMSIZE_MAP = {
    12: 4.52,
    24: 8.94,
    36: 13.36,
    48: 17.40,
    60: 21.83,
    72: 26.24,
    84: 30.68,
}


HATCHES = {
    'original': None,
    'uvm': '//',
    'cpu': None,
    'cpu-omp': '//',
    'dragon': None,
}

COLORS = {
    'original': (0.9, 0.9, 0.9,),
    'uvm': (0.9, 0.9, 0.9,),
    'cpu': (0.5, 0.5, 0.5,),
    'cpu-omp': (0.5, 0.5, 0.5,),
    'dragon': (0.0, 0.0, 0.0,),
}

LABELS = {
    'original': 'Default',
    'uvm': 'UM-P',
    'cpu': 'C++ ATLAS',
    'cpu-omp': 'C++ OPENBLAS',
    'dragon': 'DRAGON',
    'dragon exec time': 'DRAGON exec time',
}

def parseargs():
    parser = argparse.ArgumentParser(
        description = 'Benchmark result time comparison plotter for ResNet'
    )

    parser.add_argument(
        '--save',
        help = 'Output filename'
    )

    return parser.parse_args()

def plot_resnet(ax, name = 'resnet'):
    data_raw = {
        'original': dict(),
        'uvm': dict(),
        'cpu': dict(),
        'cpu-omp': dict(),
        'dragon': dict(),
    }

    memsize_map = dict()

    with open('../results/{}/memsize.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            size = int(row['size (B)'])
            memsize_map[model] = size

    for benchmark_type in ('original', 'uvm', 'cpu', 'cpu-omp', 'dragon',):
        exec_time_array = data_raw[benchmark_type]

        with open('../results/{}/result-{}.data'.format(name, benchmark_type), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row['model']
                if int(model.strip().split('_')[1]) < 32:
                    continue
                exec_time = float(row['total_time (s)'])

                assert exec_time > 0, '%s %s %d' % (name, benchmark_type, exec_time,)

                exec_time_array[model] = exec_time

    normalized_data = dict()
    for benchmark_type, time_data in data_raw.items():
        normalized_data[benchmark_type] = dict()
        for model, exec_time in time_data.items():
            normalized_data[benchmark_type][model] = exec_time / data_raw['dragon'][model]

    legends = dict()

    sorted_progs = ['original', 'uvm', 'cpu', 'cpu-omp', 'dragon',]
    num_progs = len(sorted_progs)
    i = 0
    for prog in sorted_progs:
        prog_data = normalized_data[prog]
        x_array = np.arange(len(prog_data)) + (i - (float(num_progs) / 2.0 - 0.5)) * width

        y_array = np.asarray([y for x, y in sorted(prog_data.items(), key = lambda item: int(item[0].split('_')[1]))])

        b = ax.bar(
            x_array,
            y_array,
            width * 0.9,
            label = prog,
            hatch = HATCHES[prog],
            color = COLORS[prog],
            edgecolor = 'k'
        )

        legends[prog] = b

        for x, y in zip(x_array, y_array):
            if y >= 5.0:
                ax.text(x, 5.1, '{:.2f}'.format(y), 
                    fontdict = {
                        'size': BAR_NUM_FONTSIZE,
                        'weight': 'bold',
                    },
                    ha = 'left',
                    rotation = 45,
                    va = 'bottom'
                )
        
        i += 1

    ax_right = ax.twinx()
    b = ax_right.plot(
        np.arange(len(prog_data)),
        np.asarray([y / 60.0 for x, y in sorted(data_raw['dragon'].items(), key = lambda item: int(item[0].split('_')[1]))]),
        color = 'k',
        linewidth = 5
    )

    legends['dragon exec time'] = b[0]

    ax_right.set_yscale('log')
    ax_right.yaxis.set_major_formatter(ScalarFormatter())

    for label in ax_right.get_yticklabels():
        label.set_weight('bold')
        label.set_size(25)

    ax.set_xticks(np.arange(len(data_raw['dragon'])))
    ax.set_xticklabels(
        ['{:s}\n[{:.2f}]'.format(model, memsize_map[model] / float(2 ** 30)) for model in sorted(data_raw['dragon'].keys(), key = lambda item: int(item.split('_')[1]))],
        fontdict = {
            'weight': 'bold',
            'size': 25,
        }
    )


    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(25)

    ax.set_ylim(top = 5)

    ax.set_xlabel("Model\n[Memory footprint (GiB)]", size = 30, weight = 'bold')
    #ax.set_ylabel("Normalized Time", size = 35, weight = 'bold')
    #ax_right.set_ylabel("Execution time (mins)", size = 35, weight = 'bold')

    ax.set_title('ResNet', size = 30, weight = 'bold')

    return legends

def plot_c3d(ax, name = 'c3d'):
    data_raw = {
        'original': dict(),
        'uvm': dict(),
        'cpu': dict(),
        'cpu-omp': dict(),
        'dragon': dict(),
    }

    for benchmark_type in ('original', 'uvm', 'cpu', 'cpu-omp', 'dragon',):
        exec_time_array = data_raw[benchmark_type]

        with open('../results/{}/result-{}.data'.format(name, benchmark_type), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                length = int(row['length'])
                exec_time = float(row['total_time (s)'])

                assert exec_time > 0, '%s %s %d' % (name, benchmark_type, exec_time,)

                exec_time_array[length] = exec_time

    normalized_data = dict()
    for benchmark_type, time_data in data_raw.items():
        normalized_data[benchmark_type] = dict()
        for length, exec_time in time_data.items():
            normalized_data[benchmark_type][length] = exec_time / data_raw['dragon'][length]

    legends = dict()

    sorted_progs = ['original', 'uvm', 'cpu', 'cpu-omp', 'dragon',]
    num_progs = len(sorted_progs)
    i = 0
    for prog in sorted_progs:
        prog_data = normalized_data[prog]
        x_array = np.arange(len(prog_data)) + (i - (float(num_progs) / 2.0 - 0.5)) * width

        y_array = np.asarray([y for x, y in sorted(prog_data.items(), key = lambda item: item[0])])

        b = ax.bar(
            x_array,
            y_array,
            width * 0.9,
            label = prog,
            hatch = HATCHES[prog],
            color = COLORS[prog],
            edgecolor = 'k'
        )

        legends[prog] = b

        for x, y in zip(x_array, y_array):
            if y >= 5.0:
                ax.text(
                    x - 0.1 if i == 2 else x, 
                    5.1, 
                    '{:.2f}'.format(y), 
                    fontdict = {
                        'size': BAR_NUM_FONTSIZE,
                        'weight': 'bold',
                    },
                    ha = 'left',
                    rotation = 45,
                    va = 'bottom'
                )
        
        i += 1

    ax_right = ax.twinx()
    b = ax_right.plot(
        np.arange(len(prog_data)),
        np.asarray([y / 60.0 for x, y in sorted(data_raw['dragon'].items(), key = lambda item: item[0])]),
        color = 'k',
        linewidth = 5
    )

    legends['dragon exec time'] = b[0]

    ax_right.set_yscale('log')
    ax_right.yaxis.set_major_formatter(ScalarFormatter())

    for label in ax_right.get_yticklabels():
        label.set_weight('bold')
        label.set_size(25)

    ax.set_xticks(np.arange(len(data_raw['dragon'])))
    ax.set_xticklabels(
        ['{:d}\n[{:.2f}]'.format(length, LENTH_MEMSIZE_MAP[length]) for length in sorted(data_raw['dragon'].keys())],
        fontdict = {
            'weight': 'bold',
            'size': 25,
        }
    )


    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(25)

    ax.set_ylim(top = 5)

    ax.set_xlabel("Video length\n[Memory footprint (GiB)]", size = 30, weight = 'bold')
    #ax.set_ylabel("Normalized time", size = 35, weight = 'bold')
    #ax_right.set_ylabel("Execution time (mins)", size = 35, weight = 'bold')

    ax.set_title('C3D', size = 30, weight = 'bold')

    return legends

def main(args):
    fig, axes = plt.subplots(2, 1)
    legends = plot_resnet(axes[0])
    plot_c3d(axes[1])

    sorted_progs = ['original', 'uvm', 'cpu', 'cpu-omp', 'dragon', 'dragon exec time']
    sorted_legend_objs = list()
    for prog in sorted_progs:
        sorted_legend_objs.append(legends[prog])

    fig.legend(
        sorted_legend_objs,
        [LABELS[prog] for prog in sorted_progs],
        loc = "upper center", 
        ncol = 3,
        prop = {
            'size': 25,
            'weight': 'bold',
        },
        bbox_to_anchor = (0.49, 1.15)
    )

    fig.text(0.05, 0.5, 'Normalized time', va = 'center', rotation = 'vertical', size = 40, weight = 'bold')
    fig.text(0.97, 0.5, 'Execution time (mins)', va = 'center', rotation = 270, size = 40, weight = 'bold')

    plt.subplots_adjust(hspace = 1)

    if args.save:
        fig.set_size_inches(15, 11)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())
