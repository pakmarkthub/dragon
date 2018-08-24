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

width = 0.15

BAR_NUM_FONTSIZE = 25

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
    'dragon': '\\\\',
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
        description = 'Benchmark result time comparison plotter for C3D'
    )

    parser.add_argument(
        '--save',
        help = 'Output filename'
    )

    return parser.parse_args()

def plot_prog(name, ax):
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
            width,
            label = prog,
            hatch = HATCHES[prog],
            color = COLORS[prog],
            edgecolor = 'k'
        )

        legends[prog] = b

        for x, y in zip(x_array, y_array):
            if y >= 5.0:
                ax.text(x, min(y + 0.1, 5.1), '{:.2f}'.format(y), 
                    fontdict = {
                        'size': BAR_NUM_FONTSIZE,
                        'weight': 'bold',
                    },
                    ha = 'center',
                    rotation = 90,
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

    ax.set_xlabel("Video length (frames)\n[Memory footprint (GiB)]", size = 35, weight = 'bold')
    ax.set_ylabel("Normalized time", size = 35, weight = 'bold')
    ax_right.set_ylabel("Execution time (mins)", size = 35, weight = 'bold')

    #ax.set_title(name, size = 20, weight = 'bold')

    return legends

def main(args):
    fig, ax = plt.subplots()
    legends = plot_prog('c3d', ax)

    sorted_progs = ['original', 'uvm', 'cpu', 'cpu-omp', 'dragon', 'dragon exec time']
    sorted_legend_objs = list()
    for prog in sorted_progs:
        sorted_legend_objs.append(legends[prog])

    ax.legend(
        sorted_legend_objs,
        [LABELS[prog] for prog in sorted_progs],
        loc = "upper center", 
        ncol = 3,
        prop = {
            'size': 25,
            'weight': 'bold',
        },
        bbox_to_anchor = (0.5, 1.5)
    )

    if args.save:
        fig.set_size_inches(21, 7)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())
