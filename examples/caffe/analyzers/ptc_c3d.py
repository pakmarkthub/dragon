import csv
import argparse
import os.path
import math

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

width = 0.15

BAR_NUM_FONTSIZE = 35

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
    'original': (0.8, 0.8, 0.8,),
    'uvm': (0.8, 0.8, 0.8,),
    'cpu': (0.5, 0.5, 0.5,),
    'cpu-omp': (0.5, 0.5, 0.5,),
    'dragon': (0.3, 0.3, 0.3,),
}

LABELS = {
    'original': 'Default',
    'uvm': 'UM-P',
    'cpu': 'C++ ATLAS',
    'cpu-omp': 'C++ OPENBLAS',
    'dragon': 'DRAGON',
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

    total_base_time = 60.0 #float(sum([val[0] for key, val in data['dragon'].items() if key != 'length']))

    legends = dict()

    sorted_progs = ['original', 'uvm', 'cpu', 'cpu-omp', 'dragon',]
    num_progs = len(sorted_progs)
    i = 0
    for prog in sorted_progs:
        prog_data = data_raw[prog]
        x_array = np.arange(len(prog_data)) + (i - (float(num_progs) / 2.0 - 0.5)) * width
        bottom = np.asarray([0.0,] * len(prog_data))

        y_array = np.asarray([float(y) / total_base_time for x, y in sorted(prog_data.items(), key = lambda item: item[0])])

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
            ax.text(x, y + 1, '{:.2f}'.format(y), 
                fontdict = {
                    'size': BAR_NUM_FONTSIZE,
                    'weight': 'bold',
                },
                ha = 'center',
                rotation = 90,
                va = 'bottom'
            )
        
        i += 1

    ax.set_xticks(np.arange(len(data_raw['dragon'])))
    ax.set_xticklabels(
        ['{:d}\n[{:.2f}]'.format(length, LENTH_MEMSIZE_MAP[length]) for length in sorted(data_raw['dragon'].keys())],
        fontdict = {
            'weight': 'bold',
            'size': 35,
        }
    )

    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(35)

    ax.set_ylim(top = 150)

    ax.set_xlabel("Video length (frames)\n[Memory footprint (GiB)]", size = 40, weight = 'bold')
    ax.set_ylabel("Time (mins)", size = 40, weight = 'bold')

    #ax.set_title(name, size = 20, weight = 'bold')

    return legends

def main(args):
    fig, ax = plt.subplots()
    legends = plot_prog('c3d', ax)

    sorted_progs = ['original', 'uvm', 'cpu', 'cpu-omp', 'dragon',]
    sorted_legend_objs = list()
    for prog in sorted_progs:
        sorted_legend_objs.append(legends[prog])

    ax.legend(
        sorted_legend_objs,
        [LABELS[prog] for prog in sorted_progs],
        loc = 'upper left', 
        ncol = 2,
        prop = {
            'size': 40,
            'weight': 'bold',
        }
    )

    if args.save:
        fig.set_size_inches(25, 10)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())
