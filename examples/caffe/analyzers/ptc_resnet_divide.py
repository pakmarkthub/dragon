import csv
import argparse
import os.path
import math

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

width = 0.15

BAR_NUM_FONTSIZE = 25

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
        description = 'Benchmark result time comparison plotter for ResNet'
    )

    parser.add_argument(
        '--save',
        help = 'Output filename'
    )

    return parser.parse_args()

def plot_prog(name, ax, size_min, size_max, ylim):
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
            if size_min <= size <= size_max:
                memsize_map[model] = size

    for benchmark_type in ('original', 'uvm', 'cpu', 'cpu-omp', 'dragon',):
        exec_time_array = data_raw[benchmark_type]

        with open('../results/{}/result-{}.data'.format(name, benchmark_type), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row['model']
                if model not in memsize_map:
                    continue

                exec_time = float(row['total_time (s)'])

                assert exec_time > 0, '%s %s %d' % (name, benchmark_type, exec_time,)

                exec_time_array[model] = exec_time

    total_base_time = 60.0 #float(sum([val[0] for key, val in data['dragon'].items() if key != 'length']))

    legends = dict()

    sorted_progs = ['original', 'uvm', 'cpu', 'cpu-omp', 'dragon',]
    num_progs = len(sorted_progs)
    i = 0
    for prog in sorted_progs:
        prog_data = data_raw[prog]
        x_array = np.arange(len(prog_data)) + (i - (float(num_progs) / 2.0 - 0.5)) * width

        y_array = np.asarray([float(y) / total_base_time for x, y in sorted(prog_data.items(), key = lambda item: int(item[0].split('_')[1]))])

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
            ax.text(x, min(y, ylim) + (ylim * 0.01), '{:.2f}'.format(y), 
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
        ['{:s}\n[{:.2f}]'.format(model, memsize_map[model] / float(2 ** 30)) for model in sorted(data_raw['dragon'].keys(), key = lambda item: int(item.split('_')[1]))],
        fontdict = {
            'weight': 'bold',
            'size': 25,
        }
    )


    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(25)

    ax.set_ylim(top = ylim)
    ax.set_xlim(-0.5, 1.5)


    #ax.set_title(name, size = 20, weight = 'bold')

    return legends

def main(args):
    fig, axes = plt.subplots(1, 3)
    legends = plot_prog('resnet', axes[0], 0, 12 * (2 ** 30), 1)
    plot_prog('resnet', axes[1], 12 * (2 ** 30), 24 * (2 ** 30), 40)
    plot_prog('resnet', axes[2], 24 * (2 ** 30), 256 * (2 ** 30), 500)

    sorted_progs = ['original', 'uvm', 'cpu', 'cpu-omp', 'dragon',]
    sorted_legend_objs = list()
    for prog in sorted_progs:
        sorted_legend_objs.append(legends[prog])

    fig.text(0.5, -0.2, "Model\n[Memory footprint (GiB)]", ha = 'center', size = 35, weight = 'bold')
    axes[0].set_ylabel("Time (mins)", size = 35, weight = 'bold')

    axes[0].legend(
        sorted_legend_objs,
        [LABELS[prog] for prog in sorted_progs],
        loc = 'upper center', 
        ncol = 3,
        prop = {
            'size': 25,
            'weight': 'bold',
        },
        bbox_to_anchor = (1.75, 1.5)
    )

    if args.save:
        fig.set_size_inches(20, 7)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())
