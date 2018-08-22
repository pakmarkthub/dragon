import csv
import argparse
import os.path
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import numpy as np
import scipy.stats

LINESTYLES = {
    #64: '-',
    128: '-',
    256: '--',
}


def parseargs():
    parser = argparse.ArgumentParser(
        description = 'Benchmark result time comparison plotter'
    )

    parser.add_argument(
        '--save',
        help = 'Output filename'
    )

    parser.add_argument(
        '--title',
        help = 'Graph title'
    )

    parser.add_argument(
        'rfile',
        help = 'Result file',
        type = argparse.FileType('r')
    )

    return parser.parse_args()

def plot_prog(args):
    data = dict()

    reader = csv.DictReader(args.rfile)
    for row in reader:
        folder = int(row['dfolder'][:-1])
        if folder < 128:
            continue
        kernel_time = float(row['kernel_time (ms)']) / 1000.0
        map_time = float(row['map_time (ms)']) / 1000.0
        free_time = float(row['free_time (ms)']) / 1000.0
        nrpages = int(row['nrpages'])

        if nrpages <= (2 ** 20):
            continue

        if folder not in data:
            data[folder] = dict()

        reserved_size = nrpages * (2 ** 12) / (2 ** 30)
        
        data[folder][reserved_size] = kernel_time + map_time + free_time

    for memsize, items in sorted(data.items(), key = lambda x: x[0]):
        x, y = zip(*sorted(items.items(), key = lambda item: item[0]))
        y = np.asarray(y) / y[0]
        plt.plot(
            x, y,
            linestyle = LINESTYLES[memsize],
            color = 'k',
            linewidth = 2,
            marker = 'o',
            label = '{} GiB'.format(memsize)
        )

    for label in plt.xticks()[1]:
        label.set_weight('bold')
        label.set_size(15)

    for label in plt.yticks()[1]:
        label.set_weight('bold')
        label.set_size(15)

    plt.legend(
        loc = 'upper left',
        prop = {
            'size': 15,
            'weight': 'bold',
        }
    )

    plt.xlabel("Reserved memory threshold (GiB)", size = 20, weight = 'bold')
    plt.ylabel("Normalized time", size = 20, weight = 'bold')

    if args.title:
        plt.text(
            plt.xlim()[1] / 2.0, 
            plt.ylim()[1] * 0.95, 
            args.title,
            ha = 'center',
            size = 20,
            weight = 'bold'
        )


def main(args):
    plt.figure(figsize = (10, 3))
    plot_prog(args)
    if args.save:
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())
