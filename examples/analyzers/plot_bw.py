import csv
import argparse
import math

import matplotlib.pyplot as plt

import numpy as np

width = 0.7

def parseargs():
    parser = argparse.ArgumentParser(
        description = 'Benchmark result bandwidth plotter'
    )

    parser.add_argument(
        '--save',
        help = 'Output filename'
    )

    return parser.parse_args()

def plot_prog(name, ax):
    read_bw_array = dict()
    write_bw_array = dict()

    with open('../{}/results/result-nvmgpu.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dfolder = int(row['dfolder'][:-1])
            if dfolder < 4:
                continue
            num_sys_pages_read = int(row['num_sys_pages_read'])
            num_sys_pages_cpu_evict = int(row['num_sys_pages_cpu_evict'])
            num_sys_pages_flush = int(row['num_sys_pages_flush'])

            readfile_time = float(row['readfile (s)'])
            flushfile_time = float(row['flushfile (s)'])
            evictfile_time = float(row['evictfile (s)'])

            read_bw_array[dfolder] = float(num_sys_pages_read * (2 ** 12)) / float(10 ** 9) / readfile_time
            write_bw_array[dfolder] = float((num_sys_pages_cpu_evict + num_sys_pages_flush) * (2 ** 12)) / float(10 ** 9) / (flushfile_time + evictfile_time)

    mem_footprint, read_bw_array = zip(*sorted(read_bw_array.items(), key = lambda item: item[0]))
    write_bw_array = list(zip(*sorted(write_bw_array.items(), key = lambda item: item[0])))[1]

    x_array = range(len(mem_footprint))

    legend_array = list()

    legend_array.append((
        ax.plot(
            x_array,
            read_bw_array,
            color = 'k',
            linewidth = 4,
            linestyle = '--',
        )[0],
        'nvm-read'
    ))
    legend_array.append((
        ax.plot(
            x_array,
            write_bw_array,
            color = 'k',
            linewidth = 4,
            linestyle = ':',
        )[0],
        'nvm-write'
    ))

    ax.set_xticks(x_array)
    ax.set_xticklabels(
        mem_footprint,
        fontdict = {
            'weight': 'bold',
            'size': 15,
        }
    )

    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(15)

    ax.set_title(name, size = 20, weight = 'bold')

    return legend_array

def main(args):
    progs = ['backprop', 'binomialOptions', 'BlackScholes', 'hotspot', 'lavaMD', 'pathfinder', 'srad_v2', 'vectorAdd',]
    fig, axes = plt.subplots(2, 4)

    i = 0
    for prog in progs:
        row = int(i / 4)
        col = i % 4
        legends = plot_prog(prog, axes[row][col])
        i += 1

    fig.text(0.5, 0.05, 'Memory footprint (GiB)', ha = 'center', size = 25, weight = 'bold')
    fig.text(0.09, 0.5, 'Bandwidth (GB/s)', va = 'center', rotation = 'vertical', size = 25, weight = 'bold')

    objs, labels = list(zip(*legends))
    fig.legend(
        objs, 
        labels,
        loc = 'upper center',
        ncol = int(math.ceil(len(legends) / 1)),
        prop = {
            'size': 20,
            'weight': 'bold',
        }
    )

    if args.save:
        fig.set_size_inches(30, 10)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())

