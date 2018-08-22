import csv
import argparse
import math

import matplotlib.pyplot as plt

import numpy as np

width = 0.7

def parseargs():
    parser = argparse.ArgumentParser(
        description = 'Benchmark result evictions plotter'
    )

    parser.add_argument(
        '--save',
        help = 'Output filename'
    )

    return parser.parse_args()

def plot_prog(name, ax):
    evict_pages_array = dict()
    read_pages_array = dict()
    write_pages_array = dict()
    input_pages_array = dict()
    output_pages_array = dict()

    with open('../{}/results/result-nvmgpu.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dfolder = int(row['dfolder'][:-1])
            if dfolder < 4:
                continue
            num_sys_pages_gpu_evict = int(row['num_sys_pages_gpu_evict'])
            num_sys_pages_read = int(row['num_sys_pages_read'])
            num_sys_pages_cpu_evict = int(row['num_sys_pages_cpu_evict'])
            num_sys_pages_flush = int(row['num_sys_pages_flush'])

            input_pages = int(row['input_data_size (B)']) / float(2 ** 12)
            output_pages = int(row['output_data_size (B)']) / float(2 ** 12)

            evict_pages_array[dfolder] = num_sys_pages_gpu_evict * 512
            read_pages_array[dfolder] = num_sys_pages_read
            write_pages_array[dfolder] = num_sys_pages_cpu_evict + num_sys_pages_flush

            input_pages_array[dfolder] = input_pages
            output_pages_array[dfolder] = output_pages
            
    mem_footprint, evict_pages_array = zip(*sorted(evict_pages_array.items(), key = lambda item: item[0]))
    read_pages_array = list(zip(*sorted(read_pages_array.items(), key = lambda item: item[0])))[1]
    write_pages_array = list(zip(*sorted(write_pages_array.items(), key = lambda item: item[0])))[1]

    input_pages_array = list(zip(*sorted(input_pages_array.items(), key = lambda item: item[0])))[1]
    output_pages_array = list(zip(*sorted(output_pages_array.items(), key = lambda item: item[0])))[1]

    x_array = range(len(mem_footprint))

    legend_array = list()

    legend_array.append((
        ax.plot(
            x_array,
            input_pages_array,
            color = (0.5, 0.5, 0.5),
            linewidth = 4,
            marker = 'D',
            markersize = 10
        )[0],
        'nvm-read (ideal)'
    ))
    legend_array.append((
        ax.plot(
            x_array, 
            read_pages_array,
            color = 'k',
            linewidth = 4,
            linestyle = '-.',
        )[0],
        'nvm-read (measured)',
    ))
    legend_array.append((
        ax.plot(
            x_array,
            output_pages_array,
            color = (0.5, 0.5, 0.5),
            linewidth = 4,
            marker = 'o',
            markersize = 10
        )[0],
        'nvm-write (ideal)'
    ))
    legend_array.append((
        ax.plot(
            x_array, 
            write_pages_array,
            color = 'k',
            linewidth = 4,
            linestyle = ':',
        )[0],
        'nvm-write (measured)',
    ))
    legend_array.append((
        ax.plot(
            x_array, 
            evict_pages_array,
            color = 'k',
            linewidth = 4,
        )[0],
        'gpu-evict (measured)',
    ))

    ax.set_yscale('log')

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
    fig.text(0.09, 0.5, 'Num 4KiB pages', va = 'center', rotation = 'vertical', size = 25, weight = 'bold')

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

