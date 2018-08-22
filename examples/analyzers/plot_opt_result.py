import csv
import argparse
import os.path
import math
import pprint

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

width = 0.15

BAR_NUM_FONTSIZE = 17

HATCHES = {
    'nvm-read': None,
    'nvm-write': '//',
    'gpu-trans': '\\\\',
    'exec': None,
}

COLORS = {
    'nvm-read': (0.8, 0.8, 0.8,),
    'nvm-write': (0.8, 0.8, 0.8,),
    'gpu-trans': (0.3, 0.3, 0.3,),
    'exec': (0.1, 0.1, 0.1),
}

OPTS = ['readcache', 'aioread', 'lazywrite', 'aiowrite', 'aggr',]

def parseargs():
    parser = argparse.ArgumentParser(
        description = 'Sensitivity param result plotter'
    )

    parser.add_argument(
        '--save',
        help = 'Output filename'
    )

    return parser.parse_args()

def plot_data(name, ax, ax2):
    data_raw = dict()

    with open('../{}/results/result-opt-params.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            dfolder = row['dfolder']
            if dfolder not in data_raw:
                data_raw[dfolder] = dict()

            opt_id = 0
            for opt in OPTS:
                if row[opt] == 'True':
                    opt_id += 1

            if opt_id == 1:
                continue
            
            if opt_id not in data_raw[dfolder]:
                data_raw[dfolder][opt_id] = {
                    'nvm-read': list(),
                    'nvm-write': list(),
                    'gpu-trans': list(),
                    'exec': list(),
                }
            data_raw_inner = data_raw[dfolder][opt_id]

            kernel_time = float(row['kernel_time (ms)']) / 1000.0
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

            exec_time = kernel_time + free_time - (readfile_time + writefile_time + h2d_time + d2h_time)

            if exec_time < 0:
                d2h_time += exec_time
                exec_time = 0

            data_raw_inner['nvm-read'].append(readfile_time)
            data_raw_inner['nvm-write'].append(writefile_time)
            data_raw_inner['gpu-trans'].append(h2d_time + d2h_time)
            data_raw_inner['exec'].append(exec_time)

    #pprint.pprint(data_raw)

    legends = dict()
    datasize_array = list()
    x_base = 1
    max_bottom_height = 0
    min_top_height = 1
    for datasize, data in sorted(data_raw.items(), key = lambda item: int(item[0][:-1])):
        based_total_time = 0
        for t in data[0].values():
            based_total_time += np.mean(t)

        num_bars = len(data)
        i = 0
        for bar_id, time_dict in sorted(data.items(), key = lambda item: item[0]):
            x = x_base + (i - num_bars / 2.0) * width + width / 2.0
            bottom = 0
            for time_type in ['nvm-read', 'nvm-write', 'gpu-trans', 'exec',]:
                y = np.mean(time_dict[time_type]) / based_total_time
                b = ax.bar(
                    x,
                    y,
                    width,
                    bottom = bottom,
                    label = time_type,
                    hatch = HATCHES[time_type],
                    color = COLORS[time_type],
                    edgecolor = 'k'
                )
                ax2.bar(
                    x,
                    y,
                    width,
                    bottom = bottom,
                    label = time_type,
                    hatch = HATCHES[time_type],
                    color = COLORS[time_type],
                    edgecolor = 'k'
                )
                bottom += y 
                if time_type not in legends:
                    legends[time_type] = b
                if time_type == 'nvm-read' and i <= 1:
                    min_top_height = min(min_top_height, bottom)

            if i <= 1:
                axis = ax 
            else:
                axis = ax2
                max_bottom_height = max(bottom, max_bottom_height)
            axis.text(x, bottom + 0.003, '#{}'.format(i + 1), 
                fontdict = {
                    'size': BAR_NUM_FONTSIZE,
                    'weight': 'bold',
                },
                ha = 'center',
                rotation = 'vertical',
                va = 'bottom'
            )

            i += 1
            
        x_base += 1
        datasize_array.append(datasize)

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax2.xaxis.tick_bottom()
    ax.tick_params(labeltop = 'off')

    ax.set_ylim(min_top_height - 0.005, 1.09)
    ax2.set_ylim(0, max_bottom_height + 0.005)

    ax.set_xticks(range(1, x_base))
    ax2.set_xticklabels(
        datasize_array,
        fontdict = {
            'weight': 'bold',
            'size': 15,
        }
    )

    #ax.set_yticks([min_top_height - 0.01,] + list(ax.get_yticks()))
    for axis in [ax, ax2,]:
        for label in axis.get_yticklabels():
            label.set_weight('bold')
            label.set_size(15)

    ax.set_title(name, size = 20, weight = 'bold')

    d = 0.015
    kwargs = dict(transform = ax.transAxes, color = 'k', clip_on = False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform = ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    return legends

def main(args):
    progs = ['backprop', 'vectorAdd',]

    fig = plt.figure()

    axes = list()
    for i in range(len(progs)):
        axes.append(fig.add_subplot(2, len(progs), i + 1))
    for i in range(len(progs)):
        axes.append(fig.add_subplot(2, len(progs), i + 1 + len(progs), sharex = axes[i]))

    fig.subplots_adjust(hspace = 0.05)

    i = 0
    for prog in progs:
        legends = plot_data(prog, axes[i], axes[i + len(progs)])
        i += 1

    sorted_legend_labels = ['nvm-read', 'nvm-write', 'gpu-trans', 'exec',]
    fig.legend(
        [legends[time_type] for time_type in sorted_legend_labels], 
        sorted_legend_labels,
        loc = 'upper center', 
        ncol = len(sorted_legend_labels),
        prop = {
            'size': 20,
            'weight': 'bold',
        }
    )

    fig.text(0.5, 0.03, 'Memory footprint (GiB)', ha = 'center', size = 25, weight = 'bold')
    fig.text(0.03, 0.5, 'Normalized time', va = 'center', rotation = 'vertical', size = 25, weight = 'bold')

    if args.save:
        fig.set_size_inches(13, 8.5)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())
