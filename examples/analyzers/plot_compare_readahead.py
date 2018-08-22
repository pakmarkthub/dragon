import csv
import argparse
import os.path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import numpy as np
import scipy.stats

#mpl.rcParams['hatch.linewidth'] = 3.0

width = 0.25

BAR_NUM_FONTSIZE = 25

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
    'uvm': 'um',
    'nvmgpu': 'dg',
    'nvmgpu_rh_dis': 'no-rh',
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
        '--bench',
        help = 'Plot the result of this benchmark'
    )

    return parser.parse_args()

def plot_prog(name, ax):
    data_raw = {
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
        },
        'nvmgpu_rh_dis': {
            'nvm-read': dict(),
            'nvm-write': dict(),
            'gpu-trans': dict(),
            'exec': dict(),
        }
    }

    readfile_time_array = data_raw['uvm']['nvm-read']
    writefile_time_array = data_raw['uvm']['nvm-write']
    exec_time_array = data_raw['uvm']['exec']

    with open('../{}/results/result-uvm.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder not in (8, 64, 256,):
                continue
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


    readfile_time_array = data_raw['nvmgpu']['nvm-read']
    writefile_time_array = data_raw['nvmgpu']['nvm-write']
    gputrans_time_array = data_raw['nvmgpu']['gpu-trans']
    exec_time_array = data_raw['nvmgpu']['exec']

    with open('../{}/results/result-nvmgpu.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder not in (8, 64, 256,):
                continue
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

            if folder not in exec_time_array:
                exec_time_array[folder] = list()
                readfile_time_array[folder] = list()
                writefile_time_array[folder] = list()
                gputrans_time_array[folder] = list()
            
            exec_time_array[folder].append(exec_time)
            readfile_time_array[folder].append(readfile_time)
            writefile_time_array[folder].append(writefile_time)
            gputrans_time_array[folder].append(h2d_time + d2h_time)


    readfile_time_array = data_raw['nvmgpu_rh_dis']['nvm-read']
    writefile_time_array = data_raw['nvmgpu_rh_dis']['nvm-write']
    gputrans_time_array = data_raw['nvmgpu_rh_dis']['gpu-trans']
    exec_time_array = data_raw['nvmgpu_rh_dis']['exec']

    with open('../{}/results/result-nvmgpu-rh-disable.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder not in (8, 64, 256,):
                continue
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

            if folder not in exec_time_array:
                exec_time_array[folder] = list()
                readfile_time_array[folder] = list()
                writefile_time_array[folder] = list()
                gputrans_time_array[folder] = list()
            
            exec_time_array[folder].append(exec_time)
            readfile_time_array[folder].append(readfile_time)
            writefile_time_array[folder].append(writefile_time)
            gputrans_time_array[folder].append(h2d_time + d2h_time)


    data = dict()
    for prog, items in data_raw.items():
        data[prog] = dict()
        for key, item in items.items():
            if len(data[prog]) == 0:
                datasize_array, time_array = zip(*sorted(item.items(), key = lambda t: t[0]))
                data[prog]['datasize'] = datasize_array
            data[prog][key] = np.asarray([np.mean(time) for time in list(zip(*sorted(item.items(), key = lambda t: t[0])))[1]])

    sorted_time_types = ['nvm-read', 'nvm-write', 'map', 'free', 'gpu-trans', 'exec',]

    total_time_array = np.asarray([0.0,] * len(data['uvm']['datasize']))
    for time_type in sorted_time_types:
        if time_type in data['uvm']:
            total_time_array += data['uvm'][time_type]

    regress_result = scipy.stats.linregress(data['uvm']['datasize'], total_time_array)
    i = len(total_time_array)
    extended_time_array = list()
    estimated_x_array = list()
    while i < len(data['nvmgpu']['datasize']):
        extended_time_array.append(data['nvmgpu']['datasize'][i] * regress_result.slope + regress_result.intercept)
        estimated_x_array.append(i - width)
        i += 1
    total_time_array = np.append(total_time_array, extended_time_array)

    legends = dict()

    sorted_progs = ['uvm', 'nvmgpu', 'nvmgpu_rh_dis']
    num_progs = len(sorted_progs)
    i = 0
    for prog in sorted_progs:
        prog_data = data[prog]
        x_array = np.arange(len(prog_data['datasize'])) + (i - (float(num_progs) / 2.0 - 0.5)) * width
        bottom = np.asarray([0.0,] * len(prog_data['datasize']))

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
            ax.text(x, y + 0.02, '{}-{}'.format(i + 1, ABBR_MAP[sorted_progs[i]]), 
                fontdict = {
                    'size': BAR_NUM_FONTSIZE,
                    'weight': 'bold',
                },
                ha = 'center',
                rotation = 'vertical',
                va = 'bottom'
            )
        
        i += 1

    b = ax.bar(
        estimated_x_array,
        [1.0,] * len(extended_time_array),
        width * 0.8,
        color = 'w',
        linestyle = '--',
        linewidth = 2,
        edgecolor = 'k',
    )

    legends['estimated'] = b

    for x in estimated_x_array:
        ax.text(x, 1.02, '1-{}'.format(ABBR_MAP['uvm']), 
            fontdict = {
                'size': BAR_NUM_FONTSIZE,
                'weight': 'bold',
            },
            ha = 'center',
            rotation = 'vertical',
            va = 'bottom'
        )

    ax.set_xticks(np.arange(len(data['nvmgpu']['datasize'])))
    ax.set_xticklabels(
        data['nvmgpu']['datasize'],
        fontdict = {
            'weight': 'bold',
            'size': 25,
        }
    )

    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(25)

    ax.set_ylim(top = 15)

    ax.set_ylabel("Normalized time", size = 30, weight = 'bold')
    ax.set_title(name, size = 30, weight = 'bold')

    return legends

def main(args):
    if args.bench:
        fig, ax = plt.subplots()
        legends = plot_prog(args.bench, ax)

    else:
        progs = ['hotspot', 'vectorAdd',]
        fig, axes = plt.subplots(1, 2)

        i = 0
        for prog in progs:
            legends = plot_prog(prog, axes[i])
            i += 1

        fig.text(0.5, -0.05, 'Memory footprint (GiB)', ha = 'center', size = 30, weight = 'bold')
        #fig.text(0.06, 0.5, 'Normalized time', va = 'center', rotation = 'vertical', size = 30, weight = 'bold')

    sorted_time_types = ['nvm-read', 'nvm-write', 'gpu-trans', 'exec', 'estimated',]
    sorted_legend_objs = list()
    for time_types in sorted_time_types:
        sorted_legend_objs.append(legends[time_types])

    fig.legend(
        sorted_legend_objs,
        sorted_time_types,
        bbox_to_anchor = (0.77, 1.3),
        ncol = int(math.ceil(len(sorted_time_types) / 2.0)),
        prop = {
            'size': 25,
            'weight': 'bold',
        }
    )

    if args.save:
        fig.set_size_inches(17, 6)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())
