import csv
import argparse
import os.path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter

import numpy as np
import scipy.stats

mpl.rcParams['hatch.linewidth'] = 3.0

width = 0.17

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
    'hostreg': 'hr',
    'ap': 'ap',
    'nvmgpu': 'dg',
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
        'hostreg': {
            'map': dict(),
            'free': dict(),
            'exec': dict(),
        },
        'ap': {
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

    with open('../{}/results/result-cudamemcpy.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder >= 4:
                continue
            exec_time = float(row['kernel_time (ms)']) / 1000.0
            writefile_time = float(row['writefile_time (ms)']) / 1000.0
            readfile_time = float(row['readfile_time (ms)']) / 1000.0
            h2d_memcpy_time = float(row['h2d_memcpy_time (ms)']) / 1000.0
            d2h_memcpy_time = float(row['d2h_memcpy_time (ms)']) / 1000.0

            if folder not in exec_time_array:
                exec_time_array[folder] = list()
                writefile_time_array[folder] = list()
                readfile_time_array[folder] = list()
                gputrans_time_array[folder] = list()
            
            exec_time_array[folder].append(exec_time)
            writefile_time_array[folder].append(writefile_time)
            readfile_time_array[folder].append(readfile_time)
            gputrans_time_array[folder].append(h2d_memcpy_time + d2h_memcpy_time)


    readfile_time_array = data_raw['uvm']['nvm-read']
    writefile_time_array = data_raw['uvm']['nvm-write']
    exec_time_array = data_raw['uvm']['exec']

    with open('../{}/results/result-uvm.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder >= 4:
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


    map_time_array = data_raw['hostreg']['map']
    free_time_array = data_raw['hostreg']['free']
    exec_time_array = data_raw['hostreg']['exec']

    with open('../{}/results/result-hostreg.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder >= 4:
                continue
            exec_time = float(row['kernel_time (ms)']) / 1000.0
            map_time = float(row['map_time (ms)']) / 1000.0
            free_time = float(row['free_time (ms)']) / 1000.0

            if folder not in exec_time_array:
                exec_time_array[folder] = list()
                map_time_array[folder] = list()
                free_time_array[folder] = list()
            
            exec_time_array[folder].append(exec_time)
            map_time_array[folder].append(map_time)
            free_time_array[folder].append(free_time)

    exec_time_array = data_raw['ap']['exec']

    with open('../{}/results/result-ap.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder >= 4:
                continue
            exec_time = float(row['total_time (ms)']) / 1000.0

            if folder not in exec_time_array:
                exec_time_array[folder] = list()
            
            exec_time_array[folder].append(exec_time)


    readfile_time_array = data_raw['nvmgpu']['nvm-read']
    writefile_time_array = data_raw['nvmgpu']['nvm-write']
    gputrans_time_array = data_raw['nvmgpu']['gpu-trans']
    exec_time_array = data_raw['nvmgpu']['exec']

    with open('../{}/results/result-nvmgpu.data'.format(name), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = int(row['dfolder'][:-1])
            if folder >= 4:
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

    nvmgpu_total_time_array = np.asarray([0.0,] * len(data['nvmgpu']['datasize']))
    for time_type in sorted_time_types:
        if time_type in data['nvmgpu']:
            nvmgpu_total_time_array += data['nvmgpu'][time_type]

    total_time_array = np.asarray([0.0,] * len(data['uvm']['datasize']))
    for time_type in sorted_time_types:
        if time_type in data['uvm']:
            total_time_array += data['uvm'][time_type]

    sorted_progs = ['cudamemcpy', 'hostreg', 'uvm', 'nvmgpu', 'ap',]
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

            bottom += y_array

        b = ax.bar(
            x_array,
            bottom,
            width * 0.8,
            color = 'k',
            edgecolor = 'k'
        )
        
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

    ax.set_xticks(np.arange(len(data['nvmgpu']['datasize'])))
    ax.set_xticklabels(
        data['nvmgpu']['datasize'],
        fontdict = {
            'weight': 'bold',
            'size': 35,
        }
    )

    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(35)

    ax.set_ylim(top = ax.get_ylim()[1] * 1.2)

    #ax.set_xlabel("Memory footprint (GiB)", size = 45, weight = 'bold')
    ax.set_ylabel("Normalized time", size = 40, weight = 'bold')

    ax.set_title(name, size = 40, weight = 'bold')

def main(args):
    if args.bench:
        fig, ax = plt.subplots()
        plot_prog(args.bench, ax, True)

    else:
        progs = ['BlackScholes', 'vectorAdd',]
        fig, axes = plt.subplots(1, 2)

        i = 0
        for prog in progs:
            plot_prog(prog, axes[i])
            i += 1

        #fig.delaxes(axes[2][2])


        fig.text(.5, -0.02, 'Memory footprint (GiB)', ha = 'center', size = 40, weight = 'bold')
        #fig.text(0.07, 0.5, 'Normalized time', va = 'center', rotation = 'vertical', size = 85, weight = 'bold')
        #fig.text(0.925, 0.5, 'Execution time (mins)', va = 'center', rotation = 270, size = 85, weight = 'bold')


    if args.save:
        fig.set_size_inches(20, 7)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())
