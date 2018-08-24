import argparse
import os.path
import math

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

width = 0.1

BAR_NUM_FONTSIZE = 15

HATCHES = (None, '//', '\\\\',)

COLORS = (
    (0.8, 0.8, 0.8,),
    (0.5, 0.5, 0.5,),
    (0.3, 0.3, 0.3,),
)

def parseargs():
    parser = argparse.ArgumentParser(
        description = 'Benchmark result time comparison plotter'
    )

    parser.add_argument(
        '--save',
        help = 'Output filename'
    )

    return parser.parse_args()

def plot(ax):
    #data_raw = {
    #    256: (
    #        128.0,
    #        73.334,
    #        87.363,
    #        132.658,
    #        121.248,
    #        111.421,
    #        94.265,
    #    ),
    #    512: (
    #        235.0,
    #        129.480,
    #        155.648,
    #        253.469,
    #        221.348,
    #        195.853,
    #        168.631,
    #    )
    #}

    data_raw = {
        256: (
            128.0,
            72.255,
            107.855,
            89.72,
            139.681,
            135.93,
            117.522,
            110.418,
            96.088,
        ),
        512: (
            235.0,
            130.244,
            204.47,
            154.829,
            255.909,
            240.833,
            215.838,
            198.767,
            170.563,
        )
    }

    i = 0
    for batch_size, items in data_raw.items():
        legends = list()
        x_array = [i + ((j - (float(len(items)) / 2.0 - 0.5))) * width for j in range(len(items))]
        y_array = np.asarray(items) / items[3]

        j = 0
        for x, y in zip(x_array, y_array):
            b = ax.bar(
                x,
                y,
                width,
                edgecolor = 'k',
                color = COLORS[int(j / len(HATCHES)) % len(COLORS)],
                hatch = HATCHES[j % len(HATCHES)]
            )

            legends.append(b)

            ax.text(x, y + 0.05, '{:.2f}'.format(y), 
                fontdict = {
                    'size': BAR_NUM_FONTSIZE,
                    'weight': 'bold',
                },
                ha = 'center',
                rotation = 90,
                va = 'bottom'
            )

            j += 1

        i += 1

    ax.set_xticks(np.arange(len(data_raw)))
    ax.set_xticklabels(
        data_raw.keys(),
        fontdict = {
            'weight': 'bold',
            'size': 15,
        }
    )


    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(15)

    ax.set_ylim(top = 2)

    ax.set_xlabel("Batch size", size = 20, weight = 'bold')
    ax.set_ylabel("Normalized time", size = 20, weight = 'bold')

    #ax.set_title(name, size = 20, weight = 'bold')

    return legends

def main(args):
    fig, ax = plt.subplots()
    legends = plot(ax)

    fig.legend(
        legends,
        #['original', 'fread', 'uvm', 'dragon', 'dragon aggr', 'dragon populate', 'dragon populate aggr',],
        ['original', 'fread batch', 'fread all', 'uvm batch', 'uvm all', 'dragon', 'dragon aggr', 'dragon populate', 'dragon populate aggr',],
        loc = 'upper center',
        ncol = int(math.ceil(len(legends) / 2.0)),
        prop = {
            'size': 20,
            'weight': 'bold',
        }
    )


    if args.save:
        fig.set_size_inches(20, 10)
        plt.savefig(args.save, dpi = 200, bbox_inches = 'tight')
    else:
        plt.show()

if __name__ == '__main__':
    main(parseargs())
