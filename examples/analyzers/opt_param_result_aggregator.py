import csv
import argparse
import os.path

def type_folder(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError('The specified path "{}" does not exist'.format(path))
    return path

def parseargs():
    parser = argparse.ArgumentParser(
        description = 'Sensitivity Benchmark Result Converter'
    )

    parser.add_argument(
        'rfolder',
        type = type_folder,
        help = 'Result folder'
    )

    parser.add_argument(
        'ofile',
        type = argparse.FileType('w'),
        help = 'Output aggregated result file'
    )

    return parser.parse_args()

def main(args):
    writer = csv.writer(args.ofile)
    writer.writerow([
        'step', 'prog', 'dfolder', 'retcode',
        'readcache', 'aioread', 'lazywrite', 'aiowrite', 'aggr', 'dis',
        'kernel_time (ms)', 'free_time (ms)', 'map_time (ms)',
        'readfile (s)', 'flushfile (s)', 'evictfile (s)', 'aggrwrite (s)', 'make_resident (s)', 'h2d (s)', 'd2h (s)', 'num_gpu_pagefaults_read', 'num_sys_pages_read', 'num_sys_pages_flush', 'num_sys_pages_cpu_evict', 'num_sys_pages_gpu_evict',
    ])

    for filename in os.listdir(args.rfolder):
        if not filename.endswith('.out'):
            continue
        filepath = os.path.join(args.rfolder, filename)

        opts = filename[len('result-'):][:-len('.out')].split('-')

        opts_data = [
            opt in opts
            for opt in ['readcache', 'aioread', 'lazywrite', 'aiowrite', 'aggr', 'dis',]
        ]

        with open(filepath, 'r') as f:
            print(filepath)
            data = None
            nvmgpu_data = None
            script_footer_data = None
            for line in f:
                if line.startswith('==> data:'):
                    data = line[len('==> data: '):].strip().split(',')
                elif line.startswith('==> nvmgpu-data:'):
                    nvmgpu_data = line[len('==> nvmgpu-data: '):].strip().split(',')
                elif line.startswith('===> script_footer_data:'):
                    script_footer_data = line[len('===> script_footer_data: '):].strip().split(',')
                    dfolder = script_footer_data[2].strip().split('/')
                    for s in dfolder:
                        if s.endswith('G') and s[:-1].isdigit():
                            script_footer_data[2] = s
                            break
                    
                    writer.writerow(script_footer_data + opts_data + data + nvmgpu_data)

if __name__ == '__main__':
    main(parseargs())

