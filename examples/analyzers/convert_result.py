import argparse
import os.path
import pprint

def type_folder(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError('The specified path "{}" does not exist'.format(path))
    return path

def parseargs():
    parser = argparse.ArgumentParser(
        description = 'Benchmark Result Converter'
    )

    parser.add_argument(
        'ifile',
        type = argparse.FileType('r'),
        help = 'Raw result file'
    )

    parser.add_argument(
        'prog',
        help = 'Output only prog',
        choices = ('cudamemcpy', 'uvm', 'hostreg', 'ap', 'nvmgpu', 'all',)
    )

    return parser.parse_args()

def output_result(result):
    header = ','.join([result['script_footer_header'], result['header'],])
    if result['nvmgpu_header'] is not None:
        header = ','.join([header, result['nvmgpu_header'],])
    print(header)

    i = 0
    while i < len(result['script_footer_data']):
        data = ','.join([result['script_footer_data'][i], result['data'][i],])
        if result['nvmgpu_data'][i] is not None:
            data = ','.join([data, result['nvmgpu_data'][i],])
        print(data)
        i += 1


def main(args):
    results = dict()

    header = None
    nvmgpu_header = None
    script_footer_header = None
    data = None
    nvmgpu_data = None
    script_footer_data = None
    footer_dict = dict()
    for line in args.ifile:
        if line.startswith('==> header:'):
            header = line[len('==> header: '):].strip()
        elif line.startswith('==> data:'):
            data = line[len('==> data: '):].strip()
        elif line.startswith('==> nvmgpu-header:'):
            nvmgpu_header = line[len('==> nvmgpu-header: '):].strip()
        elif line.startswith('==> nvmgpu-data:'):
            nvmgpu_data = line[len('==> nvmgpu-data: '):].strip()
        elif line.startswith('===> script_footer:'):
            script_footer_header = line[len('===> script_footer: '):].strip().split(',')
        elif line.startswith('===> script_footer_data:'):
            script_footer_data = line[len('===> script_footer_data: '):].strip().split(',')
            assert len(script_footer_data) == len(script_footer_header), "The lengths of script_footer_header and script_footer_data should not be different!!"
            i = 0
            while i < len(script_footer_data):
                footer_dict[script_footer_header[i]] = script_footer_data[i]
                i += 1

            dfolder = footer_dict['dfolder'].strip().split('/')
            for s in dfolder:
                if s.endswith('G') and s[:-1].isdigit():
                    footer_dict['dfolder'] = s
                    break
            prog = footer_dict['prog']
            retcode = int(footer_dict['retcode'])

            if retcode != 0:
                continue

            i = 0
            while i < len(script_footer_header):
                script_footer_data[i] = footer_dict[script_footer_header[i]]
                i += 1
            script_footer_data = ','.join(script_footer_data)

            script_footer_header = ','.join(script_footer_header)

            if prog not in results:
                results[prog] = {
                    'header': header,
                    'nvmgpu_header': nvmgpu_header,
                    'script_footer_header': script_footer_header,
                    'data': list(),
                    'nvmgpu_data': list(),
                    'script_footer_data': list(),
                }

            result_prog = results[prog]

            if result_prog['header'] != header or result_prog['nvmgpu_header'] != nvmgpu_header or result_prog['script_footer_header'] != script_footer_header:
                raise Exception('Some headers are different')

            result_prog['data'].append(data)
            result_prog['nvmgpu_data'].append(nvmgpu_data)
            result_prog['script_footer_data'].append(script_footer_data)

            header = None
            nvmgpu_header = None
            script_footer_header = None
            data = None
            nvmgpu_data = None
            script_footer_data = None
            footer_dict = dict()

    if args.prog == 'all':
        for result in results.values():
            output_result(result)
    elif args.prog in results:
        output_result(results[args.prog])
    else:
        raise Exception('{} is not available'.format(args.prog))


if __name__ == '__main__':
    main(parseargs())

