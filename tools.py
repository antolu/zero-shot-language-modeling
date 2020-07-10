#!/usr/bin/env python

from os import system as shell

"""
This script aims to simplify package installation and set up the local environment for easily running the zerolm code.
"""


OPTIONS = [
    {
        'shell': 'source scripts/tools.sh && installCUDA',
        'desc': 'Install CUDA',
        'tag': 'install_cuda',
    },
    {
        'desc': 'Install Conda',
        'shell': 'source scripts/tools.sh && installConda',
        'tag': 'install_conda',
    },
    {
        'shell': 'source scripts/tools.sh && installPythonPackages',
        'desc': 'Create a conda environment with all required packages',
        'tag': 'create_condaenv',
    },
    {
        'desc': 'Make required directories',
        'shell': 'source scripts/tools.sh && makeDirectories',
        'tag': 'mkdir',
    },
    {
        'desc': 'Get bibles dataset',
        'shell': 'source scripts/tools.sh && getDataset',
        'tag': 'get_dataset',
    }
]


def main():
    tag2idx = dict()
    for option in OPTIONS:
        tag2idx[option['tag']] = len(tag2idx)

    print('The following options are available: ')
    for i, option in enumerate(OPTIONS):
        print(f'{i+1}. {option["desc"]}')
    print('Select the options to run by comma separation.')
    options = input('>>> ')

    options = options.split(',')
    options = [int(option) for option in options]
    for option in options:
        if option > len(OPTIONS) or option < 1:
            raise ValueError(f'Option {option} is out of bounds.')

    handled = list()

    def handle_option(option: int):
        if option in handled:
            return
        option = OPTIONS[option - 1]
        if 'depends' in option:
            handle_option(tag2idx[option['tag']]-1)
        if 'shell' in option:
            shell('bash -c "{}"'.format(option['shell']))
        handled.append(option)

    for option in options:
        handle_option(option)

    print('Done')


if __name__ == '__main__':
    main()
