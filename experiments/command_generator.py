#!/usr/bin/env python
from pprint import pprint
import random


def main():
    with open('experiments/configs') as f:
        data = f.readlines()

    data = [line.strip() for line in data if line.strip() != '']
    splits = list()
    for line in data:
        if line.startswith('#'):  # comment
            continue

        if line.endswith(':'):  # key
            splits.append(list())
        else:
            splits[-1].append(line)

    print('Parsed {} splits with total {} languages'.format(len(splits), sum([len(l) for l in splits])))

    held_out = int(input('Which subset to hold out (1/2/3/4)? '))
    held_out = held_out - 1
    assert 0 <= held_out < 4

    train_langs = [lang for subset in splits for lang in subset if subset != splits[held_out]]
    print(f'Train lang size: {len(train_langs)}')

    dev_langs_idx = random.sample(range(len(train_langs)), 5)

    output = "python main.py --train --refine --laplace --dropoute 0 --dropouti 0.1 --dropouth 0.1 --dropouto 0.4 --wdrop 0.2"

    output += ' \\\n\t --dev-langs'
    for i, idx in enumerate(dev_langs_idx):
        lang = train_langs[idx]
        if i % 20 == 0 and i != 0:
            output += ' \\\n\t\t'
        output += f' {lang}'
    output += ' \\\n\t --target-langs'
    for j, lang in enumerate(splits[held_out]):
        if j % 20 == 0 and j != 0:
            output += ' \\\n\t\t'
        output += f' {lang}'

    with open('experiments/experiment.sh', 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(output)

    print(f'Written command \n{output}\n to experiments/experiment.sh')


if __name__ == '__main__':
    main()
