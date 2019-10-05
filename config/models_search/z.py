#!/usr/bin/env python

import re
import yaml


def camel_to_snake(name):
    ''' Convert CamelCase names to snake_case '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


with open('z.yaml') as yaml_file:
    y = yaml.load(yaml_file, Loader=yaml.FullLoader)
    for n in y:
        s = camel_to_snake(n)
        print(f"{n}:{s}\n{y[n]}")
        file_name = f"{s}.yaml"
        d = dict()
        d[n] = y[n]
        with open(file_name, 'w') as out:
            yaml.dump(d, out, default_flow_style=False)
