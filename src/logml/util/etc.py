
import re


def array_to_str(x):
    return '[' + ' '.join([str(xi) for xi in x.ravel()]) + ']'


def camel_to_snake(name):
    ''' Convert CamelCase names to snake_case '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
