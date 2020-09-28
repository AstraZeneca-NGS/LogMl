
import sys

from collections import deque
from itertools import chain


K = 1024
M = 1024 * K
G = 1024 * M
T = 1024 * G
P = 1024 * T


def bytes2human(num):
    """
    Convert a number of bytes to a 'human readable' format
    """
    if num / 1024 < 1:
        return str(num)
    if num / M < 1:
        return f"{num / K:.1f}K"
    if num / G < 1:
        return f"{num / M:.1f}M"
    if num / T < 1:
        return f"{num / M:.1f}G"
    if num / P < 1:
        return f"{num / G:.1f}T"
    return f"{num / P:.1f}P"


def getsizeof(o, handlers={}, verbose=False):
    """
    Reference: https://code.activestate.com/recipes/577504/

    Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(f"GET SIZE OF: {s}, {type(o)}, {repr(o)}")

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
            if hasattr(o, '__dict__'):
                for k, v in o.__dict__.items():
                    if verbose:
                        print(f"GET SIZE OF OBJECT FIELD: {k}, {type(o)}, {repr(o)}")
                    s += sizeof(v)
                break
        else:
            if verbose:
                print(f"GET SIZE OF: MISSING HANDLER FOR TYPE {type(o)}")
        return s

    return sizeof(o)


def memory(o):
    return bytes2human(getsizeof(o, verbose=False))
