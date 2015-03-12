__author__ = 'liorsabag'

import numpy as np
import hdf5storage as h5


class MatObj(object):
    def __init__(self, mapping, get_keys=lambda arg: []):
        for key in get_keys(mapping):
            if get_keys(mapping[key]):
                self.__setattr__(key, MatObj(mapping[key], get_keys))
            else:
                self.__setattr__(key, mapping[key])




def mat2obj(folder=".", filename):
    f = h5.read(folder, filename, options=h5.Options.matlab_compatible)
    return MatObj(f, get_dtype_name)


def get_dtype_name(npa):
    if npa.dtype.names:
        return npa.dtype.names
    else:
        return []
