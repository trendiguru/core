__author__ = 'liorsabag'

import numpy as np
import hdf5storage as h5


class MatObj(object):
    def __init__(self, mapping, get_keys=None):
        get_keys = get_keys or get_type_names_or_keys
        for key in get_keys(mapping):
            if get_keys(mapping[key]):
                self.__setattr__(key, MatObj(mapping[key], get_keys))
            else:
                self.__setattr__(key, mapping[key])


def mat2obj(folder=".", filename="model2.mat"):
    f = h5.read(folder, filename, options=h5.Options.matlab_compatible)
    return MatObj(f)


def get_type_names_or_keys(npa):
    try:
        return npa.dtype.names
    except AttributeError:
        try:
            return npa.keys()
        except AttributeError:
            return None


def dtype_dict(npa):
    res_dict = {}
    if type(npa) == np.ndarray and npa.dtype.fields:
        for key, value in npa.dtype.fields.iteritems():
            if value[0] == np.dtype('object') and npa[key].shape == (1, 1):
                res_dict[key] = dtype_dict(npa[key][0, 0])
            else:
                res_dict[key] = value
    return res_dict

