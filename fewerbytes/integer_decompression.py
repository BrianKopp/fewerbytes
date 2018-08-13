import logging
import numpy as np
from fewerbytes.compression_details import (
    IntegerMinimizeTransformation,
    IntegerElementWiseTransformation,
    IntegerHashTransformation
)


def integer_minimize_decompression(arr: np.array, transform: IntegerMinimizeTransformation) -> np.array:
    """
    Decompresses a minimize transform
    :param arr: minimized array
    :param transform: transformation info
    :return: un-minimized array
    """
    logging.debug('decompressing minimized array with info: {}'.format(transform))
    return arr + transform.reference_value


def integer_derivative_decompression(arr: np.array, transform: IntegerElementWiseTransformation) -> np.array:
    """
    Decompresses an element-wise derivative transform
    :param arr: derivative array
    :param transform: transformation info
    :return: decompressed array
    """
    logging.debug('decompressing derivative array with info: {}'.format(transform))
    ret_array = np.cumsum(arr) + transform.reference_value
    ret_array = np.insert(ret_array, 0, transform.reference_value)
    return ret_array


def integer_hash_decompression(arr: np.array, transform: IntegerHashTransformation) -> np.array:
    """
    Decompresses a hashed integer array
    :param arr: compressed key array
    :param transform: hash transform info
    :return: decompressed array
    """
    logging.debug('decompressing hash array with info: {}'.format(transform))
    ret_array = np.zeros(arr.shape, dtype=transform.key_values_type.to_dtype())
    for i in range(len(arr)):
        ret_array[i] = transform.key_values[arr[i]]
    return ret_array
