import logging
import numpy as np
from typing import Union
from fewerbytes.compression_details import (
    IntegerMinimizeTransformation,
    IntegerElementWiseTransformation,
    IntegerHashTransformation,
    IntegerTransformTypes
)
from fewerbytes.integer_compression import downcast_integers
import fewerbytes.types as t


def integer_minimize_decompression(arr: np.array, transform: IntegerMinimizeTransformation) -> np.array:
    """
    Decompresses a minimize transform
    :param arr: minimized array
    :param transform: transformation info
    :return: un-minimized array
    """
    logging.debug('decompressing minimized array with info: {}'.format(transform))
    ret_array_type = t.NumpyType(
        kind=t.NumpyKinds.from_dtype(arr.dtype),
        size=t.NumpySizes.DOUBLE  # make it as big as possible then shrink after addition
    )
    return downcast_integers(arr.astype(ret_array_type.to_dtype()) + transform.reference_value)[0]


def integer_derivative_decompression(arr: np.array, transform: IntegerElementWiseTransformation) -> np.array:
    """
    Decompresses an element-wise derivative transform
    :param arr: derivative array
    :param transform: transformation info
    :return: decompressed array
    """
    logging.debug('decompressing derivative array with info: {}'.format(transform))
    ret_array_type = t.NumpyType(
        kind=t.NumpyKinds.from_dtype(arr.dtype),
        size=t.NumpySizes.DOUBLE
    )
    ret_array = np.cumsum(arr.astype(ret_array_type.to_dtype())) + transform.reference_value
    ret_array = np.insert(ret_array, 0, transform.reference_value)
    return downcast_integers(ret_array)[0]


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
    return downcast_integers(ret_array)[0]


def integer_decompression_from_transform(
        arr: np.array, transform: Union[IntegerElementWiseTransformation, IntegerMinimizeTransformation,
                                        IntegerHashTransformation]) -> np.array:
    """
    Decompresses an integer array from a transformation
    :param arr: compressed integer array
    :param transform: transformation information
    :return: decompressed array
    """
    if transform.transform_type == IntegerTransformTypes.MINIMIZE:
        return integer_minimize_decompression(arr, transform)
    elif transform.transform_type == IntegerTransformTypes.DERIVATIVE:
        return integer_derivative_decompression(arr, transform)
    elif transform.transform_type == IntegerTransformTypes.HASH:
        return integer_hash_decompression(arr, transform)
    raise ValueError('Unable to decompress array using transform: {}'.format(transform))


def integer_decompression_from_transforms(arr: np.array, transforms: list) -> np.array:
    """
    Decompresses an array using a series of transforms
    :param arr: compressed array
    :param transforms: list of transforms, IN THE ORDER THEY WERE APPLIED
    :return: decompressed array
    """
    ret_array = arr
    for t in transforms:
        ret_array = integer_decompression_from_transform(ret_array, t)
    return ret_array
