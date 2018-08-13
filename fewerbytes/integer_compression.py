import numpy as np
import logging
from typing import Tuple, Union
from fewerbytes.types import NumpyType, NumpySizes, NumpyKinds
from fewerbytes.compression_details import (
    IntegerMinimizeTransformation,
    IntegerElementWiseTransformation,
    IntegerHashTransformation
)
from fewerbytes.exceptions import NumpyDtypeKindInvalidException


def downcast_integers(arr: np.array) -> Tuple[np.array, NumpyType]:
    """
    Simple downcasting technique, sees if the numpy array can be downcast
    :param arr: numpy array of type uint or int of any size
    :return: numpy array, simply and safely truncated
    """
    arr_type = NumpyType.from_dtype(arr.dtype)
    logging.debug('downcasting array of shape {}, itemsize {}, and kind {}'.format(
        arr.shape,
        arr_type.size,
        arr_type.kind
    ))
    if arr_type.kind is not NumpyKinds.INTEGER and arr_type.kind is not NumpyKinds.UNSIGNED:
        raise NumpyDtypeKindInvalidException('invalid dtype kind. expecting '
                                             'INTEGER or UNSIGNED, got: {}'.format(arr.dtype.kind))
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    logging.debug('array min: {}, max: {}'.format(arr_min, arr_max))

    downcast_kind = NumpyType.from_integer(maximum=arr_max, minimum=arr_min)
    logging.debug('downcast smallest storage: {}'.format(downcast_kind))

    existing_kind = NumpyType.from_dtype(arr.dtype)
    if downcast_kind.size == existing_kind.size:  # no simple downcast
        logging.debug('existing smallest storage: {}'.format(existing_kind))
        return arr, downcast_kind

    logging.debug('truncating down to {}'.format(downcast_kind))
    return arr.astype(dtype=downcast_kind.to_dtype()), downcast_kind


def integer_minimize_compression(arr: np.array) -> Tuple[np.array, NumpyType, IntegerMinimizeTransformation]:
    """
    Performs a minimization of the integer array
    :param arr: numpy integer array
    :return: tuple of the shifted numpy array, the returned type, and the IntegerMinimizeTransformation info
    """
    logging.debug('performing a minimization compression on array')
    arr_type = NumpyType.from_dtype(arr.dtype)
    logging.debug('array type: {}'.format(arr_type))
    min_value = np.amin(arr)
    ret_array, ret_array_type = downcast_integers(arr - min_value)
    return ret_array, ret_array_type, IntegerMinimizeTransformation(min_value)


def integer_derivative_compression(arr: np.array) -> Tuple[np.array, NumpyType, IntegerElementWiseTransformation]:
    """
    Technique whereby the array is shrunk by taking an element-wise difference
    :param arr: numpy array of integers
    :return: tuple of the new numpy array, the NumpyType, and a list of IntegerTransformations
    """
    logging.debug('single derivative integer compression beginning')
    first_value = arr[0]
    arr_type = NumpyType.from_dtype(arr.dtype)
    logging.debug('first value: {}, array type: {}'.format(first_value, arr_type))
    elem_array, elem_array_type = downcast_integers(np.ediff1d(arr))
    logging.debug('element wise array NumpyType: {}'.format(elem_array_type))
    return elem_array, elem_array_type, IntegerElementWiseTransformation(first_value)


def integer_derivative_then_minimize_compression(arr: np.array) -> \
        Tuple[np.array, NumpyType, IntegerElementWiseTransformation, Union[IntegerMinimizeTransformation, None]]:
    """
    Tries to perform a derivative compression, then optionally tries to minimize. It will return the compressed array
    as well as the element-wise and minimum transformation or None
    :param arr: integer numpy array
    :return: compressed array, its NumpyType, and the element-transform and minimize-transform, or None if not done
    """
    logging.debug('integer derivative and minimization compression function')
    arr_type = NumpyType.from_dtype(arr.dtype)
    logging.debug('array NumpyType {}'.format(arr_type))
    elem_array, elem_array_type, elem_transform = integer_derivative_compression(arr)
    logging.debug('element-wise NumpyType {}'.format(elem_array_type))
    em_array, em_array_type, min_transform = integer_minimize_compression(elem_array)
    logging.debug('minimize array NumpyType {}'.format(em_array_type))
    if em_array_type.is_smaller_than(elem_array_type):
        logging.debug('minimized element-wise array is better, returning that')
        return em_array, em_array_type, elem_transform, min_transform
    else:
        logging.debug('minimized element-wise array does not improve compression')
        return elem_array, elem_array_type, elem_transform, None


def combined_integer_compression(arr: np.array) -> Tuple[np.array, NumpyType, list]:
    """
    Executes 3 single_derivative_integer_compressions as well as hash attempts and returns the compressed array,
    its type, and a list of transforms
    :param arr: numpy array of integers
    :return: tuple of the compressed array, its NumpyType, and a list of transformations
    """
    logging.debug('attempting to compress the integer array')
    max_loops = 3
    which_loop = 0
    working_array = arr
    working_type = NumpyType.from_dtype(working_array.dtype)
    working_transforms = []
    # save the best result
    best_type = working_type
    best_transforms = []
    best_array = arr
    # best hash
    best_hash_type = None
    best_hash_transforms = None
    best_hash_array = None
    while which_loop < max_loops and working_type.size > NumpySizes.BYTE:
        which_loop += 1
        logging.debug('starting {} of {} loops'.format(which_loop, max_loops))
        working_array, working_type, elem_t, min_t = integer_derivative_then_minimize_compression(working_array)
        working_transforms.extend([elem_t, min_t])

        hashed_array, hash_keys_type, hash_transform = integer_hash_compression(working_array)
        if hash_transform is not None:  # this requires 20% better improvement than working_array
            if best_hash_array is None:
                logging.debug('hash was successful, best hash saved')
                best_hash_type = hash_keys_type
                best_hash_array = hashed_array
                best_hash_transforms = working_transforms
                best_hash_transforms.append(hash_transform)
            else:  # need to see if it is better
                logging.debug('hash was successful and better than the working array, see if '
                              'it is better than previously best hash')
                prev_best_transform = best_hash_transforms[-1]
                best_hash_bytes = best_hash_type.size * len(best_hash_array) + \
                    prev_best_transform.key_values_type.size * len(prev_best_transform.key_values)
                new_hash_bytes = hash_keys_type.size * len(hashed_array) + \
                    hash_transform.key_values_type.size * len(hash_transform.key_values)
                logging.debug('previous best hash requires {} bytes, new hash '
                              'requires {}'.format(best_hash_bytes, new_hash_bytes))
                # require a 10% improvement in order to make the extra transform worth it
                if new_hash_bytes < 0.9 * best_hash_bytes:
                    logging.debug('new hash is at least 10% better, saving new hash')
                    best_hash_array = hashed_array
                    best_hash_type = hash_keys_type
                    best_hash_transforms = working_transforms
                    best_hash_transforms.append(hash_transform)
        elif working_type.is_smaller_than(best_type):  # else, we are at least byte-wise smaller, even if no hash
            logging.debug('element-wise differential and minimized array type is smaller than previous best')
            best_transforms = working_transforms
            best_array = working_array
            best_type = working_type
    if best_hash_array is not None:
        logging.debug('deciding whether best hash array is better than best non-hash array')
        hash_transform = best_hash_transforms[-1]
        hash_bytes = best_hash_type.size * len(best_hash_array) + \
            hash_transform.key_values_type.size * len(hash_transform.key_values)
        unhashed_bytes = best_type.size * len(best_array)
        if hash_bytes < 0.8 * unhashed_bytes:
            logging.debug('hashed array is sufficiently better, returning it')
            return best_hash_array, best_hash_type, best_hash_transforms
    return best_array, best_type, best_transforms


def integer_hash_compression(arr: np.array) -> Tuple[np.array, NumpyType, Union[IntegerHashTransformation, None]]:
    """
    Gets the unique values in an array, and produces a hash set
    :param arr: numpy integer array
    :return: array of keys, NumpyType of array of keys, IntegerHashTransformation info or None
    """
    array_type = NumpyType.from_dtype(arr.dtype)
    logging.debug('starting hash integer compression on array with type {}'.format(array_type))
    if array_type.size == NumpySizes.BYTE:  # no improvement possible
        logging.debug('array elements are already 1 byte, cannot compress')
        return arr, array_type, None
    array_length = len(arr)
    array_bytes = array_length * array_type.size.value
    logging.debug('array currently is {} bytes'.format(array_bytes))

    unique_values, unique_values_type = downcast_integers(np.unique(arr))
    unique_values_len = len(unique_values)
    unique_values_bytes = unique_values_len * unique_values_type.size.value
    logging.debug('{} unique values of type {}'.format(unique_values_len, unique_values_type))

    key_type = NumpyType.from_integer(unique_values_len - 1)
    logging.debug('key type: {}'.format(key_type))
    if not key_type.is_smaller_than(array_type):  # keys type isn't less than current size, no improvement
        logging.debug('key type is not smaller than original array type. hash does not make sense')
        return arr, array_type, None
    keys_bytes = array_length * key_type.size.value
    logging.debug('hash keys require {} bytes'.format(keys_bytes))
    if (keys_bytes + unique_values_bytes) < 0.8 * array_bytes:  # if a 20% byte-wise improvement, proceed
        logging.debug('at least 20% byte improvement gained, using hash table')
        hash_dict = {}
        for i in range(len(unique_values)):
            hash_dict[unique_values[i]] = i
        key_array = np.zeros(arr.shape, dtype=key_type.to_dtype())
        for i in range(len(key_array)):
            key_array[i] = hash_dict[arr[i]]
        return key_array, key_type, IntegerHashTransformation(unique_values, unique_values_type)
    else:
        logging.debug('hash does not give enough byte improvement, abandoning hash')
        return arr, array_type, None  # else, no improvement
