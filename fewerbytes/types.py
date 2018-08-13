from enum import Enum
import numpy as np
import logging
import fewerbytes.exceptions as ex


UNSIGNED_BYTE_MAX = np.iinfo(np.uint8).max
UNSIGNED_SHORT_MAX = np.iinfo(np.uint16).max
UNSIGNED_SINGLE_MAX = np.iinfo(np.uint32).max
UNSIGNED_DOUBLE_MAX = np.iinfo(np.uint64).max

INTEGER_BYTE_MAX = np.iinfo(np.int8).max
INTEGER_SHORT_MAX = np.iinfo(np.int16).max
INTEGER_SINGLE_MAX = np.iinfo(np.int32).max
INTEGER_DOUBLE_MAX = np.iinfo(np.int64).max

INTEGER_BYTE_MIN = np.iinfo(np.int8).min
INTEGER_SHORT_MIN = np.iinfo(np.int16).min
INTEGER_SINGLE_MIN = np.iinfo(np.int32).min
INTEGER_DOUBLE_MIN = np.iinfo(np.int64).min


class NumpyKinds(Enum):
    FLOAT = 'f'
    INTEGER = 'i'
    UNSIGNED = 'u'

    @staticmethod
    def from_dtype(d: np.dtype) -> 'NumpyKinds':
        d = np.dtype(d) if isinstance(d, type) else d
        if d.kind == 'f':
            return NumpyKinds.FLOAT
        if d.kind == 'i':
            return NumpyKinds.INTEGER
        if d.kind == 'u':
            return NumpyKinds.UNSIGNED
        raise ex.NumpyDtypeKindInvalidException('numpy dtype kind not of acceptable type. '
                                                'expected kind in [i, f, u], got {}'.format(d.kind))


class NumpySizes(Enum):
    BYTE = 8
    SHORT = 16
    SINGLE = 32
    DOUBLE = 64

    @staticmethod
    def from_dtype(d: np.dtype) -> 'NumpySizes':
        d = np.dtype(d) if isinstance(d, type) else d
        if d.itemsize == 1:
            return NumpySizes.BYTE
        if d.itemsize == 2:
            return NumpySizes.SHORT
        if d.itemsize == 4:
            return NumpySizes.SINGLE
        if d.itemsize == 8:
            return NumpySizes.DOUBLE
        raise ex.NumpyDtypeSizeInvalidException('numpy dtype not of acceptable type. expected '
                                                'itemsize in [1, 2, 4, 8], got {}'.format(d.itemsize))

    @staticmethod
    def from_signed(int_min: int, int_max: int) -> 'NumpySizes':
        """
        Figures out the minimum integer size from signed integer min/max
        :param int_min: minimum integer size must hold
        :param int_max: maximum integer size must hold
        :return: size of signed-integer required
        """
        logging.debug('Making NumpySizes from integer min {} and max {}'.format(int_min, int_max))
        if int_min > int_max:
            raise ValueError('int_min larger than int_max')
        if int_min >= INTEGER_BYTE_MIN and int_max <= INTEGER_BYTE_MAX:
            return NumpySizes.BYTE
        if int_min >= INTEGER_SHORT_MIN and int_max <= INTEGER_SHORT_MAX:
            return NumpySizes.SHORT
        if int_min >= INTEGER_SINGLE_MIN and int_max <= INTEGER_SINGLE_MAX:
            return NumpySizes.SINGLE
        if int_min >= INTEGER_DOUBLE_MIN and int_max <= INTEGER_DOUBLE_MAX:
            return NumpySizes.DOUBLE
        raise ValueError('integer values min: {} and max: {} could not fit inside '
                         'even a 64-bit integer'.format(int_min, int_max))

    @staticmethod
    def from_unsigned(unsigned_max: int) -> 'NumpySizes':
        """
        Figures out the minimum unsigned integer size
        :param unsigned_max: maximum value unsigned integer size must hold
        :return: size of unsigned-integer required
        """
        logging.debug('Making NumpySizes from unsigned integer max {}'.format(unsigned_max))
        if unsigned_max <= UNSIGNED_BYTE_MAX:
            return NumpySizes.BYTE
        if unsigned_max <= UNSIGNED_SHORT_MAX:
            return NumpySizes.SHORT
        if unsigned_max <= UNSIGNED_SINGLE_MAX:
            return NumpySizes.SINGLE
        if unsigned_max <= UNSIGNED_DOUBLE_MAX:
            return NumpySizes.DOUBLE
        raise ValueError('not even a 64-bit unsigned integer can hold the value: {}'.format(unsigned_max))


class NumpyType:
    def __init__(self, kind: NumpyKinds, size: NumpySizes):
        self.kind = kind
        self.size = size
        return

    def __repr__(self):
        return '<{}, {} kind={}, size={}>'.format(self.__class__.__name__, hex(id(self)), self.kind, self.size)

    def __eq__(self, other):
        if not isinstance(other, NumpyType):
            return False
        return self.kind == other.kind and self.size == other.size

    def to_dtype(self):
        if self.kind == NumpyKinds.INTEGER:
            if self.size == NumpySizes.BYTE:
                return np.int8
            if self.size == NumpySizes.SHORT:
                return np.int16
            if self.size == NumpySizes.SINGLE:
                return np.int32
            if self.size == NumpySizes.DOUBLE:
                return np.int64
            raise ex.NumpyDtypeSizeInvalidException('Could not make numpy type, unexpected size: {}'.format(self.size))
        if self.kind == NumpyKinds.UNSIGNED:
            if self.size == NumpySizes.BYTE:
                return np.uint8
            if self.size == NumpySizes.SHORT:
                return np.uint16
            if self.size == NumpySizes.SINGLE:
                return np.uint32
            if self.size == NumpySizes.DOUBLE:
                return np.uint64
            raise ex.NumpyDtypeSizeInvalidException('Could not make numpy type, unexpected size: {}'.format(self.size))
        if self.kind == NumpyKinds.FLOAT:
            if self.size == NumpySizes.SHORT:
                return np.float16
            if self.size == NumpySizes.SINGLE:
                return np.float32
            if self.size == NumpySizes.DOUBLE:
                return np.float64
            raise ex.NumpyDtypeSizeInvalidException('Could not make numpy type, unexpected size: {}'.format(self.size))
        raise ex.NumpyDtypeKindInvalidException('Could not make numpy type, unexpected kind: {}'.format(self.kind))

    def is_smaller_than(self, other_type: 'NumpyType') -> bool:
        return self.size.value < other_type.size.value

    @staticmethod
    def from_dtype(d: np.dtype) -> 'NumpyType':
        return NumpyType(NumpyKinds.from_dtype(d), NumpySizes.from_dtype(d))

    @staticmethod
    def from_integer(maximum: int, minimum: int = 0) -> 'NumpyType':
        """
        Finds the smallest NumpyType that can store the specified min/max integers.
        Determines whether they should be signed or unsigned
        :param maximum: maximum integer value to store
        :param minimum: minimum integer value to store, default 0
        :return:
        """
        if minimum < 0:
            return NumpyType(
                kind=NumpyKinds.INTEGER,
                size=NumpySizes.from_signed(minimum, maximum)
            )
        else:
            return NumpyType(
                kind=NumpyKinds.UNSIGNED,
                size=NumpySizes.from_unsigned(maximum)
            )
