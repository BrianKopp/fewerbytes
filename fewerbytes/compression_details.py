from enum import Enum


class IntegerTransformTypes(Enum):
    MINIMIZE = 'm'
    DERIVATIVE = 'e'
    HASH = 'h'


class IntegerMinimizeTransformation:
    def __init__(self, minimum_value: int):
        """
        Minimum value, baseline for calculation
        :param minimum_value:
        """
        self.transform_type = IntegerTransformTypes.MINIMIZE
        self.reference_value = minimum_value
        return


class IntegerElementWiseTransformation:
    def __init__(self, first_value: int):
        """
        first_value, baseline for calculation
        :param first_value:
        """
        self.transform_type = IntegerTransformTypes.MINIMIZE
        self.reference_value = first_value
        return


class IntegerHashTransformation:
    def __init__(self, key_values, key_value_type):
        """
        :param key_values: value of keys
        :param key_value_type: type of keys
        """
        self.transform_type = IntegerTransformTypes.HASH
        self.key_values = key_values
        self.key_values_type = key_value_type
        return


class CompressionDetails:
    """
    Class which stores all the options and information required
    to decompress a data-set.
    """
    def __init__(self):
        """
        Initializes compression details. Set types through
        static pseudo-constructor methods
        """
        # uncompressed data shape
        self._bytes_per_value_uncompressed = 0
        self._type_char_uncompressed = None  # either 'i', 'u', or 'f', or the corresponding int

        # compressed data shape
        self._bytes_per_value_compressed = 0
        self._type_char_compressed = None  # either 'i', 'u', or 'f'

        # floating point compression data
        self._floating_point_rounded = False
        self._floating_point_rounded_num_decimals = 0

        # compression data
        self._transformations = []  # elements must be TransformTypes

        # hash
        self._use_hash_map = False
        self._num_hashed_values = 0
        self._bytes_per_value_hash = 0
        self._type_char_hash = None  # either 'i', u', or 'f'
        return