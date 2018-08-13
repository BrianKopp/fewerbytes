import fewerbytes.compression_details as transforms
from fewerbytes.types import NumpyType, NumpySizes, NumpyKinds
from fewerbytes.integer_compression import (
    integer_minimize_compression,
    integer_derivative_compression,
    integer_hash_compression,
    downcast_integers
)
from fewerbytes.integer_decompression import (
    integer_minimize_decompression,
    integer_derivative_decompression,
    integer_hash_decompression,
    integer_decompression_from_transform,
    integer_decompression_from_transforms
)
