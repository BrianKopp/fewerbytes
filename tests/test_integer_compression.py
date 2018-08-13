import unittest
import numpy as np
import fewerbytes.exceptions as x
import fewerbytes.integer_compression as ic
import fewerbytes.types as t
import fewerbytes.compression_details as d


def unsigned_byte_arr():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint8)


def integer_byte_arr():
    return np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int8)


def integer_descending_array():
    return np.array([
        100000,
        99000,
        98000,
        97000,
        96000,
        95000,
        94000,
        93000,
        92000,
        91000
    ])


def integer_hashable_array():
    return np.array([
        1000000, 1000000, 1000000, 1000000, 1000000,
        1111111, 1111111, 1111111, 1111111, 1111111,
        2222222, 2222222, 2222222, 2222222, 2222222
    ], dtype=np.int64)


class TestIntegerCompression(unittest.TestCase):
    def test_integer_downcast_cannot_downcast_byte_unsigned(self):
        arr, nt = ic.downcast_integers(unsigned_byte_arr())
        self.assertEqual(10, len(arr))
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[9], 10)
        self.assertEqual(nt.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(nt.size, t.NumpySizes.BYTE)
        return

    def test_integer_downcast_cannot_downcast_byte_signed(self):
        arr, nt = ic.downcast_integers(integer_byte_arr())
        self.assertEqual(11, len(arr))
        self.assertEqual(arr[0], -5)
        self.assertEqual(arr[10], 5)
        self.assertEqual(nt.kind, t.NumpyKinds.INTEGER)
        self.assertEqual(nt.size, t.NumpySizes.BYTE)
        return

    def test_integer_downcast_downcast_to_byte_unsigned(self):
        arr, nt = ic.downcast_integers(unsigned_byte_arr().astype(np.uint16))
        self.assertEqual(10, len(arr))
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[9], 10)
        self.assertEqual(nt.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(nt.size, t.NumpySizes.BYTE)
        return

    def test_integer_downcast_downcast_to_byte_signed(self):
        arr, nt = ic.downcast_integers(integer_byte_arr().astype(np.int16))
        self.assertEqual(11, len(arr))
        self.assertEqual(arr[0], -5)
        self.assertEqual(arr[10], 5)
        self.assertEqual(nt.kind, t.NumpyKinds.INTEGER)
        self.assertEqual(nt.size, t.NumpySizes.BYTE)
        return

    def test_integer_downcast_signed_to_unsigned(self):
        arr, nt = ic.downcast_integers(unsigned_byte_arr().astype(np.int64))
        self.assertEqual(10, len(arr))
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[9], 10)
        self.assertEqual(nt.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(nt.size, t.NumpySizes.BYTE)
        return

    def test_integer_downcast_kind_error(self):
        with self.assertRaises(x.NumpyDtypeKindInvalidException):
            ic.downcast_integers(np.array([], dtype=np.float32))
        return

    def test_single_derivative_works(self):
        arr, nt, transforms = ic.single_derivative_integer_compression(unsigned_byte_arr())
        self.assertEqual(9, len(arr))
        self.assertEqual(nt.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(nt.size, t.NumpySizes.BYTE)
        self.assertEqual(1, len(transforms))
        self.assertTrue(isinstance(transforms[0], d.IntegerElementWiseTransformation))
        self.assertEqual(1, transforms[0].reference_value)
        self.assertEqual(1, arr[2])
        return

    def test_single_derivative_also_minifies(self):
        arr, nt, transforms = ic.single_derivative_integer_compression(integer_descending_array())
        self.assertEqual(9, len(arr))
        self.assertEqual(nt.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(nt.size, t.NumpySizes.BYTE)
        self.assertEqual(2, len(transforms))
        self.assertTrue(isinstance(transforms[0], d.IntegerElementWiseTransformation))
        self.assertEqual(100000, transforms[0].reference_value)
        self.assertTrue(isinstance(transforms[1], d.IntegerMinimizeTransformation))
        self.assertEqual(-1000, transforms[1].reference_value)
        return

    def test_integer_hash_works(self):

        return

if __name__ == '__main__':
    unittest.main()
