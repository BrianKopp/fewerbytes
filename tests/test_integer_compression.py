import unittest
import numpy as np
import fewerbytes.exceptions as x
import fewerbytes.integer_compression as ic
import fewerbytes.types as t


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


def integer_sequential_array():
    arr = np.zeros(1000, np.uint16)
    for i in range(1000):
        arr[i] = i
    return arr


def integer_hash_array_not_worth_it():
    return np.array([
        1000, 1000, 1000
    ], dtype=np.int16)


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

    def test_minimize_works(self):
        arr, nt, transform = ic.integer_minimize_compression(integer_descending_array())
        self.assertEqual(10, len(arr))
        self.assertEqual(nt.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(nt.size, t.NumpySizes.SHORT)
        self.assertEqual(91000, transform.reference_value)
        return

    def test_derivative_works(self):
        arr, nt, transform = ic.integer_derivative_compression(unsigned_byte_arr())
        self.assertEqual(9, len(arr))
        self.assertEqual(nt.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(nt.size, t.NumpySizes.BYTE)
        self.assertEqual(1, transform.reference_value)
        self.assertEqual(1, arr[2])
        return

    def test_derivative_then_minimize(self):
        arr, nt, elem_transform, min_transform = ic.integer_derivative_then_minimize_compression(
            integer_descending_array()
        )
        self.assertEqual(9, len(arr))
        self.assertEqual(nt.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(nt.size, t.NumpySizes.BYTE)
        self.assertEqual(100000, elem_transform.reference_value)
        self.assertEqual(-1000, min_transform.reference_value)
        return

    def test_integer_hash_works(self):
        arr, nt, transform = ic.hash_integer_compression(integer_hashable_array())
        self.assertEqual(15, len(arr))
        self.assertEqual(0, sum(arr[0:5]))
        self.assertEqual(5, sum(arr[5:10]))
        self.assertEqual(10, sum(arr[10:15]))
        self.assertEqual(t.NumpySizes.BYTE, nt.size)
        self.assertEqual(t.NumpyKinds.UNSIGNED, nt.kind)
        self.assertEqual(t.NumpySizes.SINGLE, transform.key_values_type.size)
        self.assertEqual(t.NumpyKinds.UNSIGNED, transform.key_values_type.kind)
        self.assertEqual(3, len(transform.key_values))
        self.assertEqual(1000000, transform.key_values[0])
        self.assertEqual(1111111, transform.key_values[1])
        self.assertEqual(2222222, transform.key_values[2])
        return

    def test_integer_hash_not_try_byte(self):
        arr, nt, transform = ic.hash_integer_compression(unsigned_byte_arr())
        self.assertEqual(None, transform)
        return

    def test_integer_hash_keys_not_smaller(self):
        arr, nt, transform = ic.hash_integer_compression(integer_sequential_array())
        self.assertEqual(None, transform)
        return

    def test_integer_hash_not_worth_it(self):
        # beforehand, array has 3 2-Byte elements = 6B.
        # after, array has 3 1-Byte elements = 3B, plus 1 2-Byte key = 2B. 5/6 = 0.83, not better than 80%
        arr, nt, transform = ic.hash_integer_compression(integer_hash_array_not_worth_it())
        self.assertEqual(None, transform)
        return

if __name__ == '__main__':
    unittest.main()
