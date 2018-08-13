import unittest
import numpy as np
import fewerbytes.integer_decompression as id
import fewerbytes.compression_details as cd
import fewerbytes.types as t


class TestIntegerDecompression(unittest.TestCase):
    def validate_minimize_decompression(self, arr: np.array):
        self.assertEqual(5, len(arr))
        self.assertEqual(1001, arr[0])
        self.assertEqual(1005, arr[4])
        arr_type = t.NumpyType.from_dtype(arr.dtype)
        self.assertEqual(arr_type.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(arr_type.size, t.NumpySizes.SHORT)
        return

    def test_minimize_decompression(self):
        arr = id.integer_minimize_decompression(
            np.array([1, 2, 3, 4, 5], dtype=np.uint8),
            cd.IntegerMinimizeTransformation(1000)
        )
        self.validate_minimize_decompression(arr)
        return

    def validate_elem_decompression(self, arr: np.array):
        self.assertEqual(5, len(arr))
        self.assertEqual(1000, arr[0])
        self.assertEqual(1001, arr[1])
        self.assertEqual(1001, arr[2])
        self.assertEqual(1002, arr[3])
        self.assertEqual(1002, arr[4])
        arr_type = t.NumpyType.from_dtype(arr.dtype)
        self.assertEqual(arr_type.kind, t.NumpyKinds.UNSIGNED)
        return

    def test_elem_decompression(self):
        arr = id.integer_derivative_decompression(
            np.array([1, 0, 1, 0], dtype=np.uint8),
            cd.IntegerElementWiseTransformation(1000)
        )
        self.validate_elem_decompression(arr)
        return

    def validate_hash_decompression(self, arr: np.array):
        self.assertEqual(4, len(arr))
        self.assertEqual(-1000, arr[0])
        self.assertEqual(-1000, arr[1])
        self.assertEqual(0, arr[2])
        self.assertEqual(1000, arr[3])
        arr_type = t.NumpyType.from_dtype(arr.dtype)
        self.assertEqual(arr_type.kind, t.NumpyKinds.INTEGER)
        self.assertEqual(arr_type.size, t.NumpySizes.SHORT)
        return

    def test_hash_decompression(self):
        arr = id.integer_hash_decompression(
            np.array([1, 1, 0, 2], dtype=np.uint8),
            cd.IntegerHashTransformation(
                key_values=np.array([0, -1000, 1000], dtype=np.int16),
                key_value_type=t.NumpyType(t.NumpyKinds.INTEGER, t.NumpySizes.SHORT)
            )
        )
        self.validate_hash_decompression(arr)
        return

    def test_integer_catch_all(self):
        arr = id.integer_decompression_from_transform(
            np.array([1, 2, 3, 4, 5], dtype=np.uint8),
            cd.IntegerMinimizeTransformation(1000)
        )
        self.validate_minimize_decompression(arr)

        arr = id.integer_decompression_from_transform(
            np.array([1, 0, 1, 0], dtype=np.uint8),
            cd.IntegerElementWiseTransformation(1000)
        )
        self.validate_elem_decompression(arr)

        arr = id.integer_decompression_from_transform(
            np.array([1, 1, 0, 2], dtype=np.uint8),
            cd.IntegerHashTransformation(
                key_values=np.array([0, -1000, 1000], dtype=np.int16),
                key_value_type=t.NumpyType(t.NumpyKinds.INTEGER, t.NumpySizes.SHORT)
            )
        )
        self.validate_hash_decompression(arr)
        return

    def test_integer_catch_all_fail(self):
        transform = cd.IntegerMinimizeTransformation(0)
        transform.transform_type = None
        with self.assertRaises(ValueError):
            id.integer_decompression_from_transform(
                np.array([], dtype=np.int8),
                transform
            )
        return

    def test_integer_transform_list(self):
        arr = id.integer_decompression_from_transforms(
            np.array([1, 0, 1, 0], dtype=np.uint8),
            [
                cd.IntegerElementWiseTransformation(1000),
                cd.IntegerMinimizeTransformation(100)
            ]
        )
        self.assertEqual(1100, arr[0])
        self.assertEqual(1101, arr[1])
        self.assertEqual(1101, arr[2])
        self.assertEqual(1102, arr[3])
        self.assertEqual(1102, arr[4])
        return

if __name__ == '__main__':
    unittest.main()
