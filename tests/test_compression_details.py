import unittest
import numpy as np
import fewerbytes.compression_details as c
import fewerbytes.types as fbt


class TestCompressionDetails(unittest.TestCase):
    def test_integer_minimize_transform(self):
        t = c.IntegerMinimizeTransformation(1)
        self.assertEqual(c.IntegerTransformTypes.MINIMIZE, t.transform_type)
        self.assertEqual(1, t.reference_value)
        ts = '{}'.format(t)
        self.assertTrue('reference_value=' in ts)
        return

    def test_integer_element_wise_transform(self):
        t = c.IntegerElementWiseTransformation(1)
        self.assertEqual(c.IntegerTransformTypes.DERIVATIVE, t.transform_type)
        self.assertEqual(1, t.reference_value)
        ts = '{}'.format(t)
        self.assertTrue('reference_value=' in ts)
        return

    def test_integer_hash_transform(self):
        t = c.IntegerHashTransformation(
            np.array([1, 2, 3], dtype=np.uint8),
            fbt.NumpyType(fbt.NumpyKinds.UNSIGNED, fbt.NumpySizes.BYTE)
        )
        self.assertEqual(c.IntegerTransformTypes.HASH, t.transform_type)
        self.assertEqual(3, len(t.key_values))
        self.assertEqual(1, t.key_values[0])
        self.assertEqual(t.key_values_type, fbt.NumpyType(fbt.NumpyKinds.UNSIGNED, fbt.NumpySizes.BYTE))
        ts = '{}'.format(t)
        self.assertTrue('key_values_type=' in ts and 'key_values=' in ts)
        return

if __name__ == '__main__':
    unittest.main()
