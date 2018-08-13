import unittest
import numpy as np
import fewerbytes.types as t
import fewerbytes.exceptions as x


class TypesTest(unittest.TestCase):
    def test_numpy_kinds_from_dtype(self):
        self.assertEqual(t.NumpyKinds.FLOAT, t.NumpyKinds.from_dtype(np.float16))
        self.assertEqual(t.NumpyKinds.FLOAT, t.NumpyKinds.from_dtype(np.float32))
        self.assertEqual(t.NumpyKinds.FLOAT, t.NumpyKinds.from_dtype(np.float64))
        self.assertEqual(t.NumpyKinds.INTEGER, t.NumpyKinds.from_dtype(np.int8))
        self.assertEqual(t.NumpyKinds.INTEGER, t.NumpyKinds.from_dtype(np.int16))
        self.assertEqual(t.NumpyKinds.INTEGER, t.NumpyKinds.from_dtype(np.int32))
        self.assertEqual(t.NumpyKinds.INTEGER, t.NumpyKinds.from_dtype(np.int64))
        self.assertEqual(t.NumpyKinds.UNSIGNED, t.NumpyKinds.from_dtype(np.uint8))
        self.assertEqual(t.NumpyKinds.UNSIGNED, t.NumpyKinds.from_dtype(np.uint16))
        self.assertEqual(t.NumpyKinds.UNSIGNED, t.NumpyKinds.from_dtype(np.uint32))
        self.assertEqual(t.NumpyKinds.UNSIGNED, t.NumpyKinds.from_dtype(np.uint64))
        with self.assertRaises(x.NumpyDtypeKindInvalidException):
            t.NumpyKinds.from_dtype(np.complex64)
        return

    def test_numpy_sizes_from_dtype(self):
        self.assertEqual(t.NumpySizes.BYTE, t.NumpySizes.from_dtype(np.int8))
        self.assertEqual(t.NumpySizes.SHORT, t.NumpySizes.from_dtype(np.int16))
        self.assertEqual(t.NumpySizes.SINGLE, t.NumpySizes.from_dtype(np.int32))
        self.assertEqual(t.NumpySizes.DOUBLE, t.NumpySizes.from_dtype(np.int64))

        self.assertEqual(t.NumpySizes.BYTE, t.NumpySizes.from_dtype(np.uint8))
        self.assertEqual(t.NumpySizes.SHORT, t.NumpySizes.from_dtype(np.uint16))
        self.assertEqual(t.NumpySizes.SINGLE, t.NumpySizes.from_dtype(np.uint32))
        self.assertEqual(t.NumpySizes.DOUBLE, t.NumpySizes.from_dtype(np.uint64))

        self.assertEqual(t.NumpySizes.SHORT, t.NumpySizes.from_dtype(np.float16))
        self.assertEqual(t.NumpySizes.SINGLE, t.NumpySizes.from_dtype(np.float32))
        self.assertEqual(t.NumpySizes.DOUBLE, t.NumpySizes.from_dtype(np.float64))

        with self.assertRaises(x.NumpyDtypeSizeInvalidException):
            t.NumpySizes.from_dtype(np.dtype('S16'))
        return

    def test_numpy_sizes_from_signed_integer(self):
        self.assertEqual(t.NumpySizes.BYTE, t.NumpySizes.from_signed(int_min=-128, int_max=127))
        self.assertEqual(t.NumpySizes.SHORT, t.NumpySizes.from_signed(int_min=-128, int_max=128))
        self.assertEqual(t.NumpySizes.SHORT, t.NumpySizes.from_signed(int_min=-129, int_max=127))

        self.assertEqual(t.NumpySizes.SHORT, t.NumpySizes.from_signed(int_min=0, int_max=32767))
        self.assertEqual(t.NumpySizes.SINGLE, t.NumpySizes.from_signed(int_min=0, int_max=32768))
        self.assertEqual(t.NumpySizes.SINGLE, t.NumpySizes.from_signed(int_min=0, int_max=2147483647))
        self.assertEqual(t.NumpySizes.DOUBLE, t.NumpySizes.from_signed(int_min=0, int_max=2147483648))

        self.assertEqual(t.NumpySizes.SHORT, t.NumpySizes.from_signed(int_min=-32768, int_max=0))
        self.assertEqual(t.NumpySizes.SINGLE, t.NumpySizes.from_signed(int_min=-32769, int_max=0))
        self.assertEqual(t.NumpySizes.SINGLE, t.NumpySizes.from_signed(int_min=-2147483648, int_max=0))
        self.assertEqual(t.NumpySizes.DOUBLE, t.NumpySizes.from_signed(int_min=-2147483649, int_max=0))

        with self.assertRaises(ValueError):
            t.NumpySizes.from_signed(int_min=1, int_max=0)
        with self.assertRaises(ValueError):
            t.NumpySizes.from_signed(int_min=0, int_max=9223372036854775808)
        with self.assertRaises(ValueError):
            t.NumpySizes.from_signed(int_min=0, int_max=-9223372036854775809)
        return

    def test_numpy_sizes_from_unsigned(self):
        self.assertEqual(t.NumpySizes.BYTE, t.NumpySizes.from_unsigned(255))
        self.assertEqual(t.NumpySizes.SHORT, t.NumpySizes.from_unsigned(256))
        self.assertEqual(t.NumpySizes.SHORT, t.NumpySizes.from_unsigned(65535))
        self.assertEqual(t.NumpySizes.SINGLE, t.NumpySizes.from_unsigned(65536))
        self.assertEqual(t.NumpySizes.SINGLE, t.NumpySizes.from_unsigned(4294967295))
        self.assertEqual(t.NumpySizes.DOUBLE, t.NumpySizes.from_unsigned(4294967296))
        self.assertEqual(t.NumpySizes.DOUBLE, t.NumpySizes.from_unsigned(18446744073709551615))
        with self.assertRaises(ValueError):
            t.NumpySizes.from_unsigned(18446744073709551616)
        return

    def test_numpy_type_init(self):
        d = t.NumpyType(t.NumpyKinds.FLOAT, t.NumpySizes.DOUBLE)
        self.assertEqual(t.NumpyKinds.FLOAT, d.kind)
        self.assertEqual(t.NumpySizes.DOUBLE, d.size)
        return

    def test_numpy_type_to_type(self):
        self.assertEqual(np.int8, t.NumpyType(t.NumpyKinds.INTEGER, t.NumpySizes.BYTE).to_dtype())
        self.assertEqual(np.int16, t.NumpyType(t.NumpyKinds.INTEGER, t.NumpySizes.SHORT).to_dtype())
        self.assertEqual(np.int32, t.NumpyType(t.NumpyKinds.INTEGER, t.NumpySizes.SINGLE).to_dtype())
        self.assertEqual(np.int64, t.NumpyType(t.NumpyKinds.INTEGER, t.NumpySizes.DOUBLE).to_dtype())

        self.assertEqual(np.uint8, t.NumpyType(t.NumpyKinds.UNSIGNED, t.NumpySizes.BYTE).to_dtype())
        self.assertEqual(np.uint16, t.NumpyType(t.NumpyKinds.UNSIGNED, t.NumpySizes.SHORT).to_dtype())
        self.assertEqual(np.uint32, t.NumpyType(t.NumpyKinds.UNSIGNED, t.NumpySizes.SINGLE).to_dtype())
        self.assertEqual(np.uint64, t.NumpyType(t.NumpyKinds.UNSIGNED, t.NumpySizes.DOUBLE).to_dtype())

        self.assertEqual(np.float16, t.NumpyType(t.NumpyKinds.FLOAT, t.NumpySizes.SHORT).to_dtype())
        self.assertEqual(np.float32, t.NumpyType(t.NumpyKinds.FLOAT, t.NumpySizes.SINGLE).to_dtype())
        self.assertEqual(np.float64, t.NumpyType(t.NumpyKinds.FLOAT, t.NumpySizes.DOUBLE).to_dtype())
        return

    def test_numpy_type_smaller_than(self):
        eight = t.NumpyType(t.NumpyKinds.UNSIGNED, t.NumpySizes.BYTE)
        also_eight = t.NumpyType(t.NumpyKinds.UNSIGNED, t.NumpySizes.BYTE)
        sixteen = t.NumpyType(t.NumpyKinds.UNSIGNED, t.NumpySizes.SHORT)
        self.assertTrue(eight.is_smaller_than(sixteen))
        self.assertFalse(eight.is_smaller_than(also_eight))
        self.assertFalse(sixteen.is_smaller_than(eight))
        return

    def test_numpy_type_from_dtype(self):
        nt = t.NumpyType.from_dtype(np.float32)
        self.assertEqual(nt.kind, t.NumpyKinds.FLOAT)
        self.assertEqual(nt.size, t.NumpySizes.SINGLE)
        return

    def test_numpy_type_from_integer(self):
        nt = t.NumpyType.from_integer(maximum=1000)
        self.assertEqual(nt.kind, t.NumpyKinds.UNSIGNED)
        self.assertEqual(nt.size, t.NumpySizes.SHORT)

        nt = t.NumpyType.from_integer(maximum=1000, minimum=-1000)
        self.assertEqual(nt.kind, t.NumpyKinds.INTEGER)
        self.assertEqual(nt.size, t.NumpySizes.SHORT)
        return

    def test_numpy_type_errors(self):
        nt = t.NumpyType(t.NumpyKinds.INTEGER, None)
        with self.assertRaises(x.NumpyDtypeSizeInvalidException):
            nt.to_dtype()
        nt.kind = t.NumpyKinds.UNSIGNED
        with self.assertRaises(x.NumpyDtypeSizeInvalidException):
            nt.to_dtype()
        nt.kind = t.NumpyKinds.FLOAT
        with self.assertRaises(x.NumpyDtypeSizeInvalidException):
            nt.to_dtype()
        nt.kind = None
        with self.assertRaises(x.NumpyDtypeKindInvalidException):
            nt.to_dtype()
        return

    def test_numpy_type_equality(self):
        t1 = t.NumpyType(t.NumpyKinds.INTEGER, t.NumpySizes.BYTE)
        t2 = t.NumpyType(t.NumpyKinds.INTEGER, t.NumpySizes.BYTE)
        t3 = t.NumpyType(t.NumpyKinds.UNSIGNED, t.NumpySizes.BYTE)
        t4 = t.NumpyType(t.NumpyKinds.INTEGER, t.NumpySizes.DOUBLE)
        self.assertTrue(t1 == t2)
        self.assertFalse(t1 == t3)
        self.assertFalse(t1 == t4)
        self.assertFalse(t1 == 1)
        return

if __name__ == '__main__':
    unittest.main()
