# fewerbytes
A numpy-based compression library to make your data require fewerbytes

## Quick Start
```pip install fewerbytes```

```python
import fewerbytes as fb
import numpy as np
arr = np.array([50000, 55000, 60000, 65000, 70000], dtype=np.uint32)
arr.nbytes  # 20 Bytes
new_arr, new_arr_type, details = fb.integer_minimize_compression(arr)
new_arr  # [0, 5000, 10000, 15000, 20000] dtype=uint16
new_arr_type  # NumpyType with kind UNSIGNED and size SHORT (16bit)
details  # IntegerMinimizeTransformation with reference_value 50000
decomp_arr = fb.integer_decompression_from_transform(new_arr, details)  # decompressed array
decomp_arr  # [50000, 55000, 60000, 65000, 70000]
```

## Integer Compression

fewerbytes offers four types of integer compression

### Simple Downcast

This technique simply calculates the minimum and maximum values of the array,
and then attempts to down-cast the integers to a smaller storage size, e.g. from 
64-bit integers to 16-bit integers

```python
import fewerbytes as fb
import numpy as np
arr = np.full(10, fill_value=1, dtype=np.int64)
new_arr, new_arr_type = fb.downcast_integers(arr)  # values are stored in 8-bit unsigned integers
```

### Minimize

This compression technique calculates the minimum array value and subtracts
the value from each array element. This allows switching from signed integers to
unsigned integers, which opens an extra bit, doubling the range of absolute
values that can be stored in each class of integers (8-bit, 16-bit, etc.).

This technique is effective when values in an array are large, but when the
difference between the min and max are not as large.

```python
import fewerbytes as fb
import numpy as np
arr = np.full(10, fill_value=1000000)
fb.integer_minimize_compression(arr)  # (array, NumpyType) - array is all 0's
```

### Derivative or Element-Wise

This compression technique calculates the element-wise difference of the array.
The returned array has 1 fewer element than the original array.

This technique is effective when differences in values in an array are similar.
For example, unix timestamps from some regularly occurring measurement.

```python
import fewerbytes as fb
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
fb.integer_derivative_compression(arr)  # (array, numpy compressed array is [1, 1, 1, 1]
```

### Hash Set

This compression technique calculates a unique set of values in the array. Then,
for each element in the array, it stores the index of its value in the unique set.
The transformation details contain the unique values.

This technique is effective when the original array has a relatively small set
of unique values. Due to the compression overhead of doing key-lookups, this
technique is abandoned if the byte-storage efficiency is improved by less than 20%.

```python
import fewerbytes as fb
import numpy as np
arr = np.array([1000000, 1000001, 1000000, 1000000, 1000000], dtype=np.uint32)
arr.nbytes  # 20 bytes
keys, keys_type, transform = fb.integer_hash_compression(arr)
keys  # array [0, 1, 0, 0, 0]
keys_type  # UNSIGNED BYTE
transform.key_values  # array [1000000, 1000001]
transform.key_values_type  # UNSIGNED SINGLE (32-bit)
```

## Integer Decompression

Integer decompression can be achieved using any of the following functions?

```python
import fewerbytes as fb
# arr and transform obtained from compression
fb.integer_decompression_from_transform(arr, transform)
fb.integer_minimize_decompression(arr, transform)
fb.integer_derivative_decompression(arr, transform)
fb.integer_hash_decompression(arr, transform)
fb.integer_decompression_from_transforms(arr, list_of_transforms)
```
