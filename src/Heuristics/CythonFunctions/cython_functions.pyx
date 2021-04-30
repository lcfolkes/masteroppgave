cimport numpy as np
import numpy as np

"""
compile file by adding running this line in the terminal:
python3 setup.py build_ext --inplace
"""

cpdef np.ndarray[np.int64_t, ndim=1] maximum(np.ndarray[np.int64_t, ndim=1] arr, int min_val):
	cdef int i
	cdef int arr_len = arr.shape[0]

	for i in range(arr_len):
		if arr[i] < min_val:
			arr[i] = min_val
	return arr

cpdef np.ndarray[np.int64_t, ndim=1] minimum(np.ndarray[np.int64_t, ndim=1] arr1, np.ndarray[np.int64_t, ndim=1] arr2):
	cdef int i
	cdef int arr_len = arr1.shape[0]

	for i in range(arr_len):
		if arr1[i] > arr2[i]:
			arr1[i] = arr2[i]
	return arr1

cpdef np.ndarray[np.int64_t, ndim=1] minimum_greater_than_0(np.ndarray[np.int64_t, ndim=1] arr1, np.ndarray[np.int64_t, ndim=1] arr2):
	cdef int i
	cdef int arr_len = arr1.shape[0]

	for i in range(arr_len):
		if arr1[i] > arr2[i]:
			if arr2[i] < 0:
				arr1[i] = 0
			else:
				arr1[i] = arr2[i]

	return arr1
