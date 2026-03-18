# python
import os, numpy as np
import numpy.lib.format as nf

p='../../mrifeat/subj01/subj01_early_betas_tr.npy'
print("path:", p)
print("filesize (bytes):", os.path.getsize(p))

with open(p,'rb') as f:
    version = nf.read_magic(f)
    if version == (1,0):
        header = nf.read_array_header_1_0(f)
    elif version == (2,0):
        header = nf.read_array_header_2_0(f)
    else:
        header = nf.read_array_header_1_0(f)
    shape, fortran, dtype = header
    print("header shape:", shape)
    print("dtype:", dtype)
    expected_elements = int(np.prod(shape))
    actual_elements = os.path.getsize(p) // np.dtype(dtype).itemsize
    print("expected elements (shape product):", expected_elements)
    print("actual elements (filesize / dtype.itemsize):", actual_elements)
    print("mismatch:", actual_elements != expected_elements)