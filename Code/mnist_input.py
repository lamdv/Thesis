import struct

import numpy as np
import matplotlib.pyplot as plot

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

filename = "D:\\Thesis\\Code\\MNIST\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte"
a = read_idx(filename)

plot.imshow(a[10])
plot.show()