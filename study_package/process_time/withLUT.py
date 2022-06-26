import time

import numpy as np

from src.image_processing.process.process import Process

pool_dim = (25, 10)
lut = np.load("lut.npy")

process = Process(lut, pool_dim)

start = time.time()
for _ in range(1000):
    process.test_lut((200, 200))
print(time.time() - start)
