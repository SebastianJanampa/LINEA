import time
import contextlib
import numpy as np
from PIL import Image
from collections import OrderedDict

import onnx
import torch


def to_binary_data(path, size=(640, 640), output_name='input_tensor.bin'):
    '''--loadInputs='image:input_tensor.bin'
    '''
    im = Image.open(path).resize(size)
    data = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.
    data.tofile(output_name)


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0

    def __enter__(self, ):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self, ):
        self.total = 0

    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()
