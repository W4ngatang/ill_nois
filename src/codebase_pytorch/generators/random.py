import numpy as np
from src.codebase_pytorch.utils.dataset import Dataset
import numpy as np

from src.codebase_pytorch.utils.dataset import Dataset


class RandomNoiseGenerator:
    '''
    Class for generating noise random noise with inf norm eps
    '''

    def __init__(self, args):
        self.eps = args.eps

    def generate(self, data, model, args, fh=None):
        '''

        inputs:
            - images: n_images x im_size x im_size x n_channels
            - model: a model class
        outputs:
            - adversaries: n_images x im_size x im_size x n_channels
            - noise: n_ims x im_size x im_size x n_channels
        '''

        if isinstance(data, tuple):
            ins = data[0]
            outs = data[1]
        elif isinstance(data, Dataset):
            ins = data.ins
            outs = data.outs
        else:
            raise NotImplementedError("Invalid data format")

        random_noise = np.sign(np.random.normal(0, 1, size=ins.size()))
        return self.eps * random_noise
