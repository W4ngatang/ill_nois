import pdb
import numpy as np
from src.codebase_pytorch.utils.dataset import Dataset

class FGMGenerator(object):
    '''
    Class for generating noise using fast gradient method (Goodfellow et al., 2015)
    Also allows for random initialization of the noise, which was shown to
        improve performance (Tramer et al., 2017)

    TODO
        - super generator class
        - iterated FGM
        - random FGM

    '''

    def __init__(self, args):
        self.eps = args.eps
        self.alpha = args.alpha
        self.targeted = (args.target != 'none')

    def generate(self, data, model):
        '''
        Generate adversarial noise using fast gradient method.

        TODO: generators should return the adversarial images now

        inputs:
            - images: n_images x im_size x im_size x n_channels
                Images should be in [0,1]^d (unnormalized)
            - model: a model class
        outputs:
            - adversaries: n_images x im_size x im_size x n_channels
        '''

        if isinstance(data, tuple):
            ins = data[0]
            outs = data[1]
        elif isinstance(data, Dataset):
            ins = data.ins
            outs = data.outs
        else:
            raise NotImplementedError("Invalid data format")

        num_dims = reduce(lambda x, y: x * y, ins.shape[1:])

        # TODO iterated version
        if self.alpha:
            random_noise = self.alpha * np.sign(np.random.normal(0, 1, size=ins.shape))
            ins = ins + random_noise
            self.eps = self.eps - self.alpha
            gradients = model.get_gradient(ins, outs)
            if self.targeted:
                gradients *= -1.
            adv_noise = random_noise + (self.eps - self.alpha) * np.sign(gradients)
        else:
            gradients = model.get_gradient(ins, outs).reshape(-1, num_dims)
            for i in xrange(len(gradients)):
                gradients[i] = self.eps * gradients[i] / np.linalg.norm(gradients[i])
            if self.targeted:
                gradients *= -1.
            adv_noise = gradients
        return ins.numpy() + adv_noise
