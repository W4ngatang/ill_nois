import pdb
import numpy as np
from src.codebase_pytorch.utils.dataset import Dataset

class GradientGenerator(object):
    '''
    Class for generating noise using fast sign gradient method (Goodfellow et al., 2015)
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

    def getGradient(self, data, model):

        if isinstance(data, tuple):
            ins = data[0]
            outs = data[1]
        elif isinstance(data, Dataset):
            ins = data.ins
            outs = data.outs
        else:
            raise NotImplementedError("Invalid data format")

        if self.alpha:
            random_noise = self.alpha * np.sign(np.random.normal(0, 1, size=ins.shape))
            ins = ins + random_noise
            gradients = model.get_gradient(ins, outs)

            if self.targeted:
                gradients *= -1.
            adv_noise = random_noise + (self.eps - self.alpha) * np.sign(gradients)
        else:
            gradients = model.get_gradient(ins, outs)
            if self.targeted:
                gradients *= -1.
            adv_noise = self.eps * np.sign(gradients)
        return ins.numpy() + adv_noise
