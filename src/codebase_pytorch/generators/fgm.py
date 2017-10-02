import numpy as np
from src.codebase_pytorch.utils.dataset import Dataset
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

        inputs:a
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

        num_dims = reduce(lambda x, y: x * y, ins.size()[1:])
        chunk_size = 20

        # TODO iterated version
        if self.alpha:
            if self.alpha > self.eps:
                raise ValueError("Alpha must be less than Epsilon")
            random_noise = self.alpha * np.sign(np.random.normal(0, 1, size=ins.size()))
            ins = ins + random_noise
            self.eps = self.eps - self.alpha

        noise_gradients = np.zeros((ins.size()[0], num_dims))
        
        for pos in xrange(0, len(ins), chunk_size):
            data = ins[pos: pos + chunk_size]
            labels = outs[pos: pos + chunk_size]
            gradients = model.get_gradient(data, labels).reshape(-1, num_dims)
            noise_gradients[pos: pos + chunk_size] = [self.eps * vec / np.linalg.norm(vec) for vec in gradients]

        if self.targeted:
            noise_gradients *= -1.

        noise_gradients = noise_gradients.reshape(ins.size())

        return np.clip(ins.cpu().numpy() + noise_gradients, 0, 255)
