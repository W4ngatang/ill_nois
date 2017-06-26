import pdb
import numpy as np

class FastGradientGenerator:
    '''
    Class for generating noise using fast gradient method (Goodfellow et al., 2015)
    Also allows for random initialization of the noise, which was shown to
        improve performance (Tramer et al., 2017)

    '''

    def __init__(self, args):
        self.eps = args.eps
        self.alpha = args.alpha

    def generate(self, ins, outs, model):
        '''
        Generate adversarial noise using fast gradient method.

        inputs:
            - images: n_images x im_size x im_size x n_channels
            - model: a model class
        outputs:
            - adversaries: n_images x im_size x im_size x n_channels
            - noise: n_ims x im_size x im_size x n_channels
        '''
        if self.alpha:
            random_noise = np.random.normal(0, 1, size=ins.shape)
            ins = ins + self.alpha * random_noise
            gradients = model.get_gradient(ins, outs)
            adv_noise = (self.eps - self.alpha) * np.sign(gradients)
        else:
            gradients = model.get_gradient(ins, outs) # get gradient of model's loss wrt images
            adv_noise = self.eps * np.sign(gradients)
        return ins + adv_noise, adv_noise
