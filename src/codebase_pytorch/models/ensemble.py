import pdb
import time
import numpy as np
import logging as log
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from src.codebase.utils.utils import log as log
from src.codebase_pytorch.utils.scheduler import ReduceLROnPlateau
from src.codebase_pytorch.utils.dataset import Dataset
from src.codebase_pytorch.utils.timer import Timer
from src.codebase_pytorch.models.model import Model

class Ensemble(Model):
    '''
    Optimization-based generator using an ensemble of models
    '''

    def __init__(self, args, models, n_classes, normalize=(None,None)):
        '''

        '''
        super(Ensemble, self).__init__()
        self.use_cuda = args.cuda
        self.batch_size = batch_size = args.batch_size
        self.n_classes = n_classes = args.n_classes
        self.mean = normalize[0]
        self.std = normalize[1]
        for model in models:
            model.eval()
        self.models = models
        self.n_models = n_models = len(models)
        self.w = w = torch.ones(n_models)
        if self.use_cuda:
            w = w.cuda()
        self.weights = Variable(w / w.sum())

    def forward(self, x):
        '''
        NB: Ensemble returns (log of) a valid probability distribution INSTEAD of logits
                in order to return the weighted probability distribution.
            But ensemble ASSUMES that each model returns LOGITS
            This means you should not use the train_model method because it expects logits
        '''
        try:
            pred_dists = Variable(torch.zeros( # probably faster to preallocate
                            (self.n_models, x.size()[0], self.n_classes)))
            if self.use_cuda:
                pred_dists.data = pred_dists.data.cuda()
            for i, model in enumerate(self.models):
                pred_dists[i] = F.softmax(model(x))
            pred_dists = torch.sum(pred_dists * self.weights.unsqueeze(-1).unsqueeze(-1).expand_as(pred_dists), dim=0).squeeze()
            return torch.log(pred_dists)
        except Exception as e:
            pdb.set_trace()

    def weight_experts(self, generator, data, n_steps, penalty):
        '''
        Run multiplicative weight update algorithm to find 
        optimal weights for each expert.

        Inputs:
            - n_steps: number of steps to run for
            - penalty: multiplicative penalty parameter

        Outputs:
            - w: weight for each expert
        '''
        w = self.w
        targs = data.outs
        n_ims = targs.size()[0]

        for t in xrange(n_steps):
            # generate noisy images against current ensemble
            # noisy images should be standardized
            corrupt_ims = torch.FloatTensor(generator.generate(data, self))
            if self.use_cuda:
                corrupt_ims = corrupt_ims.cuda()
            corrupt_ims = Variable(corrupt_ims)

            for i, model in enumerate(self.models):
                # make predictions for each model
                logits = model(corrupt_ims)
                preds = logits.data.max(1)[1].cpu()
                n_correct = preds.eq(targs).sum()

                # discount models that were wrong by penalty ^ (% wrong)
                w[i] *= (penalty ** ((n_ims - float(n_correct))/ n_ims))
            if self.use_cuda:
                w = w.cuda()
            self.weights = Variable(w / w.sum())
        self.w = w

        log.debug("\tFinished weighing experts! Min weight: %07.3f Max weight: %07.3f" % (self.weights.min().data[0], self.weights.max().data[0]))

        return
