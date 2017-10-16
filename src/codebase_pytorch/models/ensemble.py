import logging as log
import pdb

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.codebase_pytorch.models.model import Model


class Ensemble(Model):
    '''
    Optimization-based generator using an ensemble of models
    '''

    def __init__(self, args, models, aggregate=1):
        '''

        '''
        super(Ensemble, self).__init__()
        self.use_cuda = args.use_cuda
        self.batch_size = batch_size = args.batch_size
        self.n_classes = n_classes = args.n_classes
        for model in models:
            model.eval()
        self.models = models
        self.n_models = n_models = len(models)
        self.w = w = torch.ones(n_models)
        if self.use_cuda:
            w = w.cuda()
        self.weights = Variable(w / w.sum())
        self.aggregate = aggregate
        '''
        pred_dists = torch.zeros(n_models, batch_size, n_classes)
        if self.use_cuda:
            pred_dists = pred_dists.cuda()
        self.pred_dists = Variable(pred_dists)
        '''


    def forward(self, x):
        '''
        If aggregate forward() returns n_examples x n_classes tensor
        Else, returns n_models x n_examples x n_classes tensor 
        Both containing log probabilities.
        '''
        #pred_dists = self.pred_dists
        #pred_dists.data.zero_()
        pred_dists = Variable(torch.zeros(self.n_models, x.size()[0], self.n_classes))
        if self.use_cuda:
            pred_dists.data = pred_dists.data.cuda()
        if self.aggregate:
            for i, model in enumerate(self.models):
                pred_dists[i] = F.softmax(model(x))
            pred_dists = torch.sum(pred_dists * self.weights.unsqueeze(-1).unsqueeze(-1).expand_as(pred_dists), dim=0).squeeze()
            return torch.log(pred_dists)
        else:
            for i, model in enumerate(self.models):
                pred_dists[i] = F.log_softmax(model(x))
            return pred_dists

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
        old_w = self.w
        targs = data.outs
        n_ims = targs.size()[0]
        batch_size = self.batch_size

        log.debug("\tWeighting experts for %d steps" % n_steps)
        for t in xrange(n_steps):
            new_w = torch.zeros(old_w.size())
            loss = 0.

            # generate noisy images against current ensemble
            # noisy images should be standardized
            corrupt_ims = torch.FloatTensor(generator.generate(data, self))
            if self.use_cuda:
                corrupt_ims = corrupt_ims.cuda()
            corrupt_ims = Variable(corrupt_ims)

            for i, model in enumerate(self.models):
                preds = torch.zeros(n_ims)
                for batch_idx in range(0, n_ims, batch_size):
                    # make predictions for each model
                    logits = model(corrupt_ims[batch_idx:batch_idx+batch_size])
                    batch_preds = logits.data.max(1)[1].cpu()
                    preds[batch_idx:batch_idx+batch_size] = batch_preds 
                n_correct = preds.long().eq(targs).sum()
                frac_wrong = (n_ims - float(n_correct)) / n_ims

                # learner's loss is weighted average of 0-1 loss
                loss += self.weights[i] * frac_wrong

                # discount models that were wrong by penalty ^ (# wrong)
                new_w[i] = old_w[i] * (penalty ** frac_wrong)

            del old_w, self.weights

            if self.use_cuda:
                new_w = new_w.cuda()
            self.weights = Variable(new_w / new_w.sum())
            old_w = new_w
            log.debug("\t\tWeighting step %d" % (t+1))
            log.debug("\t\tLearner loss: %07.5f" % loss.data[0])
            log.debug("\t\tWeights: %s" % ", ".join([str(w) for w in self.weights.data]))
        self.w = new_w
        log.debug("\tFinished weighing experts! Min weight: %07.3f Max weight: %07.3f" % (self.weights.min().data[0], self.weights.max().data[0]))
        log.debug("\t\tweights: %s" % ", ".join([str(w) for w in self.weights.data]))

        return
