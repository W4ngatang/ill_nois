import logging as log
import pdb
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from src.codebase_pytorch.models.model import Model
from src.codebase_pytorch.utils.dataset import Dataset

class Ensemble(Model):
    '''
    Optimization-based generator using an ensemble of models
    '''

    def __init__(self, args, models):
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
        self.args = args
        self.w = w = torch.ones(n_models) / float(n_models)
        if self.use_cuda:
            w = w.cuda()
            for model in models:
                model.cuda()
        self.weights = Variable(w / w.sum())
        '''
        pred_dists = torch.zeros(n_models, batch_size, n_classes)
        if self.use_cuda:
            pred_dists = pred_dists.cuda()
        self.pred_dists = Variable(pred_dists)
        '''


    def forward(self, x):
        '''
        NB: Ensemble returns (log of) a valid probability distribution INSTEAD of logits
                in order to return the weighted probability distribution.
            But ensemble ASSUMES that each model returns LOGITS
            This means you should not use the train_model method because it expects logits
        '''
        try:
            #pred_dists = self.pred_dists
            #pred_dists.data.zero_()
            pred_dists = Variable(torch.zeros(self.n_models, x.size()[0], self.n_classes))
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
        chunk_size = 10

        log.debug("\tWeighting experts for %d steps" % n_steps)
        loss_history = list()


        for t in xrange(n_steps):
            
            log.debug("Iteration {}, Weights {}".format(t, list(w)))
            # generate noisy images against current ensemble
            # noisy images should be standardized
            #print "Weights, ", type(w.cpu().numpy()), w.cpu().numpy(), w.sum()
            # curr_expert = np.random.choice(self.n_models, 1, list(w.cpu().numpy()))[0]
            #print  "CURRENT EXP", curr_expert
            # curr_expert = self.models[curr_expert]
            
            
            corrupt_ims_tensor = torch.FloatTensor(generator.generate(data, self))

            if self.use_cuda:
                corrupt_ims_tensor = corrupt_ims_tensor.cuda()
           
            corrupt_ims = Variable(corrupt_ims_tensor)
            
            for i, model in enumerate(self.models):
                
                n_correct = 0
                
                for pos in xrange(0, len(corrupt_ims), chunk_size):
                    
                    data_batch = corrupt_ims[pos: pos + chunk_size]
                    labels_batch = targs[pos: pos + chunk_size]

                    # make predictions for each model
                    logits = model(data_batch)
                    preds = logits.data.max(1)[1].cpu()
                    n_correct += preds.eq(labels_batch).sum()

                # discount models that were wrong by penalty ^ (% wrong)
                w[i] *= (1 - penalty) ** ((n_ims - float(n_correct)) / n_ims)
          
            if self.use_cuda:
                w = w.cuda()
          
            # renormalize the weights of w
            for j in xrange(len(w) - 1):
                w[j] = w[j] / w.sum()
            
            w[-1] = 1 - w[:-1].sum()

            self.weights = Variable(w / w.sum())

            corrupt_dataset = Dataset(corrupt_ims_tensor.cpu(), targs, self.args.generator_batch_size, self.args)
            _, clean_top1, _ = self.evaluate(corrupt_dataset)

            loss_history.append((100 - clean_top1, w))


        # pick the best history
        print loss_history
        self.w = min(loss_history, key = lambda x: x[0])[1]
        self.weights = Variable(self.w)

        log.debug("\tFinished weighing experts! Min weight: %07.3f Max weight: %07.3f" % (self.weights.min().data[0], self.weights.max().data[0]))

        return loss_history
