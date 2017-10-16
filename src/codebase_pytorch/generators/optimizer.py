import time
import logging as log
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from src.codebase_pytorch.utils.dataset import Dataset


class OptimizationGenerator(nn.Module):
    '''
    Optimization-based generator
    '''

    def __init__(self, args, normalize):
        '''

        '''
        super(OptimizationGenerator, self).__init__()
        self.use_cuda = args.use_cuda
        self.targeted = (args.target != 'none') and (args.target != 'top5')
        self.n_classes = args.n_classes
        self.k = -1. * args.generator_confidence
        self.n_binary_search_steps = args.n_binary_search_steps
        self.init_const = args.generator_init_opt_const
        self.early_abort = args.early_abort
        self.batch_size = args.generator_batch_size
        self.optimizer = args.generator_optimizer
        self.lr = args.generator_lr
        self.n_generator_steps = args.n_generator_steps
        self.mean, self.std = normalize[0], normalize[1]

        self.debug = args.debug

    def forward(self, tanh_x, x, w, c, labels, model, mean=None, std=None):
        '''
        Function to optimize
            - tanh_x is original image in arctanh space
            - x is original image in [0,1]^d
            - w is the noise in arctanh space
            - labels should be binary (possibly multi-hot)
            - model should output log probabilities
        '''

        corrupt_im = .5 * (F.tanh(w + tanh_x) + 1) # in [0,1]
        input_im = corrupt_im
        if mean is not None:
            input_im = (corrupt_im - mean) / std
        log_prob_dist = model(input_im)
        prob_dist = torch.exp(log_prob_dist)
        if self.targeted:
            class_loss = -torch.sum(torch.log(prob_dist + 1e-6) * labels, dim=1)
        else:
            class_loss = -torch.sum(torch.log(1. - prob_dist + 1e-6) * 
                                   labels, dim=1)
        dist_loss = torch.sum(torch.pow(corrupt_im - x, 2).view(self.batch_size, -1), dim=1)
        return torch.sum(dist_loss + c * class_loss), dist_loss, \
                corrupt_im, log_prob_dist

    def generate(self, data, model):
        '''
        Generate 
        '''

        corrupt_ims = []
        log.debug("\tGenerating for %d binary search steps and %d optimization steps" % 
                (self.n_binary_search_steps, self.n_generator_steps))

        for batch_idx in xrange(data.n_batches):
            log.debug("\tGenerating noise for batch %d of %d" % 
                       (batch_idx+1, data.n_batches))
            ins, targs = data[batch_idx]
            batch_corrupt_ims = self.generate_batch((ins, targs), model)
            corrupt_ims.append(batch_corrupt_ims)

        return np.vstack(corrupt_ims)

    def generate_batch(self, data, model):
        '''
        Generate adversarial noise using fast gradient method.

        inputs:
            - data: tuple of (ins, outs) or Dataset class
            - model: a model class
            - args: argparse object with training parameters
        outputs:
            - corrupt_ims: n_ims x im_size x im_size x n_channels (normalized)

        TODO
            - handle non batch_size inputs (smaller and larger)
        '''

        def compare(x, y):
            '''
            Check if predicted class is target class
                or not targeted class if untargeted
            '''
            return x == y if self.targeted else x != y

        if isinstance(data, tuple):
            ins = data[0].numpy() if isinstance(data[0], torch.FloatTensor) else data[0]
            outs = data[1].numpy() if not isinstance(data[1], np.ndarray) else data[1]
        elif isinstance(data, Dataset):
            ins = data.ins.numpy()
            outs = data.outs.numpy()
        else:
            raise TypeError("Invalid data format")

        # convert inputs to arctanh space
        if self.mean is not None:
            ins = (ins * self.std) + self.mean
        ins = np.clip(ins, 0., 1.)
        assert ins.max() <= 1.0 and ins.min() >= 0.0 # in [0, 1]
        tanh_ins = 1.999999 * (ins - .5) # in (-1, 1)
        tanh_ins = torch.FloatTensor(np.arctanh(tanh_ins)) # in arctanh space
        ins = torch.FloatTensor(ins)

        # make targs one-hot
        one_hot_targs = np.zeros((outs.shape[0], self.n_classes))
        if len(outs.shape) > 1:
            for i in xrange(outs.shape[1]):
                one_hot_targs[np.arange(outs.shape[0]), outs[:,i].astype(int)] = 1
            outs = torch.LongTensor(outs[:,0])
        else:
            one_hot_targs[np.arange(outs.shape[0]), outs.astype(int)] = 1
            outs = torch.LongTensor(outs)
        one_hot_targs = torch.FloatTensor(one_hot_targs)

        batch_size = self.batch_size
        lower_bounds = np.zeros(batch_size)
        upper_bounds = np.ones(batch_size) * 1e10
        opt_consts = torch.ones((batch_size,1)) * self.init_const

        overall_best_ims = np.zeros(ins.size())
        overall_best_dists = [1e10] * batch_size
        overall_best_classes = [-1] * batch_size

        w = torch.zeros(ins.size())

        if self.use_cuda:
            tanh_ins, ins, one_hot_targs, w, opt_consts = tanh_ins.cuda(), \
                ins.cuda(), one_hot_targs.cuda(), w.cuda(), opt_consts.cuda()
        tanh_ins, ins, one_hot_targs, w, opt_consts = \
                Variable(tanh_ins), Variable(ins), Variable(one_hot_targs), \
                Variable(w, requires_grad=True), Variable(opt_consts)

        if self.mean is not None:
            mean = torch.FloatTensor(self.mean)
            std = torch.FloatTensor(self.std)
            if self.use_cuda:
                mean, std = mean.cuda(), std.cuda()
            mean = Variable(mean.expand_as(ins))
            std = Variable(std.expand_as(ins))
        else:
            mean, std = None, None

        start_time = time.time()
        for b_step in xrange(self.n_binary_search_steps):
            log.debug(('\tBinary search step %d \tavg const: %s' 
                     '\tmin const: %s \tmax const: %s') % 
                     (b_step, opt_consts.mean().data[0], 
                     opt_consts.min().data[0], opt_consts.max().data[0]))

            # lazy way to reset optimizer parameters
            w.data.zero_()
            if self.optimizer == 'sgd':
                optimizer = optim.SGD([w], lr=self.lr)
            elif self.optimizer == 'adam':
                optimizer = optim.Adam([w], lr=self.lr)
            elif self.optimizer == 'adagrad':
                optimizer = optim.Adagrad([w], lr=self.lr)
            else:
                raise NotImplementedError

            best_dists = [1e10] * batch_size
            best_classes = [-1] * batch_size

            # repeat binary search one more time?

            prev_loss = 1e6
            for step in xrange(self.n_generator_steps):
                optimizer.zero_grad()
                obj, dists, corrupt_ims, pred_dists = \
                    self(tanh_ins, ins, w, opt_consts, 
                            one_hot_targs, model, mean, std)
                total_loss = obj.data[0]
                obj.backward()
                optimizer.step()

                if not (step % (self.n_generator_steps / 10.)) and step:
                    # Logging every 1/10
                    _, preds = pred_dists.topk(1, 1, True, True)
                    n_correct = torch.sum(torch.eq(preds.data.cpu(), outs))
                    if not self.targeted:
                        n_correct = batch_size - n_correct
                    log.debug(('\t\tStep %d \tobjective: %010.5f '
                             '\tavg dist: %010.5f'
                             '\tn targeted class: %d'
                             '\t(%.3f s)') % (step, total_loss, 
                             dists.data.pow(2).view(batch_size, -1).mean(dim=-1).sqrt().mean(), 
                             n_correct, time.time() - start_time))

                    # Check for early abort
                    if self.early_abort and total_loss > prev_loss*.9999:
                        log.debug('\t\tAborted search because stuck')
                        break
                    prev_loss = total_loss
                    #scheduler.step(total_loss, step)

                # bookkeeping
                for e, (dist, pred_dist, im) in \
                        enumerate(zip(dists, pred_dists, corrupt_ims)):
                    #logit[outs[e]] += self.k
                    pred = np.argmax(pred_dist.data.cpu().numpy())
                    if not compare(pred, outs[e]): # if not the targeted class
                        continue
                    if dist < best_dists[e]: # if smaller noise within bsearch step
                        best_dists[e] = dist
                        best_classes[e] = pred
                    if dist < overall_best_dists[e]: # if smaller noise overall
                        overall_best_dists[e] = dist
                        overall_best_classes[e] = pred
                        overall_best_ims[e] = im.data.cpu().numpy()

            # binary search stuff
            for e in xrange(batch_size):
                if compare(best_classes[e], outs[e]) and best_classes[e] != -1:
                    # success; looking for lower c
                    upper_bounds[e] = \
                        min(upper_bounds[e], opt_consts.data[e][0])
                    if upper_bounds[e] < 1e9:
                        opt_consts.data[e][0] = \
                            (lower_bounds[e] + upper_bounds[e]) / 2
                else: # failure, search with greater c
                    lower_bounds[e] = \
                        max(lower_bounds[e], opt_consts.data[e][0])
                    if upper_bounds[e] < 1e9:
                        opt_consts.data[e][0] = \
                            (lower_bounds[e] + upper_bounds[e]) / 2
                    else:
                        opt_consts.data[e][0] *= 10

        if self.mean is not None:
            overall_best_ims = (overall_best_ims - self.mean) / self.std
        return overall_best_ims
