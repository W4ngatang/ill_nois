import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from src.codebase.utils.utils import log as log
from src.codebase_pytorch.utils.dataset import Dataset


class CarliniL2Generator(nn.Module):
    '''
    Class for generating noise using method in Carlini and Wagner, '17

    TODO
        - deal with discretization
        - multiple starting point gradient descent
    '''

    def __init__(self, args, normalize):
        '''

        '''
        super(CarliniL2Generator, self).__init__()
        self.use_cuda = args.cuda
        self.targeted = args.target != 'none'
        self.k = -1. * args.generator_confidence
        self.binary_search_steps = args.n_binary_search_steps
        self.init_const = args.generator_init_opt_const
        self.early_abort = args.early_abort
        self.batch_size = args.generator_batch_size # rename to batch_size
        self.mean = normalize[0]
        self.std = normalize[1]

    def forward(self, tanh_x, x, w, c, labels, model, mean=None, std=None):
        '''
        Function to optimize
            - Labels should be one-hot
            - tanh_x should be inputs in tanh space
            - x should be images in [0,1]^d (unnormalized)
        '''
        corrupt_im = .5 * (F.tanh(w + tanh_x) + 1) # puts into [0,1]
        input_im = corrupt_im
        if mean is not None: # if model expects normalized input
            input_im = (corrupt_im - mean) / std
        logits = model(input_im)
        target_logit = torch.sum(logits * labels, dim=1)
        second_logit = torch.max(logits*(1.-labels)-(labels*10000), dim=1)[0]
        if self.targeted:
            class_loss = torch.clamp(second_logit - target_logit, min=self.k)
        else:
            class_loss = torch.clamp(target_logit - second_logit, min=self.k)
        dist_loss = torch.sum(torch.pow(corrupt_im - x, 2).view(self.batch_size, -1), dim=1)  # dist we care about is between unnormalized (in [0,1]^d) images
        return torch.sum(dist_loss + c * class_loss), dist_loss, corrupt_im, logits

    def generate(self, data, model, args, fh):
        '''
        Generate adversarial noise using fast gradient method.

        inputs:
            - data: tuple of (ins, outs) or Dataset class
            - model: a model class
            - args: argparse object with training parameters
            - fh: file handle for logging progress
        outputs:
            - noise: n_ims x im_size x im_size x n_channels

        TODO
            - handle non batch_size inputs (smaller and larger)
            - in converting to tanh space, seems to expect values in (-1, 1)
        '''

        def compare(x, y):
            '''
            Check if predicted class is target class
                or not targeted class if untargeted
            '''
            return x == y if self.targeted else x != y

        if isinstance(data, tuple):
            ins = ins.numpy() if isinstance(data[0], torch.FloatTensor) else data[0]
            outs = outs.numpy() if not isinstance(data[1], np.ndarray) else data[1]
        elif isinstance(data, Dataset):
            ins = data.ins.numpy()
            outs = data.outs.numpy()
        else:
            raise TypeError("Invalid data format")

        # convert inputs to arctanh space
        if self.mean is not None:
            ins = (ins * self.std) + self.mean
        ins = ins - ins.min()
        ins = ins / ins.max()
        assert ins.max() <= 1.0 and ins.min() >= 0.0 # in [0, 1]
        tanh_ins = 1.999999 * (ins - .5) # in (-1, 1)
        tanh_ins = torch.FloatTensor(np.arctanh(tanh_ins)) # in tanh space
        ins = torch.FloatTensor(ins)

        # make targs one-hot
        one_hot_targs = np.zeros((outs.shape[0], args.n_classes))
        one_hot_targs[np.arange(outs.shape[0]), outs.astype(int)] = 1
        one_hot_targs = torch.FloatTensor(one_hot_targs)
        outs = torch.LongTensor(outs)

        batch_size = self.batch_size
        lower_bounds = np.zeros(batch_size)
        upper_bounds = np.ones(batch_size) * 1e10
        opt_consts = torch.ones((batch_size,1)) * self.init_const

        overall_best_ims = np.zeros(ins.size())
        overall_best_dists = [1e10] * batch_size
        overall_best_classes = [-1] * batch_size

        # variable to optimize
        w = torch.zeros(ins.size())

        if self.use_cuda:
            tanh_ins, ins, one_hot_targs, w, opt_consts = \
                tanh_ins.cuda(), ins.cuda(), one_hot_targs.cuda(), \
                w.cuda(), opt_consts.cuda()
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
        for b_step in xrange(self.binary_search_steps):
            log(fh, '\tBinary search step %d \tavg const: %s \tmin const: %s \tmax const: %s' % (b_step, opt_consts.mean().data[0], opt_consts.min().data[0], opt_consts.max().data[0]))

            w.data.zero_()
            # lazy way to reset optimizer parameters
            if args.generator_optimizer == 'sgd':
                optimizer = optim.SGD([w], lr=args.generator_lr, momentum=args.momentum)
            elif args.generator_optimizer == 'adam':
                optimizer = optim.Adam([w], lr=args.generator_lr)
            elif args.generator_optimizer == 'adagrad':
                optimizer = optim.Adagrad([w], lr=args.generator_lr)
            else:
                raise NotImplementedError
            #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=.5, patience=3, threshold=1e-3)

            best_dists = [1e10] * batch_size
            best_classes = [-1] * batch_size

            # repeat binary search one more time?

            prev_loss = 1e6
            for step in xrange(args.n_generator_steps):
                optimizer.zero_grad()
                obj, dists, corrupt_ims, logits, = \
                    self(tanh_ins, ins, w, opt_consts, \
                        one_hot_targs, model, mean, std)
                total_loss = obj.data[0]
                obj.backward()
                optimizer.step()

                if not (step % (args.n_generator_steps / 10.)) and step:
                    # Logging every 1/10
                    _, preds = logits.topk(1, 1, True, True)
                    n_correct = torch.sum(torch.eq(preds.data.cpu(), outs))
                    if not self.targeted:
                        n_correct = batch_size - n_correct
                    log(fh, '\t\tStep %d \tobjective: %.3f \tavg dist: %.3f \tn targeted class: %d \t(%.3f s)' % (step, total_loss, torch.mean(dists.data), n_correct, time.time() - start_time))

                    # Check for early abort
                    if self.early_abort and total_loss > prev_loss*.9999:
                        log(fh, '\t\tAborted search because stuck')
                        break
                    prev_loss = total_loss
                    #scheduler.step(total_loss, step)

                # bookkeeping
                for e, (dist, logit, im) in enumerate(zip(dists, logits, corrupt_ims)):
                    #logit[outs[e]] += self.k
                    pred = np.argmax(logit.data.cpu().numpy())
                    if not compare(pred, outs[e]): # if not the targeted class, continue
                        continue
                    if dist < best_dists[e]: # if smaller noise within the binary search step
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
