import time
import logging as log
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from src.codebase_pytorch.utils.scheduler import ReduceLROnPlateau, LambdaLR


class Model(nn.Module):
    '''
    My model class. Interface has:
        - train_model: train model
        - evaluate: evaluate
        - predict: get class prediction for input
        - get_gradient: gets gradient of outputs wrt input
    '''

    def __init__(self):
        super(Model, self).__init__()


    def forward(self):
        '''
        Should be overwritten by inheritors
        '''
        raise NotImplementedError("Not a runnable class; inherit from this.")


    def train_model(self, args, tr_data, val_data, fh):
        '''
        Train the model according to the parameters specified in args.
        NB: model is always converted to .eval() mode at the end of training.
        '''

        self.train()

        # Optimizer
        lr = args.lr
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=args.momentum, 
                                    weight_decay=args.weight_decay)
            log.debug("\tOptimizing with SGD with learning rate %.3f" % lr)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
            log.debug("\tOptimizing with adam with learning rate %.3f" % lr)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=lr,
                                        weight_decay=args.weight_decay)
            log.debug("\tOptimizing with adagrad with learning rate %.3f" % lr)
        else:
            raise NotImplementedError

        # LR scheduler
        if args.lr_scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer,
                    'max', factor=.5, patience=3, threshold=1e-1)
        elif args.lr_scheduler == 'lambda':
            lr_lambda = lambda epoch: (1 - float(epoch) / args.n_epochs)
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            raise NotImplementedError

        start_time = time.time()
        val_loss, val_acc, val_top5 = self.evaluate(val_data)
        best_loss, best_acc = val_loss, val_acc
        log.debug("\tInitial val loss: %.3f \tval acc: %.2f \tval top5: %.2f \t(%.3f s)" % 
                (val_loss, val_acc, val_top5, time.time() - start_time))
        if args.save_model_to:
            torch.save(self.state_dict(), args.save_model_to)
            log.debug("\t\tSaved model to %s" % args.save_model_to)

        for epoch in xrange(args.n_epochs):
            log.debug("\tEpoch %d, \tlearning rate: %.3f" % (epoch+1, scheduler.get_lr()[0]))
            total_loss, total_correct = 0., 0.
            start_time = time.time()
            for batch_idx in xrange(tr_data.n_batches):
                ins, targs = tr_data[batch_idx]
                if self.use_cuda:
                    ins, targs = ins.cuda(), targs.cuda()
                ins, targs = Variable(ins), Variable(targs)
                optimizer.zero_grad()
                outs = self(ins)
                loss = F.cross_entropy(outs, targs)
                total_loss += loss.data[0]
                preds = outs.data.max(1)[1]
                total_correct += preds.eq(targs.data).cpu().sum()
                loss.backward()
                optimizer.step()

            val_loss, val_acc, val_top5 = self.evaluate(val_data)
            log.debug("\t\tTraining loss: %.3f \ttop1 accuracy: %.2f"
                    % (total_loss / tr_data.n_batches,
                        100. * total_correct / tr_data.n_ins))
            log.debug("\t\tVal loss: %.3f \ttop1 accuracy: %.2f \ttop5: %.2f \t(%.3f s)" % 
                    (val_loss, val_acc, val_top5, time.time()-start_time))
            if val_acc > best_acc:
                if args.save_model_to:
                    torch.save(self.state_dict(), args.save_model_to)
                    log.debug("\t\tSaved model to %s" % args.save_model_to)
                best_acc = val_acc
            scheduler.step(epoch)

        if args.save_model_to:
            self.load_state_dict(torch.load(args.save_model_to))
        log.debug("\tFinished training in %.3s s!" % (time.time()-start_time))

        self.eval()


    def evaluate(self, data):
        '''
        Evaluate model on data, usually either validation or test
        '''
        was_train = self.training
        self.eval()
        total_loss, top1_correct, top5_correct = 0., 0., 0.
        for batch_idx in xrange(data.n_batches):
            ins, targs = data[batch_idx]
            if self.use_cuda:
                ins, targs = ins.cuda(), targs.cuda()
            ins, targs = Variable(ins, volatile=True), Variable(targs)
            outs = self(ins)
            #total_loss += F.cross_entropy(outs, targs, size_average=False).data[0]
            total_loss += F.nll_loss(outs, targs, size_average=False).data[0] # still deciding on output of forward()
            _, preds = outs.topk(5, 1, True, True)
            preds = preds.t()
            correct = preds.eq(targs.view(1, -1).expand_as(preds))
            top1_correct += correct[:1].view(-1).float().sum(0)
            top5_correct += correct[:5].view(-1).float().sum(0)
        top1_correct = top1_correct.data[0]
        top5_correct = top5_correct.data[0]
        if was_train:
            self.train()
        return total_loss / data.n_ins, 100. * top1_correct / data.n_ins, \
            100. * top5_correct / data.n_ins


    def predict(self, data):
        '''
        Get class predictions for data
        '''
        self.eval()
        preds, dists = [], []
        for batch_idx in xrange(data.n_batches):
            ins, targs = data[batch_idx]
            if self.use_cuda:
                ins, targs = ins.cuda(), targs.cuda()
            ins, targs = Variable(ins, volatile=True), Variable(targs)
            outs = self(ins)
            preds.append(outs.data.max(1)[1].cpu().numpy())
            dists.append(outs.data.cpu().numpy())
        return np.vstack(preds), np.vstack(dists)


    def get_gradient(self, ins, targs):
        '''
        Return gradient of loss wrt ins
        '''
        self.eval()
        if self.use_cuda:
            ins, targs = ins.cuda(), targs.cuda()
        ins, targs = Variable(ins, requires_grad=True), Variable(targs)
        outs = self(ins)
        #loss = F.cross_entropy(outs, targs)
        loss = F.nll_loss(outs, targs)
        loss.backward(retain_variables=True)
        return ins.grad.data.cpu().numpy()
