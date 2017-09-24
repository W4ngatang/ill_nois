import pdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from src.codebase.utils.utils import log
from src.codebase_pytorch.utils.scheduler import ReduceLROnPlateau

'''

Modular CNN

'''

class ModularCNN(nn.Module):
    def __init__(self, args):
        '''
        Model variables
        '''
        super(ModularCNN, self).__init__()
        self.use_cuda = not args.no_cuda
        self.weights = nn.ModuleList()
        for i in xrange(args.n_modules):
            if not i:
                conv = nn.Conv2d(args.n_channels, args.n_kerns, 
                        kernel_size=args.kern_size, padding=1)
            else:
                conv = nn.Conv2d(args.n_kerns, args.n_kerns, 
                        kernel_size=args.kern_size, padding=1)
            if args.init_dist == 'uniform':
                init.uniform(conv.weight, -args.init_scale, args.init_scale)
                init.uniform(conv.bias, -args.init_scale, args.init_scale)
            elif args.init_dist == 'normal':
                init.normal(conv.weight, 0, args.init_scale)
                init.normal(conv.bias, 0, args.init_scale)
            else:
                raise NotImplementedError
            bn = nn.BatchNorm2d(args.n_kerns)
            self.weights.append(conv)
            self.weights.append(bn)

        self.fc = nn.Linear(args.n_kerns, args.n_classes)
        if args.init_dist == 'uniform':
            init.uniform(self.fc.weight, -args.init_scale, args.init_scale)
            init.uniform(self.fc.bias, .1*-args.init_scale, .1*args.init_scale)
        elif args.init_dist == 'normal':
            init.normal(self.fc.weight, 0, args.init_scale)
            init.normal(conv.bias, 0, .1*args.init_scale)
        else:
            raise NotImplementedError

    def forward(self, x):
        '''
        Model definition

        Following the modular CNN from Vinyals, each module is
            conv -> bn -> relu -> max_pool
        The tutorial seems to suggest
            conv -> max_pool -> relu
        '''
        for i in xrange(len(self.weights)/2):
            #x = F.relu(F.max_pool2d(self.weights[i][0](x), 2))
            x = F.max_pool2d(F.relu(self.weights[2*i+1](self.weights[2*i](x))), 2)
        x = self.fc(torch.squeeze(x))
        return x # returns the logits instead of a (log) distribution
        #return F.log_softmax(x)

    def train_model(self, args, tr_data, val_data, fh):
        '''
        Train the model according to the parameters specified in args.
        '''
        self.train()
        lr = args.lr
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=args.momentum)
            log(fh, "\tOptimizing with SGD with learning rate %.3f" % lr)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=lr)
        else:
            raise NotImplementedError
        scheduler = ReduceLROnPlateau(optimizer, 
                'max', factor=.5, patience=3, epsilon=1e-1)

        if args.load_model_from:
            self.load_state_dict(torch.load(args.load_model_from))

        start_time = time.time()
        val_loss, val_acc = self.evaluate(val_data)
        best_loss, best_acc = val_loss, val_acc
        log(fh, "\tInitial val loss: %.3f, \tval acc: %.2f \t(%.3f s)" %
                (val_loss, val_acc, time.time() - start_time))
        if args.save_model_to:
            torch.save(self.state_dict(), args.save_model_to)
            log(fh, "\t\tSaved model to %s" % args.save_model_to)

        for epoch in xrange(args.n_epochs):
            log(fh, "\tEpoch %d, \tlearning rate: %.3f" % (epoch+1, scheduler.get_lr()[0]))
            total_loss, total_correct = 0., 0.
            start_time = time.time()
            for batch_idx in xrange(tr_data.n_batches):
                ins, targs = tr_data[batch_idx]
                if self.use_cuda:
                    ins, targs = ins.cuda(), targs.cuda()
                ins, targs = Variable(ins), Variable(targs)
                optimizer.zero_grad()
                outs = self(ins)
                #loss = F.nll_loss(outs, targs)#, size_average=False)
                loss = F.cross_entropy(outs, targs)#, size_average=False)
                total_loss += loss.data[0]
                preds = outs.data.max(1)[1]
                total_correct += preds.eq(targs.data).cpu().sum()
                loss.backward()
                optimizer.step()

            val_loss, val_acc = self.evaluate(val_data)
            log(fh, "\t\tTraining loss: %.3f \taccuracy: %.2f"
                    % (total_loss / tr_data.n_batches, 
                        100. * total_correct / tr_data.n_ins))
            log(fh, "\t\tVal loss: %.3f \taccuracy: %.2f \t(%.3f s)"
                    % (val_loss, val_acc, time.time()-start_time))
            if val_acc > best_acc:
                if args.save_model_to:
                    torch.save(self.state_dict(), args.save_model_to)
                    log(fh, "\t\tSaved model to %s" % args.save_model_to)
                best_acc = val_acc
            scheduler.step(val_acc, epoch)

        if args.save_model_to:
            self.load_state_dict(torch.load(args.save_model_to))
        _, val_acc = self.evaluate(val_data)
        log(fh, "\tFinished training in %.3f s, \tBest validation accuracy: %.2f" % 
                (time.time() - start_time, val_acc))

    def evaluate(self, data):
        '''
        Evaluate model on data, usually either validation or test
        '''
        self.eval()
        total_loss, total_correct = 0., 0.
        for batch_idx in xrange(data.n_batches):
            ins, targs = data[batch_idx]
            if self.use_cuda:
                ins, targs = ins.cuda(), targs.cuda()
            ins, targs = Variable(ins, volatile=True), Variable(targs)
            outs = self(ins)
            #total_loss += F.nll_loss(outs, targs, size_average=False).data[0]
            total_loss += F.cross_entropy(outs, targs, size_average=False).data[0]
            preds = outs.data.max(1)[1]
            total_correct += preds.eq(targs.data).cpu().sum()
        return total_loss / data.n_ins, \
                100. * total_correct / data.n_ins
