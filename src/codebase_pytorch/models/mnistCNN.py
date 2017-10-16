import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from src.codebase.utils.utils import log

'''

MNIST architecture from Carlini and Wagner, 2017

'''

class MNISTCNN(nn.Module):
    def __init__(self, args):
        '''
        Model variables
        '''
        super(MNISTCNN, self).__init__()
        self.use_cuda = not args.no_cuda

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

        '''
        if args.init_dist == 'uniform':
            init.uniform(self.fc1.weight, -args.init_scale, args.init_scale)
            init.uniform(self.fc1.bias, .1*-args.init_scale, .1*args.init_scale)
        elif args.init_dist == 'normal':
            init.normal(self.fc.weight, 0, args.init_scale)
            init.normal(conv.bias, 0, .1*args.init_scale)
        else:
            raise NotImplementedError
        '''

    def forward(self, x):
        '''
        Model definition

        Following the modular CNN from Vinyals, each module is
            conv -> bn -> relu -> max_pool
        The tutorial seems to suggest
            conv -> max_pool -> relu
        '''
        x = F.max_pool2d(F.relu(self.conv2(F.relu(self.conv1(x)))), 2)
        x = F.max_pool2d(F.relu(self.conv4(F.relu(self.conv3(x)))), 2)
        x = x.view(x.size()[0], -1)
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(torch.squeeze(x))))))
        return x # returns the logits instead of a (log) distribution

    def train_model(self, args, tr_data, val_data, fh):
        '''
        Train the model according to the parameters specified in args.
        '''
        self.train()
        lr = args.lr
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
            log(fh, "\tOptimizing with SGD with learning rate %.3f" % lr)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=args.lr)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=args.lr)
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                'max', factor=.5, patience=10, threshold=.1, cooldown=0)

        if args.load_model_from:
            self.load_state_dict(torch.load(args.load_model_from))

        start_time = time.time()
        val_loss, val_acc = self.evaluate(val_data)
        best_loss, best_acc = val_loss, val_acc
        last_acc = val_acc
        log(fh, "\tInitial val loss: %.3f, \tval acc: %.2f \t(%.3f s)" %
                (val_loss, val_acc, time.time() - start_time))
        if args.save_model_to:
            torch.save(self.state_dict(), args.save_model_to)
            log(fh, "\t\tSaved model to %s" % args.save_model_to)

        for epoch in xrange(args.n_epochs):
            log(fh, "\tEpoch %d, \tlearning rate: %.3f" % 
                    (epoch+1, scheduler.get_lr()[0]))
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
            scheduler.step(val_acc)
            last_acc = val_acc

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
            total_loss += F.cross_entropy(outs, targs, size_average=False).data[0]
            preds = outs.data.max(1)[1]
            total_correct += preds.eq(targs.data).cpu().sum()
        return total_loss / data.n_ins, \
                100. * total_correct / data.n_ins

    def predict(self, data):
        '''
        Get predictions for data
        '''
        self.eval()
        predictions = []
        for batch_idx in xrange(data.n_batches):
            ins, targs = data[batch_idx]
            if self.use_cuda:
                ins, targs = ins.cuda(), targs.cuda()
            ins, targs = Variable(ins, volatile=True), Variable(targs)
            outs = self(ins)
            predictions.append(outs.data.cpu().numpy())
        return np.vstack(predictions)

    def get_gradient(self, ins, targs):
        self.eval()
        if self.use_cuda:
            ins, targs = ins.cuda(), targs.cuda()
        ins, targs = Variable(ins, requires_grad=True), Variable(targs)
        outs = self(ins)
        loss = F.cross_entropy(outs, targs)
        loss.backward(retain_variables=True)
        return ins.grad.data.cpu().numpy()
