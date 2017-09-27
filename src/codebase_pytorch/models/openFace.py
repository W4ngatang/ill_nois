import time
from collections import OrderedDict

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from src.codebase.utils.utils import log
from src.codebase_pytorch.utils.SpatialCrossMapLRN_temp import SpatialCrossMapLRN_temp
from src.codebase_pytorch.utils.scheduler import ReduceLROnPlateau


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

def Conv2d(in_dim, out_dim, kernel, stride, padding):
    l = torch.nn.Conv2d(in_dim, out_dim, kernel, stride=stride, padding=padding)
    return l

def BatchNorm(dim):
    l = torch.nn.BatchNorm2d(dim)
    return l

def CrossMapLRN(size, alpha, beta, k=1.0):
    lrn = SpatialCrossMapLRN_temp(size, alpha, beta, k)
    n = Lambda( lambda x,lrn=lrn: Variable(lrn.forward(x.data).cuda()) if x.data.is_cuda else Variable(lrn.forward(x.data)) )
    return n

def Linear(in_dim, out_dim):
    l = torch.nn.Linear(in_dim, out_dim)
    return l

class Inception(nn.Module):
    def __init__(self, inputSize, kernelSize, kernelStride, outputSize, reduceSize, pool, useBatchNorm, reduceStride=None, padding=True):
        super(Inception, self).__init__()
        #
        self.seq_list = []
        self.outputSize = outputSize

        #
        # 1x1 conv (reduce) -> 3x3 conv
        # 1x1 conv (reduce) -> 5x5 conv
        # ...
        for i in range(len(kernelSize)):
            od = OrderedDict()
            # 1x1 conv
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            # nxn conv
            pad = int(numpy.floor(kernelSize[i] / 2)) if padding else 0
            od['4_conv'] = Conv2d(reduceSize[i], outputSize[i], kernelSize[i], kernelStride[i], pad)
            if useBatchNorm:
                od['5_bn'] = BatchNorm(outputSize[i])
            od['6_relu'] = nn.ReLU()
            #
            self.seq_list.append(nn.Sequential(od))

        ii = len(kernelSize)
        # pool -> 1x1 conv
        od = OrderedDict()
        od['1_pool'] = pool
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od['2_conv'] = Conv2d(inputSize, reduceSize[i], (1,1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['3_bn'] = BatchNorm(reduceSize[i])
            od['4_relu'] = nn.ReLU()
        #
        self.seq_list.append(nn.Sequential(od))
        ii += 1

        # reduce: 1x1 conv (channel-wise pooling)
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od = OrderedDict()
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1,1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            self.seq_list.append(nn.Sequential(od))

        self.seq_list = nn.ModuleList(self.seq_list)


    def forward(self, input):
        x = input

        ys = []
        target_size = None
        depth_dim = 0
        for seq in self.seq_list:
            #print(seq)
            #print(self.outputSize)
            #print('x_size:', x.size())
            y = seq(x)
            y_size = y.size()
            #print('y_size:', y_size)
            ys.append(y)
            #
            if target_size is None:
                target_size = [0] * len(y_size)
            #
            for i in range(len(target_size)):
                target_size[i] = max(target_size[i], y_size[i])
            depth_dim += y_size[1]

        target_size[1] = depth_dim
        #print('target_size:', target_size)

        for i in range(len(ys)):
            y_size = ys[i].size()
            pad_l = int((target_size[3] - y_size[3]) // 2)
            pad_t = int((target_size[2] - y_size[2]) // 2)
            pad_r = target_size[3] - y_size[3] - pad_l
            pad_b = target_size[2] - y_size[2] - pad_t
            ys[i] = F.pad(ys[i], (pad_l, pad_r, pad_t, pad_b))

        output = torch.cat(ys, 1)

        return output


class OpenFace(nn.Module):
    def __init__(self, args):
        super(openFace, self).__init__()
        self.use_cuda = not args.no_cuda

        self.layer1 = Conv2d(3, 64, (7,7), (2,2), (3,3))
        self.layer2 = BatchNorm(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        self.layer5 = CrossMapLRN(5, 0.0001, 0.75)
        self.layer6 = Conv2d(64, 64, (1,1), (1,1), (0,0))
        self.layer7 = BatchNorm(64)
        self.layer8 = nn.ReLU()
        self.layer9 = Conv2d(64, 192, (3,3), (1,1), (1,1))
        self.layer10 = BatchNorm(192)
        self.layer11 = nn.ReLU()
        self.layer12 = CrossMapLRN(5, 0.0001, 0.75)
        self.layer13 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        self.layer14 = Inception(192, (3,5), (1,1), (128,32), (96,16,32,64), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer15 = Inception(256, (3,5), (1,1), (128,64), (96,32,64,64), nn.LPPool2d(2, (3,3), stride=(3,3)), True)
        self.layer16 = Inception(320, (3,5), (2,2), (256,64), (128,32,None,None), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer17 = Inception(640, (3,5), (1,1), (192,64), (96,32,128,256), nn.LPPool2d(2, (3,3), stride=(3,3)), True)
        self.layer18 = Inception(640, (3,5), (2,2), (256,128), (160,64,None,None), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer19 = Inception(1024, (3,), (1,), (384,), (96,96,256), nn.LPPool2d(2, (3,3), stride=(3,3)), True)
        self.layer21 = Inception(736, (3,), (1,), (384,), (96,96,256), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer22 = nn.AvgPool2d((3,3), stride=(1,1), padding=(0,0))
        self.layer25 = Linear(736, 128)

        self.resize1 = nn.UpsamplingNearest2d(scale_factor=3)
        self.resize2 = nn.AvgPool2d(4)

    def forward(self, x):
        if x.size()[-1] == 128:
            x = self.resize2(self.resize1(x))

        x = self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))))
        x = self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(x)))))
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = x.view((-1, 736))

        x_736 = x

        x = self.layer25(x)
        x_norm = torch.sqrt(torch.sum(x**2, 1) + 1e-6)
        x = torch.div(x, x_norm.view(-1, 1).expand_as(x))

        #return (x, x_736)
        return x

class openFaceClassifier(nn.Module):
    def __init__(self, args):
        super(openFaceClassifier, self).__init__()
        self.use_cuda = not args.no_cuda
        self.embedding_fn = embedding_fn = openFace(args)
        for param in embedding_fn.parameters():
            param.requires_grad = False
        embedding_fn.load_state_dict(torch.load(args.load_openface))
        self.fc = nn.Linear(128, args.n_classes)

    def forward(self, x):
        x = self.embedding_fn(x)
        x = self.fc(torch.squeeze(x))
        return x

    def train_model(self, args, tr_data, val_data, fh):
        '''
        Train the model according to the parameters specified in args.
        '''
        self.train()
        lr = args.lr
        to_optimize = filter(lambda p: p.requires_grad, self.parameters())
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(to_optimize, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
            log(fh, "\tOptimizing with SGD with learning rate %.3f" % lr)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(to_optimize, lr=lr)
            log(fh, "\tOptimizing with adam with learning rate %.3f" % lr)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(to_optimize, lr=lr,
                    weight_decay=args.weight_decay)
            log(fh, "\tOptimizing with adagrad with learning rate %.3f" % lr)
        else:
            raise NotImplementedError
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=.5, \
            patience=2, threshold=1e-1)

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
            predictions.append(outs.data.max(1)[1].cpu().numpy())
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
