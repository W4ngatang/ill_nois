import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable

from src.codebase.utils.utils import log
from src.codebase_pytorch.models.model import Model
from src.codebase_pytorch.utils.scheduler import ReduceLROnPlateau

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}

model_paths = {
    'squeezenet1_0': '/n/regal/rush_lab/awang/models/squeezenet_v1_0.ckpt',
    'squeezenet1_1': '/n/regal/rush_lab/awang/models/squeezenet_v1_1.ckpt'
}

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(Model):

    def __init__(self, version=1.0, num_classes=1000, use_cuda=0):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        self.use_cuda = use_cuda

        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

    def train_model(self, args, tr_data, val_data, fh):
        '''
        Train the model according to the parameters specified in args.
        '''
        self.train()
        lr = args.lr
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
            log(fh, "\tOptimizing with SGD with learning rate %.3f" % lr)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
            log(fh, "\tOptimizing with adam with learning rate %.3f" % lr)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=lr,
                    weight_decay=args.weight_decay)
            log(fh, "\tOptimizing with adagrad with learning rate %.3f" % lr)
        else:
            raise NotImplementedError
        scheduler = ReduceLROnPlateau(optimizer, 
                'max', factor=.5, patience=3, threshold=1e-1)
        #lr_lambda = lambda epoch: (1 - float(epoch) / args.n_epochs)
        #scheduler = LambdaLR(optimizer, lr_lambda)

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
            if abs(last_acc - val_acc) < .01:
                #pdb.set_trace()
                #self.final_conv.register_forward_hook(print_outputs)
                #self.final_conv.register_backward_hook(print_grads)
                pass
            last_acc = val_acc
            scheduler.step(epoch)

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

def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
        model.load_state_dict(torch.load(pretrained))
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
        model.load_state_dict(torch.load(pretrained))
    return model
