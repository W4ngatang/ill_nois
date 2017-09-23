import os
import pdb
import sys
import h5py
import torch
import argparse
import numpy as np
from scipy.misc import imsave

# Helper stuff
from src.codebase_pytorch.utils.dataset import Dataset
from src.codebase.utils.utils import log as log
from src.codebase_pytorch.utils.hooks import print_outputs, print_grads

# Classifiers
from src.codebase_pytorch.models.ModularCNN import ModularCNN
from src.codebase_pytorch.models.mnistCNN import MNISTCNN
from src.codebase_pytorch.models.squeezeNet import SqueezeNet
from src.codebase_pytorch.models.openFace import openFaceClassifier
from src.codebase_pytorch.models.resnet import ResNet, Bottleneck, resnet152
from src.codebase_pytorch.models.inception import Inception3
from src.codebase_pytorch.models.densenet import DenseNet, densenet161
from src.codebase_pytorch.models.vgg import vgg19_bn
from src.codebase_pytorch.models.alexnet import AlexNet, alexnet

# Generators
from src.codebase_pytorch.generators.random import RandomNoiseGenerator
from src.codebase_pytorch.generators.fgsm import FGSMGenerator
from src.codebase_pytorch.generators.carlini_l2 import CarliniL2Generator
from src.codebase_pytorch.generators.ensembler import EnsembleGenerator

def main(arguments):
    '''
    Main logic
    '''

    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    # General options
    parser.add_argument("--no_cuda", help="disables CUDA training", type=int, default=0)
    parser.add_argument("--log_file", help="Path to file to log progress", type=str)
    parser.add_argument("--data_path", help="Path to hdf5 files containing training data", type=str, default='')
    parser.add_argument("--im_file", help="Path to h5py? file containing images to obfuscate", type=str, default='')
    parser.add_argument("--out_file", help="Optional hdf5 filepath to write obfuscated images to", type=str)
    parser.add_argument("--out_path", help="Optional path to folder to save images to", type=str)

    # Model options
    parser.add_argument("--model", help="Model architecture to use", type=str, default='modular')
    parser.add_argument("--n_kerns", help="Number of convolutional filters", type=int, default=64)
    parser.add_argument("--kern_size", help="Kernel size", type=int, default=3)
    parser.add_argument("--init_scale", help="Initialization scale (std around 0)", type=float, default=.1)
    parser.add_argument("--init_dist", help="Initialization distribution", type=str, default='normal')
    parser.add_argument("--load_model_from", help="Path to load model from. When loading a model, this argument must match the import model type.", type=str, default='')
    parser.add_argument("--save_model_to", help="Path to save model to", type=str, default='')
    parser.add_argument("--load_openface", help="Path to load pretrained openFace model from.", type=str, default='src/codebase_pytorch/models/openFace.ckpt')

    # Training options
    parser.add_argument("--train", help="1 if should train model", type=int, default=1)
    parser.add_argument("--n_epochs", help="Number of epochs to train for", type=int, default=5)
    parser.add_argument("--optimizer", help="Optimization algorithm to use", type=str, default='adam')
    parser.add_argument("--lr_scheduler", help="Type of learning rate scheduler to use", type=str, default='plateau')
    parser.add_argument("--batch_size", help="Batch size", type=int, default=200)
    parser.add_argument("--lr", help="Learning rate", type=float, default=.1)
    parser.add_argument("--momentum", help="Momentum", type=float, default=.5)
    parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)
    parser.add_argument("--nesterov", help="Momentum", type=bool, default='false')

    # ModularCNN options
    parser.add_argument("--n_modules", help="Number of convolutional modules to stack (shapes must match)", type=int, default=6)

    # Generator options
    parser.add_argument("--generate", help="1 if should build generator and obfuscate images", type=int, default=1)
    parser.add_argument("--generator", help="Type of noise generator to use", type=str, default='fast_gradient')
    parser.add_argument("--target", help="Method for selecting class generator should 'push' image towards.", type=str, default='none')

    # Generator training options
    parser.add_argument("--generator_optimizer", help="Optimizer to use for Carlini generator", type=str, default='adam')
    parser.add_argument("--generator_batch_size", help="Batch size for generator", type=int, default=50)
    parser.add_argument("--generator_lr", help="Learning rate for generator optimization when necessary", type=float, default=.1)
    parser.add_argument("--n_generator_steps", help="Number of iterations to run generator for", type=int, default=1)

    # Fast gradient and related methods options
    parser.add_argument("--eps", help="Magnitude of the noise (NB: if normalized, this will be in terms of std, not pixel values)", type=float, default=.1)
    parser.add_argument("--alpha", help="Magnitude of random initialization for noise, 0 for none", type=float, default=.0)

    # Carlini and Wagner options
    parser.add_argument("--generator_init_opt_const", help="Optimization constant for Carlini generator", type=float, default=.1)
    parser.add_argument("--generator_confidence", help="Confidence in obfuscated image for Carlini generator", type=float, default=0.)
    parser.add_argument("--early_abort", help="1 if should abort if not making progress", type=int, default=0)
    parser.add_argument("--n_binary_search_steps", help="Number of steps in binary search for optimal c", type=int, default=5)

    parser.add_argument("--n_mwu_steps", help="Number of steps for multiplicative weight update", type=int, default=5)
    parser.add_argument("--mwu_eta", help="Penalization constant for MWU", type=float, default=.1)

    args = parser.parse_args(arguments)

    log_fh = open(args.log_file, 'w')
    if args.data_path[-1] != '/':
        args.data_path += '/'
    with h5py.File(args.data_path+'params.hdf5', 'r') as fh:
        args.n_classes = int(fh['n_classes'][0])
        args.im_size = int(fh['im_size'][0])
        args.n_channels = int(fh['n_channels'][0])
        if 'mean' in fh.keys():
            mean, std = fh['mean'][:], fh['std'][:]
        else:
            mean, std = None, None
    log(log_fh, "Processing %d types of images of size %d and %d channels" % (args.n_classes, args.im_size, args.n_channels))

    # TODO log more parameters
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        log(log_fh, "Using CUDA")

    # Build the model
    log(log_fh, "Building model...")
    if args.model == 'modular':
        model = ModularCNN(args)
        log(log_fh, "\tBuilt modular CNN with %d modules" % (args.n_modules))
    elif args.model == 'mnist':
        model = MNISTCNN(args)
        log(log_fh, "\tBuilt MNIST CNN")
    elif args.model == 'squeeze':
        model = SqueezeNet(num_classes=args.n_classes, use_cuda=args.cuda)
        log(log_fh, "\tBuilt SqueezeNet")
    elif args.model == 'openface':
        model = OpenFaceClassifier(args)
        log(log_fh, "\tBuilt OpenFaceClassifier")
    elif args.model == 'resnet':
        model = ResNet(Bottleneck, [3, 8, 36, 3])
        log(log_fh, "\tBuilt ResNet152")
    elif args.model == 'inception':
        model = Inception3(num_classes=args.n_classes)
        log(log_fh, "\tBuilt Inception")
    elif args.model == 'densenet':
        model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
        log(log_fh, "\tBuilt DenseNet161")
    elif args.model == 'alexnet':
        model = AlexNet()
        log(log_fh, "\tBuilt AlexNet")
    elif args.model == 'vgg':
        model = vgg19_bn(pretrained=True)
        log(log_fh, "\tBuilt VGG19bn")

    else:
        raise NotImplementedError
    if args.cuda:
        model.cuda()
    model.eval() # this should be redundant now
    log(log_fh, "Done!")

    # Optional load model
    if args.load_model_from:
        model.load_state_dict(torch.load(args.load_model_from))
        log(log_fh, "Loaded weights from %s" % args.load_model_from)

    # Train
    with h5py.File(args.data_path + 'val.hdf5', 'r') as fh:
        val_data = Dataset(fh['ins'][:], fh['outs'][:], args.batch_size, args)
    if args.train:
        log(log_fh, "Training...")
        with h5py.File(args.data_path + 'tr.hdf5', 'r') as fh:
            tr_data = Dataset(fh['ins'][:], fh['outs'][:], args.batch_size, args)
        model.train_model(args, tr_data, val_data, log_fh)
        log(log_fh, "Done!")
        del tr_data
    _, val_acc, val_top5 = model.evaluate(val_data)
    log(log_fh, "\tBest top1 validation accuracy: %.2f \tBest top5 acc: %.2f"
        % (val_acc, val_top5))
    del val_data

    if args.generate:
        assert args.im_file

        # Load images to obfuscate
        log(log_fh, "Generating noise for images...")
        with h5py.File(args.im_file, 'r') as fh:
            clean_ims = fh['ins'][:]
            te_data = Dataset(fh['ins'][:], fh['outs'][:], args.batch_size, args)
        log(log_fh, "\tLoaded %d images!" % clean_ims.shape[0])

        # Choose a class to target
        if args.target == 'random':
            data = Dataset(clean_ims, 
                np.random.randint(args.n_classes, size=te_data.n_ins), args.generator_batch_size, args)
            log(log_fh, "\t\ttargeting random class")
        elif args.target == 'least':
            preds = model.predict(te_data)
            targs = np.argmin(preds, axis=1)
            data = Dataset(clean_ims, targs, args.generator_batch_size, args)
            log(log_fh, "\t\ttargeting least likely class")
            target_s = 'least likely'
        elif args.target == 'next':
            preds = model.predict(te_data)
            one_hot = np.zeros((te_data.n_ins, args.n_classes))
            one_hot[np.arange(te_data.n_ins), te_data.outs.numpy().astype(int)] = 1
            targs = np.argmax(preds * (1. - one_hot), axis=1)
            data = Dataset(clean_ims, targs, args.generator_batch_size, args)
            log(log_fh, "\t\ttargeting next likely class")
        elif args.target == 'none':
            data = Dataset(clean_ims, te_data.outs.numpy().copy(), args.generator_batch_size, args)
            log(log_fh, "\t\ttargeting no class")
        else:
            raise NotImplementedError

        # Create the noise generator
        if args.generator == 'random':
            generator = RandomNoiseGenerator(args)
            log(log_fh, "\tBuilt random generator with eps %.3f" % args.eps)
        elif args.generator == 'carlini_l2':
            generator = CarliniL2Generator(args, (mean, std))
            log(log_fh, "\tBuilt C&W generator")
        elif args.generator == 'fgsm':
            generator = FGSMGenerator(args)
            log(log_fh, "\tBuilt fgsm generator with eps %.3f" % args.eps)
        elif args.generator == 'ensemble':
            # build a shit ton of models
            old_model = model
            resnet = resnet152(pretrained=True)
            densenet = densenet161(pretrained=True)
            alex = alexnet(pretrained=True)
            vgg = vgg19_bn(pretrained=True)
            if args.cuda:
                resnet = resnet.cuda()
                densenet = densenet.cuda()
                alex = alex.cuda()
                vgg = vgg.cuda()
            model = [resnet, densenet, alex, vgg]

            # build the generator
            generator = EnsembleGenerator(args, model, data, te_data.outs, log_fh, (mean, std))
            log(log_fh, ("\tBuilt ensemble optimization generator with "
                            "ResNet, Densenet, AlexNet, and VGG"))
        else:
            raise NotImplementedError


        # Generate the corrupt images
        # NB: noise will be in the input space,
        #     which is not necessarily a valid image
        corrupt_ims = generator.generate(data, model, args, log_fh)
        log(log_fh, "Done!")

        if args.generator == 'ensemble':
            del model
            model = old_model

        # Compute the corruption rate
        log(log_fh, "Computing corruption rate...")
        corrupt_data = Dataset(corrupt_ims, te_data.outs, args.generator_batch_size, args)
        target_data = Dataset(corrupt_ims, data.outs, args.generator_batch_size, args)
        _, clean_top1, clean_top5 = model.evaluate(te_data)
        _, corrupt_top1, corrupt_top5 = model.evaluate(corrupt_data)
        _, target_top1, target_top5 = model.evaluate(target_data)
        log(log_fh, "\tClean top 1 accuracy: %.3f \ttop 5 accuracy: %.3f" %
                (clean_top1, clean_top5))
        log(log_fh, "\tCorrupt top 1 accuracy: %.3f \ttop 5 accuracy: %.3f" %
                (corrupt_top1, corrupt_top5))
        log(log_fh, "\tTarget top 1 accuracy: %.3f \ttop 5 accuracy: %.3f" %
                (target_top1, target_top5))

        # De-normalize to [0,1]^d
        if mean is not None:
            # dumb imagenet stuff
            mean = np.array([.485, .456, .406])[..., np.newaxis, np.newaxis]
            std = np.array([.229, .224, .225])[..., np.newaxis, np.newaxis]
            for i in xrange(clean_ims.shape[0]):
                clean_ims[i] = (clean_ims[i] * std) + mean
                corrupt_ims[i] = (corrupt_ims[i] * std) + mean

        # TODO handle out of range pixels?
        '''
        for i in xrange(clean_ims.shape[0]):
            if clean_ims[i].min() < 0:
                clean_ims[i] = clean_ims[i] - clean_ims[i].min()
            if clean_ims[i].max() > 1:
                clean_ims[i] = clean_ims[i] / clean_ims[i].max()
            if corrupt_ims[i].min() < 0:
                corrupt_ims[i] = corrupt_ims[i] - corrupt_ims[i].min()
            if corrupt_ims[i].max() > 1:
                corrupt_ims[i] = corrupt_ims[i] / corrupt_ims[i].max()
        '''

        noise = corrupt_ims - clean_ims

        # Save noise and images in ~[0,1] scale
        if args.out_file:
            with h5py.File(args.out_file, 'w') as fh:
                fh['noise'] = noise
                fh['ims'] = clean_ims
                fh['noisy_ims'] = corrupt_ims
            log(log_fh, "Saved image and noise data to %s" % args.out_file)

        if args.out_path:
            clean_ims = (clean_ims * 255.).astype(np.uint8)
            corrupt_ims = (corrupt_ims * 255.).astype(np.uint8)

            for i, (clean, corrupt) in enumerate(zip(clean_ims, corrupt_ims)):
                imsave("%s/%03d_clean.png" % (args.out_path, i), np.squeeze(clean))
                imsave("%s/%03d_corrupt.png" % (args.out_path, i),
                    np.squeeze(corrupt))
            log(log_fh, "Saved images to %s" % args.out_path)

    log_fh.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
