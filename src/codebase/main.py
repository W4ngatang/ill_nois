import os
import pdb
import sys
import h5py
import argparse
import numpy as np
from scipy.misc import imsave
from src.codebase.models.simple_cnn import SimpleCNN
from src.codebase.generators.fast_gradient import FastGradientGenerator
from src.codebase.generators.carlini_l2 import CarliniL2Generator
from src.codebase.generators.random import RandomNoiseGenerator
from src.codebase.utils.dataset import Dataset
from src.codebase.utils.utils import log as log
from src.codebase_pytorch.models.ModularCNN import ModularCNN

def main(arguments):
    '''
    Main logic
    '''

    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # General options
    parser.add_argument("--cuda", help="1 if use CUDA", type=int, default=0)
    parser.add_argument("--log_file", help="Path to file to log progress", type=str)
    parser.add_argument("--data_path", help="Path to hdf5 files containing training data", type=str, default='')
    parser.add_argument("--im_file", help="Path to h5py? file containing images to obfuscate", type=str, default='')
    parser.add_argument("--out_file", help="Optional hdf5 filepath to write obfuscated images to", type=str)
    parser.add_argument("--out_path", help="Optional path to folder to save images to", type=str)

    # Model options
    parser.add_argument("--model", help="Model architecture to use", type=str, default='simple')
    parser.add_argument("--n_kerns", help="Number of convolutional filters", type=int, default=64)
    parser.add_argument("--kern_size", help="Kernel size", type=int, default=3)
    parser.add_argument("--init_scale", help="Initialization scale (std around 0)", type=float, default=.1)
    parser.add_argument("--load_model_from", help="Path to load model from. When loading a model, \
                                                    this argument must match the import model type.", 
                                                    type=str, default='')
    parser.add_argument("--save_model_to", help="Path to save model to", type=str, default='')

    # Training options
    parser.add_argument("--n_epochs", help="Number of epochs to train for", type=int, default=5)
    parser.add_argument("--optimizer", help="Optimization algorithm to use", type=str, default='adam')
    parser.add_argument("--batch_size", help="Batch size", type=int, default=200)
    parser.add_argument("--lr", help="Learning rate", type=float, default=.1)

    # SimpleCNN options
    parser.add_argument("--n_modules", help="Number of convolutional modules to stack (shapes must match)", type=int, default=6)

    # Generator options
    parser.add_argument("--generate", help="1 if should build generator and obfuscate images", type=int, default=1)
    parser.add_argument("--generator", help="Type of noise generator to use", type=str, default='fast_gradient')
    parser.add_argument("--generator_optimizer", help="Optimizer to use for Carlini generator", type=str, default='adam')
    parser.add_argument("--target", help="Method for selecting class generator should \
                                            'push' image towards.", type=str, default='none')
    parser.add_argument("--eps", help="Magnitude of the noise", type=float, default=.1)
    parser.add_argument("--alpha", help="Magnitude of random initialization for noise, 0 for none", type=float, default=.0)
    parser.add_argument("--n_generator_steps", help="Number of iterations to run generator for", type=int, default=1)
    parser.add_argument("--generator_opt_const", help="Optimization constant for Carlini generator", type=float, default=.1)
    parser.add_argument("--generator_confidence", help="Confidence in obfuscated image for Carlini generator", type=float, default=0)
    parser.add_argument("--generator_learning_rate", help="Learning rate for generator optimization when necessary", type=float, default=.1)

    args = parser.parse_args(arguments)

    log_fh = open(args.log_file, 'w')
    if args.data_path[-1] != '/':
        args.data_path += '/'
    with h5py.File(args.data_path+'params.hdf5', 'r') as f:
        args.n_classes = f['n_classes'][0]
        args.im_size = f['im_size'][0]
        args.n_channels = f['n_channels'][0]
    log(log_fh, "Processing %d types of images of size %d and %d channels" % (args.n_classes, args.im_size, args.n_channels))
    # TODO log more parameters

    # Build the model
    log(log_fh, "Building model...")
    if args.model == 'simple':
        model = SimpleCNN(args)
    if args.model == 'modular':
        model = ModularCNN(args)
    else:
        raise NotImplementedError
    log(log_fh, "\tDone!")

    # TODO build off of default PyTorch Dataset
    log(log_fh, "Training...")
    with h5py.File(args.data_path+'tr.hdf5', 'r') as fh:
        tr_data = Dataset(fh['ins'][:], fh['outs'][:], args)
    with h5py.File(args.data_path+'val.hdf5', 'r') as fh:
        val_data = Dataset(fh['ins'][:], fh['outs'][:], args)
    model.train(tr_data, val_data, args, log_fh)
    log(log_fh, "Done!")

    if args.generate:
        assert args.im_file

        # Load images to obfuscate
        log(log_fh, "Generating noise for images...")
        with h5py.File(args.im_file, 'r') as fh:
            te_data = Dataset(fh['ins'][:], fh['outs'][:], args)
        log(log_fh, "\tLoaded images!")
        _, clean_acc_old = model.validate(te_data)

        # Generate the noise
        if args.generator == 'random':
            generator = RandomNoiseGenerator(args)
        elif args.generator == 'carlini':
            generator = CarliniL2Generator(args, model, te_data.n_ins)
        elif args.generator == 'fast_gradient':
            generator = FastGradientGenerator(args)
        else:
            raise NotImplementedError
        log(log_fh, "\tGenerator built!")
        if args.target == 'random':
            data = Dataset(te_data.ins, np.random.randint(args.n_classes, size=te_data.n_ins), args)
        elif args.target == 'least':
            preds = model.predict(te_data.ins)
            targs = np.argmin(preds, axis=1)
            data = Dataset(te_data.ins, targs, args)
        elif args.target == 'next_likely':
            preds = model.predict(te_data.ins)
            one_hot = np.zeros((te_data.n_ins, args.n_classes))
            one_hot[np.arange(te_data.n_ins), te_data.outs.astype(int)] = 1
            targs = np.argmax(preds * (1. - one_hot), axis=1)
            data = Dataset(te_data.ins, targs, args)
        elif args.target == 'none':
            data = te_data
        else:
            raise NotImplementedError
        noise = generator.generate(data, model, args, log_fh)

        # Compute the corruption rate
        log(log_fh, "\tComputing corruption rate...")
        _, clean_acc = model.validate(te_data)
        corrupt_ins = te_data.ins + noise
        # TODO handle out of range [0, 1] pixels (clip)
        #corrupt_ins = np.max(np.min(te_data.ins + noise, 1.0), 0.0)
        corrupt_data = Dataset(corrupt_ins, te_data.outs, args)    
        _, corrupt_acc = model.validate(corrupt_data)
        assert clean_acc == clean_acc_old
        log(log_fh, "\tOriginal accuracy: %.3f, new accuracy: %.3f" % 
                (clean_acc, corrupt_acc))

        # Save noise and images
        if args.out_file:
            with h5py.File(args.out_file, 'w') as fh:
                fh['noise'] = noise
                fh['ims'] = te_data.ins
                fh['noisy_ims'] = corrupt_data.ins
            log(log_fh, "Saved image and noise data to %s" % args.out_file)

        if args.out_path:
            for i, (clean_im, corrupt_im) in \
                    enumerate(zip(te_data.ins, corrupt_data.ins)):
                imsave("%s/clean_%d.png" % (args.out_path, i), np.squeeze(clean_im))
                imsave("%s/corrupt_%d.png" % (args.out_path, i), np.squeeze(corrupt_im))
            log(log_fh, "Saved images to %s" % args.out_path)

        log(log_fh, "Done!")

    log_fh.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
