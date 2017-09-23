import os
import sys
import pdb
import h5py
import argparse
import numpy as np
from scipy.misc import imread, imresize, imsave

def load_images(path, max_class_ims):
    '''
    Load data from raw images
    '''
    if path[-1] != '/':
        path = path + '/'

    ims = []
    n_ims = []
    #idxs = [353, 806, 631, 951, 4]
    idxs = [1, 130, 296, 847, 963]
    #idxs = [1]
    folders = os.listdir(path)
    folders.sort()
    for im_dir in folders:
        n_class_ims = 0
        for im_path in os.listdir(path+im_dir):
            if im_path[-4:] != 'JPEG':
                continue
            ims.append(load_im(path+im_dir+'/'+im_path))
            n_class_ims += 1
            if max_class_ims > 0 and n_class_ims >= max_class_ims:
                break
        n_ims.append(n_class_ims)
    assert sum(n_ims) == len(ims)

    ims = np.array(ims)
    outs = []
    for n, idx in zip(n_ims, idxs):
        outs.append(np.ones((n,)) * idx)
    outs = np.hstack(outs)

    return ims, outs

def load_im(path):
    '''
    Load an image from path
    '''
    try:
        raw_im = imread(path).astype(np.float32)
        if len(raw_im.shape) == 2:
            raw_im = np.repeat(raw_im, args.n_channels).reshape(raw_im.shape[0], raw_im.shape[1], args.n_channels)
        min_dim = min(raw_im.shape[0], raw_im.shape[1])
        im = imresize(raw_im[:min_dim, :min_dim, :args.n_channels], size=[args.im_size, args.im_size, args.n_channels]) # or just resize the whole thing
        return np.array(im, dtype=np.float32) / 255.
    except IOError as e:
        print "\tUnable to open %s" % path

def create_dataset(ins, outs, split):
    with h5py.File(args.out_path + split + ".hdf5", 'w') as f:
        f['ins'] = ins
        f['outs'] = outs

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_path', help='path to raw data', type=str, default='')
    parser.add_argument('--load_data_from', help='optional path to hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--save_data_to', help='optional path to save hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--out_path', help='path to folder to contain output files', type=str, default='')

    parser.add_argument('--n_classes', help='number of classes to use', type=int, default=1000)
    parser.add_argument('--n_class_ims', help='number of images per class to use', type=int, default=200)
    parser.add_argument('--im_size', help='size of image along a side (assuming square images)', type=int, default=224)
    parser.add_argument('--n_channels', help='number of image channels', type=int, default=3)
    parser.add_argument('--normalize', help='1 if normalize', type=int, default=1)
    args = parser.parse_args(arguments)
    
    print 'Loading data...'
    if args.load_data_from:
        print '\tReading data from %s' % args.load_data_from
        with h5py.File(args.load_data_from, 'r') as f:
            ins = f['ins'][:]
            outs = f['outs'][:]
            args.n_classes = f['n_classes'][:]
    else:
        print '\tReading data from images'
        ins, outs = load_images(args.data_path, args.n_class_ims)

        if args.save_data_to:
            with h5py.File(args.save_data_to, 'w') as f:
                f['ins'] = ins
                f['outs'] = outs
                f['n_classes'] = np.array([args.n_classes])
            print '\tSaved loaded data to %s' % args.save_data_to
        else:
            print '\tAre you sure you don\'t want to save images to HDF5?'

    ins = ins.transpose((0,3,1,2))
    n_ims, n_classes = ins.shape[0], args.n_classes
    print '\tLoaded %d images for %d classes' % (n_ims, n_classes)

    if args.normalize:
        #mean = np.expand_dims(np.expand_dims(np.expand_dims(np.array([.485, .456, .406]), axis=0), axis=-1), axis=-1)
        #std = np.expand_dims(np.expand_dims(np.expand_dims(np.array([.229, .224, .225]), axis=0), axis=-1,), axis=-1)
        #mean = np.mean(ins, axis=(0,2,3), keepdims=1)
        #std = np.std(ins, axis=(0,2,3), keepdims=1)
        #ins = (ins - mean) / std

        mean = np.array([.485, .456, .406])[..., np.newaxis, np.newaxis]
        std = np.array([.229, .224, .225])[..., np.newaxis, np.newaxis]
        for i in xrange(ins.shape[0]):
            ins[i] = (ins[i] - mean) / std
        print '\tNormalized data!'
    print 'Data loaded!'

    # create episodes
    if args.out_path[-1] != '/':
        args.out_path += '/'
    print 'Creating data...'
    create_dataset(ins, outs, "te")
    print '\tFinished test data'
    print 'Done!'

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
