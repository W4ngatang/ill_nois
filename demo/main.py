import os
import io
import sys
import pdb
import string
#import h5py
#import boto3
import base64
import random
import argparse
#import torch
import numpy as np
#import tensorflow as tf
from scipy.misc import imread, imsave

from flask import Flask, render_template, request, redirect, \
        send_from_directory, flash, url_for, jsonify, send_file
from werkzeug.utils import secure_filename

#from src.codebase.models.simple_cnn import SimpleCNN
#from src.codebase.generators.fast_gradient import FastGradientGenerator

# constants
API_NAME = 'illnoise'
VERSION = 'v0.1'
UPLOAD_FOLDER = 'demo/tmp'

# web app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'test'

# globals for hacky / lazy stuff
true_class = -1

def random_string(length=10):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(length))

def encode_arr(arr):
    fh = io.BytesIO()
    imsave(fh, np.squeeze(arr), format='png')
    base64_im = base64.b64encode(fh.getvalue())
    return str(base64_im)[2:-1] # gets rid of b'...'

@app.route('/')
def index():
    return render_template('index.html')

###################
### API METHODS ###
###################
# Long term, probably not sustainable to send the entire image over the server

@app.route('/%s/api/%s/obfuscate' % (API_NAME, VERSION), methods=['POST'])
def obfuscate():
    '''
    im = np.array(request.json).reshape((1,CIFAR_IM_DIM, CIFAR_IM_DIM,3)) / 255.
    true_class = np.argmax(model.predict(im)[0])
    noise = generator.generate((im, np.array([true_class])), model, args)
    preds = model.predict(im+noise)
    enc_noise = encode_arr(noise / (2*generator.eps) + .5)
    enc_im = encode_arr(255.*(im+noise))
    return jsonify(preds=preds[0].tolist(),
        noise_src='data:image/png;base64,'+enc_noise,
        obf_src='data:image/png;base64,'+enc_im)
    '''
    height, width = request.json[0], request.json[1]
    im = np.array(request.json[2:]).reshape((1, width, height, 3))
    enc_im = encode_arr(np.random.randint(0, 2, im.shape))
    return jsonify(obf_src='data:image/png;base64,'+enc_im)

@app.route('/%s/api/%s/predict' % (API_NAME, VERSION), methods=['POST'])
def predict():
    im = np.array(request.json).reshape((1,CIFAR_IM_DIM, CIFAR_IM_DIM,3)) / 255.
    preds = model.predict(im)
    return jsonify(preds=preds[0].tolist())

@app.route('/%s/api/%s/identify' % (API_NAME, VERSION), methods=['POST'])
def identify():
    im = np.array(request.json).reshape((128,128,3))
    im_name = '%s/%s.png' % (UPLOAD_FOLDER, random_string())
    imsave(im_name, im)
    with open(im_name) as fh:
        im = fh.read()
        im_bytes = bytearray(im)
    resp = client.recognize_celebrities(Image={'Bytes':im_bytes})
    celebs, confidences = [], [] # how to handle unrecognized faces
    for celeb in resp['CelebrityFaces']:
        celebs.append(celeb['Name'])
        confidences.append(celeb['MatchConfidence'])
    if not celebs:
        celebs.append("No faces recognized")
        confidences.append("NA")
    return jsonify(celebs=celebs, confidences=confidences)

@app.route('/%s/api/%s/celebfuscate' % (API_NAME, VERSION), methods=['POST'])
def celebfuscate():
    im = np.array(request.json['image']).reshape((1,128,128,3)) / 255.
    targ = int(request.json['target'])
    print('Received target %d' % targ)
    if targ < 0:
        preds = model.predict(im)
        targ = np.argmin(preds)
    print('Targeting class %d' % targ)
    noise = generator.generate((im, np.array([targ])), model, args)
    enc_noise = encode_arr(noise / (2*generator.eps) + .5)
    enc_im = encode_arr(im+noise)

    im_name = '%s/%s.png' % (UPLOAD_FOLDER, random_string())
    imsave(im_name, (noise+im)[0])
    with open(im_name) as fh:
        im = fh.read()
        im_bytes = bytearray(im)
    resp = client.recognize_celebrities(Image={'Bytes':im_bytes})
    celebs, confidences = [], [] # how to handle unrecognized faces
    for celeb in resp['CelebrityFaces']:
        celebs.append(celeb['Name'])
        confidences.append(celeb['MatchConfidence'])
    if not celebs:
        celebs.append("No faces recognized")
        confidences.append("NA")
    return jsonify(noise_src='data:image/png;base64,'+enc_noise,
        obf_src='data:image/png;base64,'+enc_im,
        celebs=celebs, confidences=confidences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    # General options
    parser.add_argument("--run_local", help="1 if run locally", type=int, default=1)
    parser.add_argument("--data_path", help="Path to hdf5 files containing training data", type=str, default='')

    # Model options
    parser.add_argument("--model", help="Model architecture to use", type=str, default='simple')
    parser.add_argument("--n_kerns", help="Number of convolutional filters", type=int, default=64)
    parser.add_argument("--kern_size", help="Kernel size", type=int, default=3)
    parser.add_argument("--init_scale", help="Initialization scale (std around 0)", type=float, default=.1)
    parser.add_argument("--load_model_from", help="Path to load model from. \
                                                    When loading a model, \
                                                    this argument must match \
                                                    the import model type.",
                                                    type=str, default='')

    # Training options
    parser.add_argument("--n_epochs", help="Number of epochs to train for", type=int, default=5)
    parser.add_argument("--optimizer", help="Optimization algorithm to use", type=str, default='adam')
    parser.add_argument("--batch_size", help="Batch size", type=int, default=200)
    parser.add_argument("--learning_rate", help="Learning rate", type=float, default=1.)

    # SimpleCNN options
    parser.add_argument("--n_modules", help="Number of convolutional modules to stack (shapes must match)", type=int, default=6)

    # Generator options
    parser.add_argument("--generate", help="1 if should build generator and obfuscate images", type=int, default=1)
    parser.add_argument("--generator", help="Type of noise generator to use", type=str, default='fast_gradient')
    parser.add_argument("--generator_optimizer", help="Optimizer to use for Carlini generator", type=str, default='adam')
    parser.add_argument("--target", help="Default way to select class to target", type=str, default='least')
    parser.add_argument("--eps", help="Magnitude of the noise", type=float, default=1.)
    parser.add_argument("--alpha", help="Magnitude of random initialization for noise, 0 for none", type=float, default=.0)
    parser.add_argument("--n_generator_steps", help="Number of iterations to run generator for", type=int, default=1)
    parser.add_argument("--generator_opt_const", help="Optimization constant for Carlini generator", type=float, default=.1)
    parser.add_argument("--generator_confidence", help="Confidence in obfuscated image for Carlini generator", type=float, default=0)
    parser.add_argument("--generator_learning_rate", help="Learning rate for generator optimization when necessary", type=float, default=.1)

    args = parser.parse_args(sys.argv[1:])

    #client = boto3.client(service_name='rekognition', region_name='us-east-1')

    '''
    if args.data_path[-1] != '/':
        args.data_path += '/'
    with h5py.File(args.data_path+'params.hdf5', 'r') as f:
        args.n_classes = f['n_classes'][0]
        args.im_size = f['im_size'][0]
        args.n_channels = f['n_channels'][0]

    generator = FastGradientGenerator(args)
    model = SimpleCNN(args)
    model.load_weights(args.load_model_from)
    print('Loaded model from %s' % args.load_model_from)
    '''

    if args.run_local:
        app.run()
    else:
        app.run('0.0.0.0', debug=True, port=80)
