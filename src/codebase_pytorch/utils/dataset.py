import pdb
import math
import torch

class Dataset:
    '''
    Data loader class.

    TODO
        - shuffle methods
        - make a generator thing
    '''

    def __init__(self, ins, outs, batch_size, args):
        '''
         
        '''
        self.batch_size = batch_size
        if not isinstance(ins, torch.FloatTensor):
            # TODO assert numpy
            assert len(ins.shape) == 4
            assert ins.shape[1] == args.n_channels
            assert ins.shape[2] == args.im_size and ins.shape[3] == args.im_size
            self.n_ins = ins.shape[0]
            self.n_batches = int(math.ceil(1.0 * self.n_ins / self.batch_size))
            ins = torch.FloatTensor(ins)
        else:
            self.n_ins = int(ins.size()[0])
            self.n_batches = int(math.ceil(1.0 * self.n_ins / self.batch_size))
        self.ins = ins

        if not isinstance(outs, torch.LongTensor):
            outs = torch.LongTensor(outs.astype(int))
        self.outs = outs

    def __getitem__(self, idx):
        batch_size = self.batch_size
        return (self.ins[idx * batch_size: (idx + 1) * batch_size],
                self.outs[idx * batch_size:(idx + 1) * batch_size])
