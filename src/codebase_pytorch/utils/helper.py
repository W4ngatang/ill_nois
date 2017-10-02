

def writeCSV(lst, filename, args):
    csv_file = open(filename, "w+")
    csv_file.write("Noise Target Type, {}\n".format(args.target))
    csv_file.write("MWU Iterations, {}\n".format(args.n_mwu_steps))
    csv_file.write("Epsilon, {}\n".format(args.eps))
    csv_file.write("MWU Penalty, {}\n".format(args.mwu_penalty))
    for line in lst:
        str_line = ",".join([str(val) for val in line]) + "\n"
        csv_file.write(str_line)
    csv_file.close()



class UnNormalize(object):
    """ Undo normalization of an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will unnormalize each channel of the torch.*Tensor, i.e.
    channel = channel * std + mean

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to remove normalization.

        Returns:
            Tensor: (Un)Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
