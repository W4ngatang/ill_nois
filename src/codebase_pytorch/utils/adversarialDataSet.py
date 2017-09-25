import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

default_transform = transforms.Compose([
   transforms.Scale((256, 256)),
   transforms.ToTensor()
])

class AdversarialImageDataSet(Dataset):

    def __init__(self, img_dir, labelCSVfile, transform=default_transform):
        self.img_dir = img_dir
        self.imgInfo = pd.read_csv(labelCSVfile)
        self.transform = transform
        self.data = []
        for index in xrange(100):
            name, true_label, target_label = self.imgInfo.iloc[index][
                ['image name', 'true label id', 'target label id']]
            img = Image.open(self.img_dir + name).convert('RGB')
            self.data.append((self.transform(img), true_label, target_label))

    def __getitem__(self, index):
        """
        returns:

        img = pytorch tensor of image with transforms applied
        true label = int corresponding to ground truth
        target label = int corresponding to target class
        """
        return self.data[index]

    def __len__(self):
        return self.imgInfo.shape[0]

    def get_info(self, index):
        """
        returns Pandas DataFrame row with information about the test sample
        """
        return self.imgInfo.iloc[index]

if __name__ == "__main__":
    import sys
    dataset = AdversarialImageDataSet("/Users/juanperdomo/juanky/test_data/", "/Users/juanperdomo/juanky/image_label_target.csv")
    print dataset[0][1:]
    print float(sys.getsizeof(dataset)) # fairly small data set should be easy to deal with
