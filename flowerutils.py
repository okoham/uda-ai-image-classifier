from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms


# define constants
resize = 255
rot = 30
imsize = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
batch_size = 32

class DataSet(object):
    
    def __init__(self, data_dir):
        """
        Create data transformations and return dataloaders.

        Params:
        - data_dict (str): top data directory.

        """
        # Set data directories    
        self.data_dir = data_dir
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        # same transformation for validation and test
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomRotation(rot),
                transforms.RandomResizedCrop(imsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
            'valid_test': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
        }

        self.image_datasets = {
            'train': datasets.ImageFolder(train_dir, transform=self.data_transforms['train']),
            'valid': datasets.ImageFolder(valid_dir, transform=self.data_transforms['valid_test']),
            'test': datasets.ImageFolder(test_dir, transform=self.data_transforms['valid_test']),
        }

        self.dataloaders = {
            'train': torch.utils.data.DataLoader(self.image_datasets['train'], batch_size=batch_size, shuffle=True),
            'valid': torch.utils.data.DataLoader(self.image_datasets['valid'], batch_size=batch_size),
            'test': torch.utils.data.DataLoader(self.image_datasets['test'], batch_size=batch_size),
        }
        

    def class_to_idx(self):
        return self.image_datasets['train'].class_to_idx


    def n_classes(self):
        return len(self.class_to_idx())


    def sizes(self):
       return {x: len(self.image_datasets[x]) for x in ['train', 'valid', 'test']}

 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a 3D torch tensor.
        
    Params:
    - image: PIL.Image instance
    
    Returns:
    - torch.tensor, shape (3, W, H)
    '''    
    # resize so that the shortest side is 256 pixels, keeping the aspect ratio
    w, h = image.width, image.height
    scale = 256/min(w, h)
    ws, hs = int(scale*w), int(scale*h)
    image = image.resize((ws, hs))
    
    # crop center 224x224 portion
    cx, cy = ws//2, hs//2
    image = image.crop((cx-imsize//2, cy-imsize//2, cx+imsize//2, cy+imsize//2))
    
    np_image = np.array(image, dtype=np.float32)     # make an array
    np_image /= resize                               # scale to 0-1        
    np_image -= mean                                 # normalize
    np_image /= std
    np_image = np_image.transpose((2, 0, 1))         # transpose

    # note: I return a torch tensor here - numpy array not needed.
    return torch.from_numpy(np_image)


if __name__ == '__main__':
    pass