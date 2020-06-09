import argparse
import os
import datetime
import numpy as np
from PIL import Image
import torch

from flowerutils import DataSet
from flowermodel import architectures, build_model, train_model, test_model, save_checkpoint

"""
Train a new network on a data set with train.py
Basic usage: `python train.py data_directory`
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
- Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
- Choose architecture: `python train.py data_dir --arch "vgg13"`
- Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
- Use GPU for training: `python train.py data_dir --gpu`
"""

parser = argparse.ArgumentParser(description='Train a new image classifier.')
parser.add_argument('data_directory', type=str, help='path to image data directory')
parser.add_argument('--save_dir', type=str, default=".", help='path to directory where trained model checkpoints should be saved')
parser.add_argument('--arch', type=str, default='vgg16', help='basic network architecture, one of: %s. Default vgg16' % list(architectures.keys()))
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate (default 0.001)')
parser.add_argument('--hidden_units', type=int, default=512, help='number of hidden units in the classifier (default 512)')
parser.add_argument('--epochs', type=int, default=4, help='number of epochs for training (default 4)')
parser.add_argument('--gpu', action='store_true', help='use GPU for training (no if argument is not set)')

args = parser.parse_args()

# create dataloaders
try:
    dataset = DataSet(args.data_directory)
    # number of classes
    n_classes = dataset.n_classes()
    print("number of classes:", n_classes)
    # show sizes of datasets
    print("dataset sizes:", dataset.sizes())
except Exception as e:
    print("Error loading image data: %s" % e)
    quit()
   
# does save_dir exist?
if not os.path.exists(args.save_dir):
    print("save_dir does not exist: %s" % args.save_dir)
    quit()
    
# is arch supprted?
if args.arch not in architectures.keys():
    print("Artitecture '%s' not supported. See help." % args.arch)
    quit()
    
# gpu?
if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Using device: %s" % device)

# build model
model = build_model(args.arch, n_classes, n_hidden=args.hidden_units, pretrained=True)

# train it
classifier_name = architectures[args.arch][0]
classifier = getattr(model, classifier_name)
optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
model.to(device)

print("\n*** Start training ***\n")
train_model(model, device, optimizer, num_epochs=args.epochs, 
            train_dataloader=dataset.dataloaders['train'], 
            valid_dataloader=dataset.dataloaders['valid'])

# test it
print("\n*** Start testing ***\n")
test_loss, test_acc = test_model(model, device, dataset.dataloaders['test'], prefix="Test")
print("\nTest accuracy: %.4f" % test_acc)

# save it - model name is a concatenation of architecture and timestamp
now = datetime.datetime.now() 
ts = int(datetime.datetime.timestamp(now))
description = "%s, %i hidden units, trained on %s for %i epochs with Adam and learning rate %.4f, test accuracy %.4f" % (
                    args.arch, args.hidden_units, now, args.epochs, args.learning_rate, test_acc)
model_path = os.path.join(args.save_dir, "%s_%i.pth" % (args.arch, ts))
print("\nSaving model to: %s" % model_path)
save_checkpoint(model, model_path, args.arch, args.hidden_units, dataset.class_to_idx(), 
                optimizer, args.epochs, args.learning_rate, description=description)
