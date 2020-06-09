import sys
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import models


# architecture name, name of classifier layer, feature size
architectures = {
    'alexnet':      ['classifier', 9216],
    'densenet121':  ['classifer', 1024],
    'densenet161':  ['classifer', 2208],
    'densenet169':  ['classifer', 1664],
    'densenet201':  ['classifer', 1920],
    'inception_v3': ['fc', 2048],
    'resnet101':    ['fc', 2048],
    'resnet152':    ['fc', 2048],
    'resnet18':     ['fc', 512],
    'resnet34':     ['fc', 512],
    'resnet50':     ['fc', 2048],
    'vgg11':        ['classifer', 25088],
    'vgg11_bn':     ['classifer', 25088],
    'vgg13':        ['classifer', 25088],
    'vgg13_bn':     ['classifer', 25088],
    'vgg16':        ['classifer', 25088],
    'vgg16_bn':     ['classifer', 25088],
    'vgg19':        ['classifer', 25088],
    'vgg19_bn':     ['classifer', 25088],
}

# Loss function is Negative Log Likelihood loss. It'd hard coded into this project.
criterion = nn.NLLLoss()

def build_model(arch, n_classes, n_hidden=512, pretrained=True):
    """Return a (pretrained or empty) model with a custom classifier.
    
    Params: 
    - arch (string): One of ['alexnet', 'densenet121', 'densenet161', 'densenet169', 
            'densenet201', 'inception_v3', 'resnet101', 'resnet152', 'resnet18', 
            'resnet34', 'resnet50', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 
            'vgg16_bn', 'vgg19', 'vgg19_bn']
    - c_classes (int): number of classes for output
    - n_hidden (int): number of hidden units in the classifier. Default 512
    - pretrained (bool): Whether to return a model with pretrained features (for training the new classifier), 
      or an empty model (when loading a trained model and setting the weights from the checkpoint).

    Returns:
    - model
    """
    # classifier name and number of features. In some models the classifier layer 
    # is named 'classifier', in others 'fc'
    classifier_name, n_features = architectures[arch]
    # load (pretrained) model
    model = getattr(models, arch)(pretrained=pretrained)
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    # define a new trainable classifier
    classifier = nn.Sequential(OrderedDict([
        ('drop0', nn.Dropout(p=0.5)),
        ('fc1', nn.Linear(n_features, n_hidden)),  
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(n_hidden, n_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # set the classifier
    model.classifier = classifier
    setattr(model, classifier_name, classifier)    
    setattr(model, 'arch', arch)
    setattr(model, 'n_hidden', n_hidden)
    return model



def test_model(model, device, dataloader, prefix=""):
    """Evaluate the model. Can be used for validation (during training) and testing.
    
    Params: 
    - model: the model
    - dataloader: a torch.utils.data.DataLoader instance
    - prefix: What to display on screen during evaluation (usually "Validation" or "Test"
    
    Returns:
    - loss
    - accuracy
    """

    model.eval()   # Set model to evaluate mode

    dataset_size = len(dataloader.dataset)
    count = 0
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device) # shape (batchsize, 3, width, height)
        labels = labels.to(device) # 1-d array, shape (batchsize,)
        # forward
        with torch.no_grad():
            outputs = model(inputs) # shape: 
            proba = torch.exp(outputs) # shape: 
            preds = torch.argmax(proba, dim=1) # 1-d array, shape (batchsize,)
            loss = criterion(outputs, labels) 
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        count += inputs.size(0)
        # display progress
        sys.stdout.write("\r%s: %i/%i - loss: %.3f, acc: %.3f" % (prefix, count, 
                                                 dataset_size, 
                                                 running_loss/count, 
                                                 running_corrects.double()/count))
    sys.stdout.write("\n")
    
    loss = running_loss / dataset_size
    acc = running_corrects.item() / dataset_size
    return loss, acc



def train_model(model, device, optimizer, train_dataloader, valid_dataloader, num_epochs=5):
    """Train a model.
    
    Parameters:
    - model: a model instance
    - optimizer: an optimizer instance
    - train_dataloader: 
    - valid_dataloader: 
    - num_epochs: number of epochs to train (default 5)
    
    Returns:
    - None
    """
    training_size = len(train_dataloader.dataset)
    model.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        running_loss = 0.0             # running training loss
        running_corrects = 0           # running training correct samples
        count = 0                      # running number of samples
        
        # Iterate over data batches
        for inputs, labels in train_dataloader: 
            inputs = inputs.to(device)
            labels = labels.to(device) # 1-d array, shape (batchsize,)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            proba = torch.exp(outputs)
            preds = torch.argmax(proba, dim=1) # 1-d array, shape (batchsize,)
            loss = criterion(outputs, labels)
            # backward 
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            count += inputs.size(0)
            sys.stdout.write("\rTrain: %i/%i - loss: %.3f, acc: %.3f" % (count, 
                                                     training_size, 
                                                     running_loss/count, 
                                                     running_corrects.double()/count))
                
        epoch_loss = running_loss / training_size
        epoch_acc = running_corrects.double() / training_size
        print()
        # validation 
        model.eval()
        valid_loss, valid_acc = test_model(model, device, valid_dataloader, prefix="Validation")
        print('Train loss: {:.4f}, acc: {:.4f}    Valid loss: {:.4f}, acc: {:.4f}\n'.format(epoch_loss, epoch_acc, valid_loss, valid_acc))
        model.train()


    
def save_checkpoint(model, fpath, arch, n_hidden, class_to_idx, optimizer, epochs, lr, description=""):
    """Save a model alsong with some hyperparameters.
    
    Params:
    - model: the model instance to save
    - fpath (string): path where the model should be saved
    - arch (string): the model architechture (like 'vgg16', vgg16_bn', ...)
    - n_hidden (int): number of hidden units in classifier layer
    - class_to_idx (dict of str: int): distionary, maps class names (directory names) to index of output layer
    - optimizer: the optimizer instance used to train the model.
    - epochs (int): number of epochs trained so far
    - lr (float): learning rate used
    - desription (str): some descriptive information, e.g. how well it performed om test data
    
    Returns:
    - None
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "arch": model.arch,
        "n_hidden": model.n_hidden,
        "class_to_idx": class_to_idx,
        "description": description,
        "epochs": epochs,
        "lr": lr,
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, fpath)
        
        
        
def load_checkpoint(fpath):
    """Return a saved model.
    
    Params:
    - fpath (str): path to model checkpoint file
    
    Returns: 
    - model
    """
    
    checkpoint = torch.load(fpath, map_location='cpu')
    print(checkpoint["description"])
    
    arch = checkpoint['arch']
    class_to_idx = checkpoint['class_to_idx']
    # build empty model (without pretrained weights), then load the saved weights.
    model = build_model(arch, len(class_to_idx), n_hidden=checkpoint['n_hidden'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    # attach class names and inverted index
    model.class_to_idx = class_to_idx
    model.idx_to_class = {v:k for k, v in class_to_idx.items()}
    model.eval()
    
    return model
    


if __name__ == '__main__':
    pass