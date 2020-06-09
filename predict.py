import argparse
import numpy as np
from PIL import Image
import torch
import json

from flowermodel import load_checkpoint 
from flowerutils import process_image


"""
Predict flower name from an image with predict.py along with the probability of that name. 
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: `python predict.py /path/to/image checkpoint`

Options:
- Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
- Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
- Use GPU for inference: `python predict.py input checkpoint --gpu`
"""

# default value for number of images to predict
TOPK_DEFAULT = 5


def predict_one(image, model, topk=TOPK_DEFAULT):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    Params:
    - image: PIL Image instance
    - model: model instance
    - topk (int): number of classes to show
    
    Returns:
    - p: sorted (descending) list of probabilities, length topk
    - cls: sorted (descending) list of associated class indices, length topk
    '''
    inputs = process_image(image)
    inputs = inputs.reshape((1, *inputs.shape)) # make it 4d
    inputs = inputs.to(device)
    
    with torch.no_grad():
        model.eval()
        output = model.forward(inputs)
        logp, class_indices = output.topk(topk, dim=1)
        logp, class_indices = logp[0].cpu().numpy(), class_indices[0].cpu().numpy() # just one sample; 
        p = np.exp(logp)
        cls = np.array([model.idx_to_class[i] for i in class_indices])
        sorted_indices = p.argsort()[::-1]
        
    return p[sorted_indices], cls[sorted_indices]


parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
parser.add_argument('filepath', type=str, help='Path to image file')
parser.add_argument('checkpoint', type=str, help='Path to trained model')
parser.add_argument('--top_k', type=int, default=TOPK_DEFAULT, help='how many classes to predict')
parser.add_argument('--category_names', type=str, help='json file containing category names')
parser.add_argument('--gpu', action='store_true', help='use GPU for prediction')
args = parser.parse_args()


# load image
try:
    image = Image.open(args.filepath)
except Exception as e:
    print("Error loading image: %s" % e)
    quit()

# check gpu
if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
# load model
try:
    model = load_checkpoint(args.checkpoint)
    model = model.to(device)
except Exception as e:
    print("Error loading model checkpoint: %s" % e)
    quit()
    
# determine number of classes
class_to_idx = model.class_to_idx
nclasses = len(model.class_to_idx)
    
# check top_k argument. should be in 1 <= top_k <= num_classes
if (args.top_k < 1) or (args.top_k > nclasses):
    topk = TOPK_DEFAULT
else:
    topk = args.top_k
    
# load category names if available
cat_to_name = None
if args.category_names:
    try:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        assert set(cat_to_name.keys()) == set(class_to_idx.keys()), "category keys do not match the keys of the model"
    except Exception as e:
        print("Error loading category names: %s" % e)
        cat_to_name = None
        

# finally, make prediction
probas, classes = predict_one(image, model, topk=topk)

# print result
print("\nproba   index   category\n---------------------------------")
for p, c in zip(probas, classes):
    catname = cat_to_name[c] if cat_to_name else "---"
    print("{:.3f}   {:4s}    {}".format(p, c, catname))

