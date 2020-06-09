# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Data Source

- Related publication: http://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/
- Data: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

## Review

https://review.udacity.com/#!/reviews/2154650


## Part 1: Developing an Image Classifier with Deep Learning

<p>In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch. We'll provide some tips and guide you, but for the most part the code is left up to you. As you work through this project, please <a target="_blank" href="https://review.udacity.com/#!/rubrics/1663/view">refer to the rubric</a> for guidance towards a successful submission.</p>
<p>Remember that your code should be your own, please do not plagiarize (<a target="_blank" href="https://udacity.zendesk.com/hc/en-us/articles/360001451091-What-is-plagiarism-">see here</a> for more information).</p>
<p>This notebook will be required as part of the project submission. After you finish it, make sure you download it as an HTML file and include it with the files you write in the next part of the project.</p>
<p>We've provided you a workspace with a GPU for working on this project. If you'd instead prefer to work on your local machine, you can find the files <a target="_blank" href="https://github.com/udacity/aipnd-project">on GitHub here</a>.</p>
<p>If you are using the workspace, be aware that saving large files can create issues with backing up your work. You'll be saving a model checkpoint in Part 1 of this project which can be multiple GBs in size if you use a large classifier network. Dense networks can get large very fast since you are creating N x M weight matrices for each new layer. In general, it's better to avoid wide layers and instead use more hidden layers, this will save a lot of space. Keep an eye on the size of the checkpoint you create. You can open a terminal and enter <code>ls -lh</code> to see the sizes of the files. If your checkpoint is greater than 1 GB, reduce the size of your classifier network and re-save the checkpoint.</p>


## Part 2 - Building the command line application

Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part.
Specifications

The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.

    Train a new network on a data set with train.py
        Basic usage: python train.py data_directory
        Prints out training loss, validation loss, and validation accuracy as the network trains
        Options:
            Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
            Choose architecture: python train.py data_dir --arch "vgg13"
            Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
            Use GPU for training: python train.py data_dir --gpu

    Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
        Basic usage: python predict.py /path/to/image checkpoint
        Options:
            Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
            Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
            Use GPU for inference: python predict.py input checkpoint --gpu

The best way to get the command line input into the scripts is with the argparse module in the standard library. You can also find a nice tutorial for argparse here.

## Rubric

<div class="_main--content-container--ILkoI"><div><div class="index--container--2OwOl"><div class="index--atom--lmAIo layout--content--3Smmq"><div class="ltr"><div class="index-module--markdown--2MdcR ureact-markdown "><h3 id="files-submitted">Files submitted</h3>
<div class="index-module--table-responsive--1zG6k"><table class="index-module--table--8j68C index-module--table-striped--3HHC-">
<thead>
<tr>
<th><strong>Criteria</strong></th>
<th><strong>Specification</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Submission Files</td>
<td>The submission includes all required files.</td>
</tr>
</tbody>
</table>
</div></div></div><span></span></div></div></div><div><div class="index--container--2OwOl"><div class="index--atom--lmAIo layout--content--3Smmq"><div class="ltr"><div class="index-module--markdown--2MdcR ureact-markdown "><h3 id="part-1-development-notebook">Part 1 - Development Notebook</h3>
<div class="index-module--table-responsive--1zG6k"><table class="index-module--table--8j68C index-module--table-striped--3HHC-">
<thead>
<tr>
<th><strong>Criteria</strong></th>
<th><strong>Specification</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Package Imports</td>
<td>All the necessary packages and modules are imported in the first cell of the notebook</td>
</tr>
<tr>
<td>Training data augmentation</td>
<td>torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping</td>
</tr>
<tr>
<td>Data normalization</td>
<td>The training, validation, and testing data is appropriately cropped and normalized</td>
</tr>
<tr>
<td>Data loading</td>
<td>The data for each set (train, validation, test) is loaded with torchvision's ImageFolder</td>
</tr>
<tr>
<td>Data batching</td>
<td>The data for each set is loaded with torchvision's DataLoader</td>
</tr>
<tr>
<td>Pretrained Network</td>
<td>A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen</td>
</tr>
<tr>
<td>Feedforward Classifier</td>
<td>A new feedforward network is defined for use as a classifier using the features as input</td>
</tr>
<tr>
<td>Training the network</td>
<td>The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static</td>
</tr>
<tr>
<td>Validation Loss and Accuracy</td>
<td>During training, the validation loss and accuracy are displayed</td>
</tr>
<tr>
<td>Testing Accuracy</td>
<td>The network's accuracy is measured on the test data</td>
</tr>
<tr>
<td>Saving the model</td>
<td>The trained model is saved as a checkpoint along with associated hyperparameters and the <code>class_to_idx</code> dictionary</td>
</tr>
<tr>
<td>Loading checkpoints</td>
<td>There is a function that successfully loads a checkpoint and rebuilds the model</td>
</tr>
<tr>
<td>Image Processing</td>
<td>The <code>process_image</code> function successfully converts a PIL image into an object that can be used as input to a trained model</td>
</tr>
<tr>
<td>Class Prediction</td>
<td>The <code>predict</code> function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image</td>
</tr>
<tr>
<td>Sanity Checking with matplotlib</td>
<td>A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names</td>
</tr>
</tbody>
</table>
</div></div></div><span></span></div></div></div><div><div class="index--container--2OwOl"><div class="index--atom--lmAIo layout--content--3Smmq"><div class="ltr"><div class="index-module--markdown--2MdcR ureact-markdown "><h2 id="part-2-command-line-application">Part 2 - Command Line Application</h2>
<div class="index-module--table-responsive--1zG6k"><table class="index-module--table--8j68C index-module--table-striped--3HHC-">
<thead>
<tr>
<th><strong>Criteria</strong></th>
<th><strong>Specification</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Training a network</td>
<td><code>train.py</code> successfully trains a new network on a dataset of images</td>
</tr>
<tr>
<td>Training validation log</td>
<td>The training loss, validation loss, and validation accuracy are printed out as a network trains</td>
</tr>
<tr>
<td>Model architecture</td>
<td>The training script allows users to choose from at least two different architectures available from torchvision.models</td>
</tr>
<tr>
<td>Model hyperparameters</td>
<td>The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs</td>
</tr>
<tr>
<td>Training with GPU</td>
<td>The training script allows users to choose training the model on a GPU</td>
</tr>
<tr>
<td>Predicting classes</td>
<td>The <code>predict.py</code> script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability</td>
</tr>
<tr>
<td>Top K classes</td>
<td>The <code>predict.py</code> script allows users to print out the top K classes along with associated probabilities</td>
</tr>
<tr>
<td>Displaying class names</td>
<td>The <code>predict.py</code> script allows users to load a JSON file that maps the class values to other category names</td>
</tr>
<tr>
<td>Predicting with GPU</td>
<td>The <code>predict.py</code> script allows users to use the GPU to calculate the predictions</td>
</tr>
</tbody>
</table>
</div></div></div><span></span></div></div></div></div>

## Project submission

For a successful project submission, you'll need to include these files in a ZIP archive:

    The completed Jupyter Notebook from Part 1 as an HTML file and any extra files you created that are necessary to run the code in the notebook
    The train.py and predict.py files from Part 2, as well as any other files necessary to run those scripts

You can download these files individually from the workspaces.

NOTE: Do not include the data in the submission archive.