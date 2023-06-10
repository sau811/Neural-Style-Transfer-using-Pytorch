# Neural-Style-Transfer-using-Pytorch
## Introduction
Neural style transfer is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.
## Content and Style Representation
### Content Representation
Along the processing hierarchy, the convolutional neural network generates image representations. As we progress further into the network, the representations will be more concerned with structural features or actual content rather than detailed pixel data. We can reconstruct the images using the feature maps of that layer to obtain these representations. The exact image will be reproduced by reconstruction from the lower layer. In contrast, the higher layer's reconstruction will capture the high-level content, so we refer to the higher layer's feature responses as the content representation.

<img src="https://github.com/sau811/Neural-Style-Transfer-using-Pytorch/assets/92877918/6d57041b-eb56-4652-9312-419c07fc97f9" width="300" height="400">

### Style Representation
To extract the style content representation, we build a feature space on top of the filter responses in each network layer. It consists of the correlations between the various filter responses over the feature maps' spatial extent. The texture information of the input image is captured by the filter correlation of different layers. This generates images on an increasing scale that match the style of a given image while discarding information from the global arrangement. Style representation refers to this multi-scale representation.

<img src="https://github.com/sau811/Neural-Style-Transfer-using-Pytorch/assets/92877918/c337afb5-8d7a-43e1-a8de-04f2fad5f991" width="563" height="395">

We perform content and style reconstructions using a pre-trained VGG19 network's convolutional neural network. We generate the artistic image by entangling the structural information from the content representation and the texture/style information from the style representation. We can emphasise either style or content reconstruction. A strong emphasis on style will result in images that match the appearance of the artwork, effectively providing a texturized version of it, but show very little of the photograph's content. When the emphasis is on content, the photograph is easily identified, but the painting style is not as well-matched. We use gradient descent on the generated image to find another image that matches the feature responses of the original image.
## Implementation
### Importing Libraries
We'll start by importing the necessary libraries. To implement the style transfer using PyTorch, we will import torch, torchvision, and PIL.
### Load the model
In this case, we'll use torchvision.models() to load the pre-trained VGG19 model. The vgg19 model has three components: avgpool, classifier, and avgpool.

The feature contains all of the convolutional, maximum pool, and ReLu layers, while avgpool contains the average pool layer.

The dense layers are held by the classifier.
To implement style transfer, we will only use the convolutional neural network, so import vgg19 features. Don't forget to use the GPU if one is available. It will shorten the training period.
### Image Preprocessing
Preprocessing is required to make an image compatible with the model. We will perform some basic preprocessing

ToTensor: to transform an image into a tensor

For the pretrained vgg19 model, you can normalise the tensor with mean  and standard deviation. But remember to return it to its original scale. Define a function that loads and preprocesses the image using the PIL library. Add an extra dimension at the 0th index, using unsqueeze() for batch size, and then load and return it to the device.
### Get feature representations
Let's create a class to provide feature representations for the intermediate layers. Because intermediate layers serve as a complex feature extractor, they are used. As a result, these can be used to describe the style and content of the input image. In this class, we will create a model by removing the unused layers of the vgg19 model (layers beyond conv5_1) and extracting the activations or feature representations of the 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2' and 'conv5_1' layers (index values [0, 5, 10, 19, 21, 28]). Return the array containing the activations of the five convolutional layers.
### Define loss
Define the content and style loss using the formulas.
### Train the model
Initialize the variables that we will need to train our model and set the hyperparameters.

optimizer = optim.Adam([target],lr = 0.003),
alpha = 1,
beta = 1e5,
epochs = 3000,
show_every = 500.
### Results
Find the total loss and plot the results.

<img src="https://github.com/sau811/Neural-Style-Transfer-using-Pytorch/assets/92877918/659d69e1-eda3-4b5a-8292-ef8bda602c2c" width="167" height="217">



