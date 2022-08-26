# d-CAM : Dimension-wise Class Activation Map for Explaining Multivariate Time Series Classification #


### Class Activation Map and multivariate time series classification ###

Data series classification is an important and challenging problem in data science. Explaining the classification decisions by finding the discriminant parts of the input that led the algorithm to some decision is a real need in many applications. Convolutional neural networks perform well for the data series classification task; though,the explanations provided by this type of algorithms are poor for the specific case of multivariate data series. Solving this important limitation is a significant challenge. We propose a novel method that addresses the above challenge by highlighting both the temporal and dimensional discriminant information. Our contribution is two-fold: we first describe a new convolutional architecture that enables the comparison of dimensions; then, we propose a novel method that returns dCAM, a Dimension-wise ClassActivation Map specifically designed for multivariate time series. 


### Contents ###

#### src

contains the source code of

- models/CNN_models.py:
-- CNN architecture (class ConvNet)
-- dCNN/cCNN architecture (class ConvNet2D)
-- ResNet architecture (class ResNet)
-- dResNet/cResNet architecture (class dResNet)
-- Inception Time architecture (class inceptiontime)
-- dInception Time/cInception Time architecture (class dinceptiontime)
-- CNN-MTEX architecture (class ConvNetMTEX)

- models/RNN_models.py:
-- LSTM architecture (class LSTMClassifier)
-- RNN architecture (class RNNClassifier)
-- GRU architecture (class GRUClassifier)

- explanation/:
-- CAM code (class CAM)
-- cCAM code (class cCAM)
-- dCAM code (class DCAM)
-- grad-CAM used for CNN-MTEX (class GradCAM)

#### examples

contains several notebook illustrating how one can use the source code.

#### data

contains a dataset example that we use to create our synthetic datasets 
