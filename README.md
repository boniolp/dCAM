# d-CAM : Dimension-wise Class Activation Map for Explaining Multivariate Time Series Classification #

<p align="center">
<img src="ressources/intro_figure.png" alt="drawing" width="400"/>
</p>

### Class Activation Map and multivariate time series classification ###

Data series classification is an important and challenging problem in data science. Explaining the classification decisions by finding the discriminant parts of the input that led the algorithm to some decision is a real need in many applications. Convolutional neural networks perform well for the data series classification task; though,the explanations provided by this type of algorithms are poor for the specific case of multivariate data series. Solving this important limitation is a significant challenge. We propose a novel method that addresses the above challenge by highlighting both the temporal and dimensional discriminant information. Our contribution is two-fold: we first describe a new convolutional architecture that enables the comparison of dimensions; then, we propose a novel method that returns dCAM, a Dimension-wise ClassActivation Map specifically designed for multivariate time series. 

This repository is dedicated to [our paper](https://dl.acm.org/doi/abs/10.1145/3514221.3526183) titled "dCAM : Dimension-wise Class Activation Map for Explaining Multivariate Time Series Classification" published in the [Proceedings of the 2022 International Conference on Management of Data](https://dl.acm.org/doi/proceedings/10.1145/3514221) also available on [here](https://www.researchgate.net/publication/361416963_dCAM_Dimension-wise_Class_Activation_Map_for_Explaining_Multivariate_Data_Series_Classification).

## Data 
The data used in this project comes from two sources: 
- The [UCR/UEA archive](http://timeseriesclassification.com/TSC.zip). Informations are provided in the data folder in order to download our datasets.

## Code 
The code is divided as follows: 
- The [src/](https://github.com/boniolp/dCAM/tree/main/src) folder that contains:
  - models/CNN_models.py:
    - CNN architecture (class ConvNet)
    - dCNN/cCNN architecture (class ConvNet2D)
    - ResNet architecture (class ResNet)
    - dResNet/cResNet architecture (class dResNet)
    - Inception Time architecture (class inceptiontime)
    - dInception Time/cInception Time architecture (class dinceptiontime)
    - CNN-MTEX architecture (class ConvNetMTEX)
  - models/RNN_models.py:
    - LSTM architecture (class LSTMClassifier)
    - RNN architecture (class RNNClassifier)
    - GRU architecture (class GRUClassifier)

  - the explanation/:
    - CAM code (class CAM)
    - cCAM code (class cCAM)
    - dCAM code (class DCAM)
    - grad-CAM used for CNN-MTEX (class GradCAM)

- The [examples/](https://github.com/boniolp/dCAM/tree/main/examples) folder that contains:
  - Synthetic_experiment-CAM.ipynb: An example on how to use CNN-based models and the Class Activation Map.
  - Synthetic_experiment-DCAM.ipynb: An example on how to use dCNN-based models and the dCAM.

- The [experiments/](https://github.com/boniolp/dCAM/tree/main/experiments) folder that contains:
  - classification/: scripts in order to reproduce our classification results.
  - explanation/: scripts in order to reproduce our explanation results.
  - execution-time/: scripts in order to reproduce our execution_time results.


