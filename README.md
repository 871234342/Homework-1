# HOMEWORK 1 - Car Classification

This model is for image classification for the Standford Cars dataset.
Data set can be downloaded [here](https://www.kaggle.com/c/cs-t0828-2020-hw1/data).

In order to reproduce my training and inferrinf process, please make sure the packages listed in requirement.txt are installed.

### Hardware
- Ubuntu 18.04.5 LTS
- Intel® Xeon® Silver 4210 CPU @ 2.20GHz
- NVIDIA GeForce RTX 2080 Ti

### Reproduce Submission
To reproduce my submission without training, do the following:
1. [installation]()
2. [Data Preparation]()
3. [Inference]()


### Installation
Install all the requirments sepcified in requirments.txt
`pip install -r requirments.txt`


### Data Preparation
The data should be placed as follows:
```
repo
  +- training_data
  |  +- 000001.jpg
  |  +- 000002.jpg
  |  +- ...
  |
  +- testing_data
  |  +- 000004.jpg
  |  +- 000005.jpg
  |
  +- training_labels.csv
  +- train.py
  +- infer.py
  +- weights.pth   (needed for inference)
  |  ...
```
Where training_data folder contains all the training images, and testing_data folder contains all the testing images. The training_labels.csv file should contain the file name and corresponding label of each image in training_data folder. Please check the file to see the expected format.

### Training
To train, simply run the train.py file. weights.pth file should be created beside train.py. The batch_size is set to be 12. Make it smaller if memory is not sufficent.

### Inference
for inference, please download the weights file [here](https://drive.google.com/file/d/1nQPV5yNpJn1VEM-VL7g_6Y6KT0REy0Cl/view?usp=sharing). Simply run infer.py and a csv file named testing_labels.csv containing images file names and their corresponding predictions will be created.
