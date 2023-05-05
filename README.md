# Using genetic algorithms for tuning autoencoders architecture

# Links
* [Introduction](#problem-statement--introduction)
  * [Project description](#project-description)
  * [Article](https://drive.google.com/file/d/13g3zIlCQYJ6vfD-62MEhq0lYbYSEyAOm/view?usp=share_link)
* [Installation](#installation)
  * [Python version](#python-requirements)
  * [Directories structure](#directory-strure)
  * [Requirements](#libraries)
  * [How to run](#run-project)
* [Methodology](#methodology)
  * [Gene representation](#representation)
  * [Mutation and Crossover](#mutation-and-crossover)
  * [Fitness function](#fitness-function)
* [Credits](#credits)



# Problem statement / Introduction

### Project description
[Autoencoders](https://en.wikipedia.org/wiki/Autoencoder) are gaining 
more and more interest as their application expands. 
However, creating these models is not as simple 
as it may seem, as advanced understanding of models workflow, 
clean training data and good architecture are minimum requirements 
to achieve acceptable performance. 
We overcame one of these challenges by using 
[genetic algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm) 
for architecture tuning. As a result, we achieved a pipeline to improve
architecture on [Cats faces dataset](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models)
cat generation.

### Article link
If you are interested in our research, you can read our findings [here](https://drive.google.com/file/d/13g3zIlCQYJ6vfD-62MEhq0lYbYSEyAOm/view?usp=share_link)


# Installation
### Python requirements
First of all you should have python installed ([version 3.9](https://www.python.org/downloads/) or above). 
Please make sure have it installed before moving to a next step

### Directory strure
Second thing you need to do is configure directories. You need to create directories:
* `data` - all datasets (we worked with [these](https://drive.google.com/drive/folders/1XaxLegjsSXP0ReW9h0h2E4wpWlUBBTh4))
* `checkpoints` - all models checkpoints 
* `models` -  all final models
* `samples` - image generation results of models

If you are working on linux, here is a shell script to create all directories:
```shell
mkdir "checkpoints"
mkdir "data"
mkdir "models"
mkdir "samples"
```

**Important** all directories above should be on the same level as `src` and `test`!

### Libraries
And last things are downloading required libraries:
```shell
pip install -r requirements.txt
```

### Run project
After you have done everything above, you can run `main.py` via console or any IDE you use:
```shell
python main.py
```


# Methodology

The goal of optimizing neural network architecture is to determine the most effective parameters, layers, 
and internal structure of the network to maximize its performance.

### Representation
To achieve this, we first need to select a suitable representation. We can focus on defining the encoder 
part of the network, as the decoder is typically the same but reversed. We define a representation as an 
activation function and sequence of layers in the format of `layerType_featuresIn_featuresOut_kernelSize`. 
However, we must apply certain restrictions such as ensuring that the number of features decreases towards the 
end of the encoder and that convolutions come before fully connected layers. Additionally, the activation 
function should be consistent throughout the network.

### Mutation and crossover
Next, we need to define mutation and crossover operations. Mutation can be achieved by slightly 
altering the number of features in some layers and adding or deleting layers. Crossover involves exchanging 
portions of the networks while maintaining the mentioned restrictions.

### Fitness function
Finally, we must define a fitness function. For our problem we decided that the fitness function will 
be identical to validation loss, so we aim to minimize it.



# Credits
* Polina Zelenskaya (p.zelenskaya@innopolis.university)
* Karim Galliamov (k.galliamov@innopolis.university)
* Igor Abramov (ig.abramov@innopolis.university)