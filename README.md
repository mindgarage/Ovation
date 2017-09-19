# Ovation

![Ovation](https://docs.google.com/drawings/d/e/2PACX-1vT8ek6nSC2UNliv7exfhoG20y1_GU1aXJoQSuZqza7z4KyxZpnfQ-XX4JyvFerMufwgked-5kAHCqAh/pub?w=1172&h=259)


(This repository is still under construction. Please let us know if you
find any issues :-) )

The Ovation Framework is a new framework for developing your own Conversational
Intelligence model. The framework was constructed by the
[MindGarage](mindgarage.de) in collaboration with
[Insiders Technologies](http://www.insiders-technologies.de/).

The framework contains several utility classes to help you to easily build
new Conversational Intelligence architectures. The idea is to give you a head
start in your NLP/Deep Learning project, removing from you the
burden of
thinking about how to get the data, or how to develop a simple Deep
Learning model. You can start from the examples here and build your
own model based on them.

This repository was constructed with a set of NLP/Deep Learning tasks in
mind. The datasets and models that we chose to initially support reflect
this set of tasks. We intend to keep expanding it to more and more datasets,
tasks and models.

# How to use this Repository?

## Installation

This code expects Python 3.4+ and portaudio (as dependency for PyAudio) installed. We recommend you to use a virtual python environment.
You can create a new virtual environment with:

```sh
# Change directory to the place where you'd like to create a new virtual
# environment
cd /path/to/where/you/want/to/put/your/environment

# Install virtualenv
pip install virtualenv
# Create the virtual environment with Python 3
virtualenv -p 3 name_of_your_enviroment
## (sometimes one have to provide the full path to python, e.g. on mac: virtualenv -p /usr/local/bin/python3 )
# Activate the virtual environmnent
source name_of_your_enviroment/bin/activate
```

This will create a new folder with the name `name_of_your_enviroment`. With
the environment set, you will now need to install some dependencies in the
repository. For convenience,
we provide a `requirements.txt` file that you can use directly. Just run:

```sh
# Clone this repository:
git clone https://github.com/mindgarage/Ovation.git

# Enter the new folder
cd Ovation

# Install all the requirements
pip install -r requirements.txt
sh setup_packages.sh

# Tell python where to find the modules
export PYTHONPATH=$PYTHONPATH:$PWD
```

Now you are all set to use the code!


# What you will find here?

Below is a small summary of what you can find in this repository.
This repository was created with the following _pipeline_ in mind:

 * **datasets**: To build any Deep Learning model, you need data. Datasets
	that can be found in the internet come in any format, and it may take
	hours for one to reorganize them into the format that is convenient
	for him. In the `datasets` folder, you will find a set of utility classes
	that simply load the data for you and allow it to be accessed in
	several ways that we deemed useful for performing the Deep Learning
	tasks we had in mind.
 * **models**: Now that we have access to the data, we need to write models
	that receive the data (in a suitable format) and output some result.
	In the `models` folder you will find some example model classes that
	perform tasks such as Named Entity Recognition, Sentiment
	Classification and Intent Classification.
 * **templates**: Examples of how to use the classes are written in the
	`templates` folder.

Additionally, the following folders have some other useful code:

 * **tools**: Some standalone scripts.
 * **utils**: Utility functions used by the rest of the code.
 * **tests**: Some code used for testing the functionalities above


