# Projet-Recherche

Hand Gesture Recognition using OpenCV and Tensorflow

## Description
As fourth-year student in a IT engineer school in France, we were asked to do research on Hand Gesture Recognition and to make a prototype out of it as an innovation.

## Installation and Requirements
Python 3.7
Anaconda Environment with OpenCV 4.2.0.34 and TensorFlow 2.0.0
Python IDE - Anything is fine. For our case, we used JetBrains PyCharm

## Main Content
### Python Code
[TrainModel.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/TrainModel.py) : to train both Images model and Labels model.
[capture-image.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/capture-image.py) : to create datasets using our webcam.
[createModel.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/createModels.py) : to create an Images model and a Labels model from our datasets (not included in the repository).
[dataAugmentation.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/dataAugmentation.py) : to create and increase considerably various versions of our datasets.
[gesture.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/gesture.py) : to create datasets using our webcam.
[predictModel.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/predictModel.py) : to predict our pattern on a webcam.

### Others
[trainModelImagesAndLabels.zip](https://github.com/sambao407/Projet-Recherche/blob/master/data/Datasets/trainModelImagesAndLabels.zip) : zip containing Images model and Labels model needed to be trained.
[dataTest folder](https://github.com/sambao407/Projet-Recherche/tree/master/data/dataTest) : folder containing pictures to test our recognition system on it.
[trainModel folder](https://github.com/sambao407/Projet-Recherche/tree/master/data/trainModel) : folder containing our trained model.

## Hand patterns
We went for 6 different patterns: finger pointing to the left side, finger pointing to the right side, fist, palm, none and thumb.

## Usage
Use [predictModel.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/predictModel.py) to predict directly our hand recognition system on your computer.

If you want to use your own pattern, make sure to put your datasets into [Datasets folder](https://github.com/sambao407/Projet-Recherche/tree/master/data/Datasets) divided per class pattern. In our case, there are 6 folders : fingerLeft, fingerRight, fist, none, palm, thumb containing their respective datasets. You have to change the class names on [createModel.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/createModels.py) and [predictModel.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/predictModel.py) according to the class pattern folders you have created.
Then, you have to use [createModel.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/createModels.py) and [TrainModel.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/TrainModel.py).
From that, you can test your own trained model with [predictModel.py](https://github.com/sambao407/Projet-Recherche/blob/master/code/predictModel.py).

## License
[CESI École d’Ingénieurs - parcours exia](https://orleans.cesi.fr/)
