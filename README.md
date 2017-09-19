# DotA_Predictor

#### this repo is still a work in progress

## Overview
DotA2 is a MOBA and has been one of the most played games on Steam, with over a million concurrent players at its peak. Its competitive scene is lively and boisterous, and players with the compedium of the season can often bet with virtual currency on the outcome of the game.
Drafting of the heroes in each team are of paramount importance and is a huge factor in determining which team has the highest probability of winning. As such, I have decided to write a simple predictor based on hero picks to guess accurately the outcome of the game.

This model is based on patch 7.06f. More features like MMR and roles might be added in the future.

## Replicating the results
Dependancies:
```
python
pandas
sklearn
```
## Experiments

#### First Approach - categorical columns of players
Trained LogReg, SVC and Perceptron with columns as player positions (player1-10) and the column data as the hero name.  
Did badly with the classifiers predicting all True for every data  
Possible causes: classifier was unable to learn that there were two contesting teams

#### Second (and last) Approach - binary inputs of features for every hero in every team
Transformed the data using transformdata.py into 225 columns.  
Columns are the names of every hero in each team i.e rad_abaddon, dire_abaddon, rad_axe, dire_axe ... and it's 1 if present and 0 if not.  
Used a MLP with 4 hidden layers [100,50,10,5] with high regularization to account for the inherent variance in the dataset.
LogReg, SVC and Perceptron did worse (probably due to their linearity) . 
```
Training a MLPClassifier...
Trained model in 18.870368003845215 seconds
Predictions made in 0.05964803695678711 seconds
F1 score and accuracy score for training set: 0.987127429806, 0.984199363733
Predictions made in 0.018578290939331055 seconds
F1 score and accuracy score for training set: 0.672674078408, 0.600428571429
MLPClassifier model saved!
```

## TODO
1) Add more training data .   
2) Try using a smaller set of features.  
