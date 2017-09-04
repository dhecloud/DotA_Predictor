import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from IPython.display import display
from sklearn.preprocessing import scale
from time import time
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
import pickle
import os.path
import itertools

def show_data_stats(data):
    n_matches = data.shape[0]
    n_features = data.shape[1] - 1
    n_radwins = (data['rad_win'].value_counts())[0]
    win_rate = (float(n_radwins) / (n_matches)) * 100
    print("Total number of matches: " + str(n_matches))
    print("Number of features: " + str(n_features))
    print("Number of matches won by Radiant: " + str(n_radwins))
    print("Win rate of Radiant: " + str(win_rate) + "\n")

def augment_data(data):
    df = pd.DataFrame()
    for i in range(data.shape[0]):
        vals = data.values[i]
        b1 = list(itertools.permutations(vals[:5], 5))
        b2 = list(itertools.permutations(vals[5:10], 5))
        if (data['rad_win'][i]) == True:
            rad_win = tuple("T")
        else:
            rad_win = tuple("F")
        tmp = pd.DataFrame([y + x + rad_win for x in b2 for y in b1], columns=data.columns)
        df = df.append(data)
        if i % 50 == 0 and i != 0:
            print(str(i) + " data has been augmented")


    return df

def read_data(name):
    data = pd.read_csv(name)
    print("showing first few columns..")
    display(data.head())
    show_data_stats(data)
    print(data.shape[0])
    data = augment_data(data)
    print(data.shape[0])
    return data

def prepare_data(data):
    x_data = data.drop(['rad_win'],1)
    y_data = data['rad_win']
    # convert categorical data to continuous data
    x_data.player1 = x_data.player1.astype('str')
    x_data.player2 = x_data.player2.astype('str')
    x_data.player3 = x_data.player3.astype('str')
    x_data.player4 = x_data.player4.astype('str')
    x_data.player5 = x_data.player5.astype('str')
    x_data.player6 = x_data.player6.astype('str')
    x_data.player7 = x_data.player7.astype('str')
    x_data.player8 = x_data.player8.astype('str')
    x_data.player9 = x_data.player9.astype('str')
    x_data.player10 = x_data.player10.astype('str')

    output = pd.DataFrame(index = x_data.index)
    for col, col_data in x_data.iteritems():
        #convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
        output = output.join(col_data)
    print(str(len(output.columns)) + " proccessed feature columns\n")
    x_data, x_test, y_data, y_test = train_test_split(output, y_data,
                                                    test_size = 5000,
                                                    random_state = 2,
                                                    stratify = y_data)
    return x_data, y_data, x_test, y_test

def train_classifier(clf, x_data, y_data):

    start = time()
    clf.fit(x_data, y_data)
    end = time()

    print("Trained model in " + str(end-start) + " seconds")

def predict_outcome(clf, features, target):

    start = time()
    y_pred = clf.predict(features)
    end = time()
    print(len(y_pred))
    print(len(y_pred==True))
    print("Predictions made in " + str(end-start) + " seconds")
    f1score = f1_score(target, y_pred, pos_label=True)
    acc = sum(target == y_pred)/float(len(y_pred))
    print("F1 score and accuracy score for training set: " + str(f1score) + ", " + str(acc))

    return f1score, acc

def train_predict(clf, x_data, y_data, x_test, y_test):

    print("\nTraining a " + clf.__class__.__name__ + "...")

    train_classifier(clf, x_data, y_data)
    #train
    f1, acc = predict_outcome(clf, x_data, y_data)
    #test
    f1, acc = predict_outcome(clf, x_test, y_test)

def save_clf(clf):
    with open( clf.__class__.__name__ + '.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    print(clf.__class__.__name__ + " model saved!")

def load_clf(clf):
    with open( clf + '.pkl', 'rb') as fid:
        loaded_clf = pickle.load(fid)
    print(loaded_clf.__class__.__name__ + " model loaded!")
    return loaded_clf

def save_file_exist(name):
    return os.path.exists(name+".pkl")

if __name__ == "__main__":

    for clf in ["SVC", "LogisticRegression"]:
        if save_file_exist(clf):
            load_clf(clf)
        else:
            data = read_data("data.csv")
            x_data, y_data, x_test, y_test = prepare_data(data)
            print(y_data.head())
            if clf == "SVC":
                SVC = SVC(random_state = 912, kernel='rbf')
                train_predict(SVC, x_data, y_data, x_test, y_test)
                predict_outcome(SVC, x_test, y_test)
                save_clf(SVC)
            elif clf == "LogisticRegression":
                logreg = LogisticRegression(random_state=42)
                train_predict(logreg, x_data, y_data, x_test, y_test)
                predict_outcome(logreg, x_test, y_test)
                save_clf(logreg)
