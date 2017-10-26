import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from IPython.display import display
from sklearn.preprocessing import scale
from time import time
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
import pickle
import os.path
import os
import itertools

def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

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
    frames = pd.read_csv(name)
    print(frames.shape[0])
    return frames

def prepare_data_new(data):
    y_data = data['rad_win']
    y_data = y_data.values
    x_data = data.drop(['rad_win'],1)
    x_data = x_data.values
    x_data1, x_test, y_data1, y_test = train_test_split(x_data, y_data,
                                                    test_size = 15000,
                                                    random_state = 2,
                                                    stratify = y_data)
    return x_data1, y_data1, x_test, y_test

def prepare_data(data):
    y_data = data['rad_win']
    x_data = data.drop(['rad_win'],1)

    output = pd.DataFrame(index = x_data.index)
    for col, col_data in x_data.iteritems():
        #convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
        output = output.join(col_data)
    print(str(len(output.columns)) + " proccessed feature columns\n")
    x_data, x_test, y_data, y_test = train_test_split(output, y_data,
                                                    test_size = 3000,
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
    for clf in ["MLPClassifier"]:
        if save_file_exist(clf):
            data1 = read_data("newdata.csv")
            data2 = read_data("newdata1.csv")
            frames = [data1, data2]
            data = pd.concat(frames)
            show_data_stats(data)
            clfa = load_clf(clf)
            x_data, y_data, x_test, y_test = prepare_data_new(data)
            #train_predict(clfa, x_data, y_data, x_test, y_test)
            predict_outcome(clfa, x_test, y_test)
            #save_clf(clfa)

        else:
            data1 = read_data("newdata.csv")
            data2 = read_data("newdata1.csv")
            data3 = read_data("newdata2.csv")
            frames = [data1, data2, data3]
            data = pd.concat(frames)
            show_data_stats(data)
            x_data, y_data, x_test, y_test = prepare_data_new(data)
            if clf == "SVC":
                clfa = SVC(random_state = 912, kernel='rbf')
            elif clf == "LogisticRegression":
                clfa = LogisticRegression(random_state=42, warm_start = True)
            elif clf == "Perceptron":
                clfa = Perceptron(random_state=42, warm_start = True)
            elif clf == "MLPClassifier":
                clfa = MLPClassifier(solver = 'adam', alpha = 1.2, hidden_layer_sizes=(100, 50, 25, 10), random_state=1, warm_start=True)
            train_predict(clfa, x_data, y_data, x_test, y_test)
            save_clf(clfa)
