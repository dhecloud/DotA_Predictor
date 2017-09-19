from train_prediction import load_clf
import pandas as pd
from sklearn.neural_network import MLPClassifier


file1 = open("heroscolumns.txt", "r")
HEROS_COLUMNS = file1.read().strip("[").strip("]").strip("'").split("', '")

def predict_win(clf, features):

    y_pred = clf.predict(features)
    return y_pred[0]


def input_heroes():
    radcolors = ["Blue", "Teal", "Purple", "Yellow", "Orange"]
    direcolors= ["Pink", "Olive", "Light Blue", "Dark Green", "Brown"]
    radheroes = []
    direheroes = []
    for color1 in radcolors:
        print(color1+" : ", end= "")
        tmp = "rad_" + input()

        while tmp not in HEROS_COLUMNS:
            print("Invalid hero! Try again")
            tmp = "rad_"+input()
        radheroes.append(tmp)

    for color2 in direcolors:
        print(color2+" : ", end = "")
        tmp = "dire_"+input()
        while tmp not in HEROS_COLUMNS:
            print("Invalid hero! Try again")
            tmp = "dire_" + input()
        direheroes.append(tmp)
    return radheroes, direheroes

def create_df(rad, dire):
    tmpdict = {}
    for tmp in HEROS_COLUMNS:
        tmpdict[tmp] = [0]
    for radh in rad:
        tmpdict[radh] = [1]
    for direh in dire:
        tmpdict[direh] = [1]
    df = pd.DataFrame(tmpdict, columns=HEROS_COLUMNS)
    return df

if __name__ == "__main__":
    rad, dire = input_heroes()
    df = create_df(rad,dire)
    clf = load_clf("MLPClassifier")
    result = predict_win(clf, df)
    if result is True:
        print("Radiant is predicted to win!")
    else:
        print("Radiant is predicted to lose!")
