import pandas as pd
import numpy as np

CSV_TO_BE_TRANSFORMED1="data1.csv"
CSV_TO_BE_TRANSFORMED="data.csv"

file1 = open("heroscolumns.txt", "w")
def run():
    data1 = pd.read_csv(CSV_TO_BE_TRANSFORMED1)
    data2 = pd.read_csv(CSV_TO_BE_TRANSFORMED)
    frames = [data1, data2]
    data = pd.concat(frames)
    rows = data.shape[0]
    print(rows)
    radcols = ["player1", "player2", "player3", "player4", "player5"]
    direcols = ["player6","player7", "player8","player9", "player10"]
    heroes = set(list(data.player1.unique())+list(data.player2.unique()) +list(data.player3.unique()) + list(data.player4.unique()) + list(data.player5.unique()) )
    print(len(heroes))
    col = []
    for name in heroes:
        col.append("rad_"+name)
        col.append("dire_"+name)
    print(len(col))
    file1.write(str(col))
    newcsv = []
    for i in range(rows):
        tmpdict = {}
        for tmp in col:
            tmpdict[tmp] = 0
        for rad in radcols:
            colname = "rad_"+data[rad][i]
            tmpdict[colname] = 1
        for dire in direcols:
            colname = "dire_"+data[dire][i]
            tmpdict[colname] = 1
        tmpdict["rad_win"] = data["rad_win"][i]
        newcsv.append(tmpdict)

    newdf = pd.DataFrame(newcsv, columns=col)
    #newdf.to_csv("new"+CSV_TO_BE_TRANSFORMED,index=False, encoding='utf-8')




if __name__ == "__main__":
    run()
