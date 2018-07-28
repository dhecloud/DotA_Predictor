import dota2api
import pandas as pd
import sys
import time
#returns list of ids of the most recent 100 all pick matches
def get_allpick_ids():
    matchids= []
    api = dota2api.Initialise("2CE465B224F03B438575322A3B85BD34")
    match = api.get_match_history(game_mode=1,min_players=10,matches_requested=100)
    for j in range(100):
        matchids.append(match['matches'][j]['match_id'])
    start_id = match['matches'][len(match['matches'])-1]['match_id'] + 1
    #remove duplicates
    matchids= remove_duplicates(matchids)
    return matchids

def remove_duplicates(ids):
    return sorted(list(set(ids)))

#return list of integers
def remove_duplicates_num_in_str(line):
    return remove_duplicates([int(i) for i in line.split(", ")])

def tidy_dataset():
    file1 = open("allpick.txt","r")
    line = file1.read()
    file1.close()
    print(len(line.split(", ")))
    line = remove_duplicates_num_in_str(line)
    print(len(line))
    file1 = open("allpick.txt","w")
    file1.write(str(line).strip('[').strip(']'))
    file1.close

def get_ap_details(lists):
    file2 = open("apdetails.txt","a")
    hero_ids=[]
    api = dota2api.Initialise("2CE465B224F03B438575322A3B85BD34")
    print("Api Initialised!")
    k=0
    l=0
    for i in lists:
        try:
            details = api.get_match_details(match_id=i)
            #print("details retrieved")
            tmpdict={}
            for j in range(10):
                tmpname ='player'+str(j+1)
                tmpdict[tmpname] = details["players"][j]["hero_name"]
            tmpdict["rad_win"] = details["radiant_win"]
            hero_ids.append(tmpdict)
            file2.write(str(details))
            k += 1
            if k % 20 == 0:
                print(str(k)+"matches retrieved!")
        except KeyError:
            l += 1
            print ("Error!")
            print ("Pausing for 5 seconds :(..")
            time.sleep(5)
            continue
        except:
            l += 1
            print ("Timeout!")
            print ("Pausing for 10 minutes :(..")
            time.sleep(600)
            continue


    print(str(l) + " errors ignored!")
    print(str(k) + " matches downloaded!")
    df = pd.DataFrame(hero_ids, columns=hero_ids[0].keys())
    df.to_csv("data.csv",index=False, encoding='utf-8')
    print("saved to csv")
    file2.close()


if __name__ == "__main__":
    # 0 for getting new ap ids, 1 for getting ap details
    if sys.argv[1] == "0":
        ap_ids = get_allpick_ids()
        file1 = open("allpick.txt","a")
        file1.write(", "+str(ap_ids).strip('[').strip(']'))
        file1.close()
        tidy_dataset()
    elif sys.argv[1] == "1":
        file1 = open("allpick.txt","r")
        lists = remove_duplicates_num_in_str(file1.read())
        file1.close()
        get_ap_details(lists)
