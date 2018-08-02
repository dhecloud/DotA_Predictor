def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pickle
import time
import od_python
import pandas as pd
from od_python.rest import ApiException
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier
import sys



def get_heroes_stats(write=False):
    stats_api = od_python.HeroStatsApi()
    herostats=[]
    api_response = stats_api.hero_stats_get()
    for i in range(len(api_response)):
        stats = api_response[i]
        herostats.append({"id":stats.id, "name":stats.name, "localized_name":stats.localized_name, "pro_win":stats.pro_win, "pro_pick":stats.pro_pick, "hero_id":stats.hero_id, "pro_ban":stats.pro_ban, "_1_pick":stats._1_pick, "_1_win":stats._1_win,
         	              "_1_lose":(stats._1_pick - stats._1_win), "_2_pick":stats._2_pick, "_2_win":stats._2_win, "_2_lose":(stats._2_pick - stats._2_win), "_3_pick":stats._3_pick, "_3_win":stats._3_win, "_3_lose":(stats._3_pick - stats._3_win), 
                          "_4_pick":stats._4_pick, "_4_win":stats._4_win, "_4_lose":(stats._4_pick - stats._4_win),"_5_pick":stats._5_pick, "_5_win":stats._5_win, "_5_lose":(stats._5_pick - stats._5_win),
                          "_6_pick":stats._6_pick, "_6_win":stats._6_win, "_6_lose":(stats._6_pick - stats._6_win),"_7_pick":stats._7_pick, "_7_win":stats._7_win, "_7_lose":(stats._7_pick - stats._7_win)})
    if (write):
        with open("hero_data/herostats.txt","w") as f:
            f.write(str(herostats))
        
    return herostats

def get_heroes(write=False):
    hero_api = od_python.HeroesApi()
    api_response = hero_api.heroes_get()
    heroes = []
    for i in range(len(api_response)):
        hero = api_response[i]
        heroes.append({"id":hero.id, "name":hero.name, "localized_name":hero.localized_name, "primary_attr":hero.primary_attr, "attack_type":hero.attack_type, "roles":hero.roles})
    if (write):
        with open("hero_data/heroes.txt","w") as f:
            f.write(str(heroes))
    return heroes

def get_heroes_stats_combined(refresh=False):
    if (refresh):
        print("Getting data from API..")
        herostats = get_heroes_stats(True)
        heroes = get_heroes(True)
    
        assert len(heroes) == len(herostats)
        heroes_stats_combined = []
        for i in range(len(heroes)):
            heroes_stats_combined.append({**heroes[i], **herostats[i]})
            
        with open("hero_data/heroes_stats_combined.txt","w") as f:
            f.write(str(heroes_stats_combined))
    else:
        with open("hero_data/heroes_stats_combined.txt","r") as f:
            heroes_stats_combined = eval(f.readline())

    positions = {}
    for i in range(len(heroes_stats_combined)):
        positions[heroes_stats_combined[i]['id']] = i
        
    with open("hero_data/hero_positions.txt","w") as f:
        f.write(str(positions))
        
    return heroes_stats_combined

def add_win_rate(heroes_stats_combined):
    for i in range(len(heroes_stats_combined)):
        heroes_stats_combined[i]['_1_win_rate'] = (heroes_stats_combined[i]["_1_win"] / heroes_stats_combined[i]["_1_pick"])
        heroes_stats_combined[i]['_2_win_rate'] = (heroes_stats_combined[i]["_2_win"] / heroes_stats_combined[i]["_2_pick"])
        heroes_stats_combined[i]['_3_win_rate'] = (heroes_stats_combined[i]["_3_win"] / heroes_stats_combined[i]["_3_pick"])
        heroes_stats_combined[i]['_4_win_rate'] = (heroes_stats_combined[i]["_4_win"] / heroes_stats_combined[i]["_4_pick"])
        heroes_stats_combined[i]['_5_win_rate'] = (heroes_stats_combined[i]["_5_win"] / heroes_stats_combined[i]["_5_pick"])
        heroes_stats_combined[i]['_6_win_rate'] = (heroes_stats_combined[i]["_6_win"] / heroes_stats_combined[i]["_6_pick"])
        heroes_stats_combined[i]['_7_win_rate'] = (heroes_stats_combined[i]["_7_win"] / heroes_stats_combined[i]["_7_pick"])
    
    return heroes_stats_combined
    
def add_features(heroes_stats_combined):
    heroes_stats_combined = add_win_rate(heroes_stats_combined)
    
    return heroes_stats_combined


def get_matches():
    api_instance = od_python.PublicMatchesApi()
    mmr_ascending = 56 # int | Order by MMR ascending (optional)
    mmr_descending = 56 # int | Order by MMR descending (optional)
    less_than_match_id = 56 # int | Get matches with a match ID lower than this value (optional)
    matches = []
    while (1):
        with open("matches/matches_ap.txt","r") as f:
            matches = eval(f.readline())
        print("Number of matches: " + str(len(matches)))
        api_response = api_instance.public_matches_get()
        
        for i in range(len(api_response)):
            matches.append({"match_id":api_response[i].match_id, "match_seq_num":api_response[i].match_seq_num, "radiant_win":api_response[i].radiant_win, "start_time":api_response[i].start_time, "duration":api_response[i].duration, "radiant_team":api_response[i].radiant_team, "dire_team":api_response[i].dire_team})
        dup = []
        for i in range(len(matches)):
            for j in range(i+1,len(matches)):
                if int(matches[i]['match_id']) == int(matches[j]['match_id']):
                    dup.append(j)
        dup = sorted(list(set(dup)),reverse=True)
        for i in dup:
            matches.pop(i)
        print("Number of unique matches: " + str(len(matches)))
        with open("matches/matches.txt","w") as f:
            f.write(str(matches))
        time.sleep(2)
        get_ap_matches()
        features_check()
        create_one_hot_hero_features()
        print("sleeping...")
        time.sleep(70)

def get_ap_matches():
    api_instance = od_python.MatchesApi()
    with open("matches/matches.txt","r") as f:
        matches = eval(f.readline())
    with open("matches/matches_ap.txt","r") as f:
        matches_all_pick = eval(f.readline())
    ids = [x['match_id'] for x in matches_all_pick]
    for match in matches:
        if match['match_id'] in ids:
            continue
        abandon = False
        api_response = api_instance.matches_match_id_get(match['match_id'])
        # print((type(api_response.game_mode)))
        # print((api_response.game_mode))
        if (api_response.game_mode == 1 or 22):
            # print("allpick!")
            for i in range(10):
                if (api_response.players[i].abandons > 0):
                    abandon = True
                    # print("someone abandoned")
            
            if not abandon:  
                match["throw"] = api_response.throw
                match["loss"] = api_response.loss
                match['patch'] = api_response.patch
                match['region'] = api_response.region
                match['skill'] = api_response.skill
                t1gpm, t1xpm, t1kpm, t1lhpm, t1hdpm, t1hhpm, t1td, t2gpm, t2xpm, t2kpm, t2lhpm, t2hdpm, t2hhpm, t2td, t1_complexity, t2_complexity, t1_escape, t1_initiator, t1_pusher, t1_durable, t1_nuker, t1_carry, t1_disable, t1_jungler, t1_support, t2_escape, t2_initiator, t2_pusher, t2_durable, t2_nuker, t2_carry, t2_disable, t2_jungler, t2_support, p1_wr, p2_wr, p3_wr, p4_wr, p5_wr, p6_wr, p7_wr, p8_wr, p9_wr, p10_wr= create_features(list(map(int, match['radiant_team'].split(',') + match['dire_team'].split(','))))
                match['t1gpm'] = t1gpm
                match['t1xpm'] = t1xpm
                match['t1kpm'] = t1kpm
                match['t1lhpm'] = t1lhpm
                match['t1hdpm'] = t1hdpm
                match['t1hhpm'] = t1hhpm
                match['t1td'] = t1td
                match['t1_complexity'] = t1_complexity
                match['t1_escape'] = t1_escape
                match['t1_initiator'] = t1_initiator
                match['t1_pusher'] = t1_pusher
                match['t1_durable'] = t1_durable
                match['t1_nuker'] = t1_nuker
                match['t1_carry'] = t1_carry
                match['t1_disable'] = t1_disable
                match['t1_jungler'] = t1_jungler
                match['t1_support'] = t1_support
                
                match['t2gpm'] = t2gpm
                match['t2xpm'] = t2xpm
                match['t2kpm'] = t2kpm
                match['t2lhpm'] = t2lhpm
                match['t2hdpm'] = t2hdpm
                match['t2hhpm'] = t2hhpm
                match['t2td'] = t2td
                match['t2_complexity'] = t2_complexity                
                match['t2_escape'] = t2_escape
                match['t2_initiator'] = t2_initiator
                match['t2_pusher'] = t2_pusher
                match['t2_durable'] = t2_durable
                match['t2_nuker'] = t2_nuker
                match['t2_carry'] = t2_carry
                match['t2_disable'] = t2_disable
                match['t2_jungler'] = t2_jungler
                match['t2_support'] = t2_support
                
                match['p1_wr'] = p1_wr
                match['p2_wr'] = p2_wr
                match['p3_wr'] = p3_wr
                match['p4_wr'] = p4_wr
                match['p5_wr'] = p5_wr
                match['p6_wr'] = p6_wr
                match['p7_wr'] = p7_wr
                match['p8_wr'] = p8_wr
                match['p9_wr'] = p9_wr
                match['p10_wr'] = p10_wr
                
                # print(match['throw'])
                # print(type(match['throw']))
                # print(match['loss'])
                # print(type(match['loss']))
                matches_all_pick.append(match)
                with open("matches/matches_ap.txt","w") as f:
                    f.write(str(matches_all_pick))
        time.sleep(2)
    with open("matches/matches_ap.txt","w") as f:
        f.write(str(matches_all_pick))

def input_heroes_complexity():
    new_stats = []
    with open("hero_data/heroes_stats_combined.txt","r") as f:
        heroes = eval(f.readline())
    for hero in heroes:
        print(hero['localized_name'])
        complexity = input()
        hero['complexity'] = int(complexity)
        new_stats.append(hero)
        print(hero)
    with open("hero_data/heroes_stats_combined.txt","w") as f:
        f.write(str(new_stats))
    
def get_hero_benchmark(refresh=False):
    with open("hero_data/heroes_stats_combined.txt","r") as f:
        heroes_stats_combined = eval(f.readline())
                
    if refresh:
        api_instance = od_python.BenchmarksApi()
        benchmarks = []
        for hero in heroes_stats_combined:
            api_response = api_instance.benchmarks_get(hero['id'])
            benchmarks.append(api_response)
            time.sleep(1)
        with open("hero_data/benchmarks.txt","w") as f:
            f.write(str(benchmarks).replace('\t',"").replace('\n',""))
            
    with open("hero_data/benchmarks.txt","r") as f:
        benchmarks = eval(f.readline())
        
    for i in range(len(benchmarks)):
        assert(benchmarks[i]['hero_id'] == heroes_stats_combined[i]['id'])
        result = benchmarks[i]['result']
        heroes_stats_combined[i]['gpm'] = (result['gold_per_min'][7]['value'] + result['gold_per_min'][8]['value'] + result['gold_per_min'][9]['value'] + result['gold_per_min'][10]['value'])/4
        heroes_stats_combined[i]['xpm'] = (result['xp_per_min'][7]['value'] + result['xp_per_min'][8]['value'] + result['xp_per_min'][9]['value'] + result['xp_per_min'][10]['value'])/4
        heroes_stats_combined[i]['kpm'] = (result['kills_per_min'][7]['value'] + result['kills_per_min'][8]['value'] + result['kills_per_min'][9]['value'] + result['kills_per_min'][10]['value'])/4
        heroes_stats_combined[i]['lhpm'] = (result['last_hits_per_min'][7]['value'] + result['last_hits_per_min'][8]['value'] + result['last_hits_per_min'][9]['value'] + result['last_hits_per_min'][10]['value'])/4
        heroes_stats_combined[i]['hdpm'] = (result['hero_damage_per_min'][7]['value'] + result['hero_damage_per_min'][8]['value'] + result['hero_damage_per_min'][9]['value'] + result['hero_damage_per_min'][10]['value'])/4
        heroes_stats_combined[i]['hhpm'] = (result['hero_healing_per_min'][7]['value'] + result['hero_healing_per_min'][8]['value'] + result['hero_healing_per_min'][9]['value'] + result['hero_healing_per_min'][10]['value'])/4
        heroes_stats_combined[i]['td'] = (result['tower_damage'][7]['value'] + result['tower_damage'][8]['value'] + result['tower_damage'][9]['value'] + result['tower_damage'][10]['value'])/4
    
    with open("hero_data/heroes_stats_combined.txt","w") as f:
        f.write(str(heroes_stats_combined))        
    
    
        
def get_average_benchmarks():
    with open("hero_data/heroes_stats_combined.txt","r") as f:
        heroes_stats_combined = eval(f.readline())
    gpm = xpm = kpm = lhpm = hdpm= hhpm= td=  0
    for hero in heroes_stats_combined:
        gpm += hero['gpm']
        xpm += hero['xpm']
        kpm += hero['kpm']
        lhpm += hero['lhpm']
        hdpm += hero['hdpm']
        hhpm += hero['hhpm']
        td += hero['td']
        
    gpm /= 115
    xpm /= 115
    kpm /= 115    
    lhpm /= 115
    hdpm /= 115
    hhpm /= 115
    td /= 115

    return gpm, xpm, kpm, lhpm, hdpm, hhpm, td

def get_win_rate(hero):
    wins = hero['_5_win'] + hero['_6_win'] + hero['_7_win']
    loss = hero['_5_lose'] + hero['_6_lose'] + hero['_7_lose']
    return (wins/(wins/loss))

def create_features(heroes):
    gpm, xpm, kpm, lhpm, hdpm, hhpm, td = get_average_benchmarks()
    with open("hero_data/heroes_stats_combined.txt","r") as f:
        heroes_stats_combined = eval(f.readline())
    with open("hero_data/hero_positions.txt","r") as f:
        positions = eval(f.readline())
    roles= []
    # for hero in heroes_stats_combined:
    #     roles += hero['roles']
    # roles = list(set(roles))
    # print(roles)
    roles = ['Escape', 'Initiator', 'Pusher', 'Durable', 'Nuker', 'Carry', 'Disabler', 'Jungler', 'Support'] 
    p1 = heroes_stats_combined[positions[heroes[0]]]
    p2 = heroes_stats_combined[positions[heroes[1]]]
    p3 = heroes_stats_combined[positions[heroes[2]]]
    p4 = heroes_stats_combined[positions[heroes[3]]]
    p5 = heroes_stats_combined[positions[heroes[4]]]
    p6 = heroes_stats_combined[positions[heroes[5]]]
    t1 = [p1,p2,p3,p4,p5]
    p7 = heroes_stats_combined[positions[heroes[6]]]
    p8 = heroes_stats_combined[positions[heroes[7]]]
    p9 = heroes_stats_combined[positions[heroes[8]]]
    p10 = heroes_stats_combined[positions[heroes[9]]]
    t2 = [p6,p7,p8,p9,p10]
    
    p1_wr = get_win_rate(p1)
    p2_wr = get_win_rate(p2)
    p3_wr = get_win_rate(p3)
    p4_wr = get_win_rate(p4)
    p5_wr = get_win_rate(p5)
    p6_wr = get_win_rate(p6)
    p7_wr = get_win_rate(p7)
    p8_wr = get_win_rate(p8)
    p9_wr = get_win_rate(p9)
    p10_wr = get_win_rate(p10)
    
    
    t1gpm = (p1['gpm']+p2['gpm']+p3['gpm']+p4['gpm']+p5['gpm'])/(5*gpm)
    t1xpm = (p1['xpm']+p2['xpm']+p3['xpm']+p4['xpm']+p5['xpm'])/(5*xpm)
    t1kpm = (p1['kpm']+p2['kpm']+p3['kpm']+p4['kpm']+p5['kpm'])/(5*kpm)
    t1lhpm = (p1['lhpm']+p2['lhpm']+p3['lhpm']+p4['lhpm']+p5['lhpm'])/(5*lhpm)
    t1hdpm = (p1['hdpm']+p2['hdpm']+p3['hdpm']+p4['hdpm']+p5['hdpm'])/(5*hdpm)
    t1hhpm = (p1['hhpm']+p2['hhpm']+p3['hhpm']+p4['hhpm']+p5['hhpm'])/(5*hhpm)
    t1td = (p1['td']+p2['td']+p3['td']+p4['td']+p5['td'])/(5*td)
    t1_complexity = (p1['complexity']+p2['complexity']+p3['complexity']+p4['complexity']+p5['complexity'])
    t1_roles = [0,0,0,0,0,0,0,0,0]
    for i in range(len(roles)):
        for hero in t1:
            if roles[i] in hero['roles']:
                t1_roles[i] += 1
    t1_escape = t1_roles[0]
    t1_initiator = t1_roles[1]
    t1_pusher = t1_roles[2]
    t1_durable = t1_roles[3]
    t1_nuker = t1_roles[4]
    t1_carry = t1_roles[5]
    t1_disable = t1_roles[6]
    t1_jungler = t1_roles[7]
    t1_support  = t1_roles[8]
    
    t2gpm = (p6['gpm']+p7['gpm']+p8['gpm']+p9['gpm']+p10['gpm'])/(5*gpm)
    t2xpm = (p6['xpm']+p7['xpm']+p8['xpm']+p9['xpm']+p10['xpm'])/(5*xpm)
    t2kpm = (p6['kpm']+p7['kpm']+p8['kpm']+p9['kpm']+p10['kpm'])/(5*kpm)
    t2lhpm = (p6['lhpm']+p7['lhpm']+p8['lhpm']+p9['lhpm']+p10['lhpm'])/(5*lhpm)
    t2hdpm = (p6['hdpm']+p7['hdpm']+p8['hdpm']+p9['hdpm']+p10['hdpm'])/(5*hdpm)
    t2hhpm = (p6['hhpm']+p7['hhpm']+p8['hhpm']+p9['hhpm']+p10['hhpm'])/(5*hhpm)
    t2td = (p6['td']+p7['td']+p8['td']+p9['td']+p10['td'])/(5*td)
    t2_complexity = (p6['complexity']+p7['complexity']+p8['complexity']+p9['complexity']+p10['complexity'])
    t2_roles = [0,0,0,0,0,0,0,0,0]
    for i in range(len(roles)):
        for hero in t2:
            if roles[i] in hero['roles']:
                t2_roles[i] += 1
    t2_escape = t2_roles[0]
    t2_initiator = t2_roles[1]
    t2_pusher = t2_roles[2]
    t2_durable = t2_roles[3]
    t2_nuker = t2_roles[4]
    t2_carry = t2_roles[5]
    t2_disable = t2_roles[6]
    t2_jungler = t2_roles[7]
    t2_support  = t2_roles[8]
    
    return t1gpm, t1xpm, t1kpm, t1lhpm, t1hdpm, t1hhpm, t1td, t2gpm, t2xpm, t2kpm, t2lhpm, t2hdpm, t2hhpm, t2td, t1_complexity, t2_complexity, t1_escape, t1_initiator, t1_pusher, t1_durable, t1_nuker, t1_carry, t1_disable, t1_jungler, t1_support, t2_escape, t2_initiator, t2_pusher, t2_durable, t2_nuker, t2_carry, t2_disable, t2_jungler, t2_support, p1_wr, p2_wr, p3_wr, p4_wr, p5_wr, p6_wr, p7_wr, p8_wr, p9_wr, p10_wr

def save_csv():
    with open("matches/matches_ap.txt","r") as f:
        matches = eval(f.readline())
        
    matches = pd.DataFrame.from_dict(matches)
    matches.to_csv("matches/matches.csv")

def prepare_data(data):
    y_data = data['radiant_win']
    y_data = y_data.values
    x_data = data.drop(['duration', 'loss', 'match_id', 'match_seq_num', 'patch','radiant_win','region','skill','start_time','throw','radiant_team','dire_team'],1)
    print(x_data.columns.values)
    x_data = (x_data-x_data.mean())/x_data.std()
    assert(not x_data.isnull().values.any())
    x_data, x_test, y_data, y_test = train_test_split(x_data, y_data,
                                                    test_size = int(0.1*x_data.shape[0]),
                                                    random_state = 2,
                                                    stratify = y_data)
    return x_data, y_data, x_test, y_test
    
def train_predict(clf, x_data, y_data, x_test, y_test):

    print("\n~~~ " + clf.__class__.__name__ + " ~~~~")
    print("=== Train ===")
    train_classifier(clf, x_data, y_data)
    #train
    f1, acc = predict_outcome(clf, x_data, y_data)
    #test
    print("\n=== Test ===")
    f1, acc = predict_outcome(clf, x_test, y_test)
    
def train_classifier(clf, x_data, y_data):

    start = time.time()
    clf.fit(x_data, y_data)
    end = time.time()

    print("Trained model in " + str(end-start) + " seconds")

def predict_outcome(clf, features, target):

    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print("Predictions made in " + str(end-start) + " seconds")
    f1score = f1_score(target, y_pred, pos_label=True)
    acc = sum(target == y_pred)/float(len(y_pred))
    print("F1 score and accuracy score: " + str(f1score) + ", " + str(acc))
    
    return f1score, acc

def save_clf(clf):
    with open( clf.__class__.__name__ + '.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    print(clf.__class__.__name__ + " model saved!")

def create_one_hot_hero_features():
    with open("matches/matches_ap.txt","r") as f:
        matches = eval(f.readline())
    with open("hero_data/heroes.txt","r") as f:
        heroes = eval(f.readline())
    hero_ids = [x['id'] for x in heroes]
    col = []
    for name in hero_ids:
        col.append("rad_"+str(name))
        col.append("dire_"+str(name))
    length = len(col)
    for match in matches:
        original_length = len(match.keys())
        heroes = list(map(int, match['radiant_team'].split(',') + match['dire_team'].split(',')))
        for name in col:
            match[name] = 0
        for i in range(5):
            match["rad_"+str(heroes[i])] = 1
        for i in range(5,10):
            match["dire_"+str(heroes[i])] = 1
        assert(len(match) == original_length + length)
    
    matches = pd.DataFrame.from_dict(matches)
    matches.to_csv("matches/matches.csv")
    
    
def xgb():
    params = {
    'eta': 0.02, 
    'max_depth': 7,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 100,
    'silent': True
    }
    # save_csv()
    matches = pd.read_csv("matches/matches.csv")
    x_data, y_data, x_test, y_test = prepare_data(matches)
    clfa = XGBClassifier()
    train_predict(clfa, x_data, y_data, x_test, y_test)
    save_clf(clfa)

def mlp():
    # save_csv()
    matches = pd.read_csv("matches/matches.csv")
    x_data, y_data, x_test, y_test = prepare_data(matches)
    clfa = MLPClassifier(solver = 'adam', alpha = 0.005, hidden_layer_sizes=(260, 130, 65, 30, 10), random_state=1, warm_start=True)
    train_predict(clfa, x_data, y_data, x_test, y_test)
    save_clf(clfa)
    
    
def features_check():
    with open("matches/matches_ap.txt","r") as f:
        matches = eval(f.readline())
    for match in matches:
        if 'p1_wr' in match.keys():
            continue
        print(match)
        t1gpm, t1xpm, t1kpm, t1lhpm, t1hdpm, t1hhpm, t1td, t2gpm, t2xpm, t2kpm, t2lhpm, t2hdpm, t2hhpm, t2td, t1_complexity, t2_complexity, t1_escape, t1_initiator, t1_pusher, t1_durable, t1_nuker, t1_carry, t1_disable, t1_jungler, t1_support, t2_escape, t2_initiator, t2_pusher, t2_durable, t2_nuker, t2_carry, t2_disable, t2_jungler, t2_support,p1_wr, p2_wr, p3_wr, p4_wr, p5_wr, p6_wr, p7_wr, p8_wr, p9_wr, p10_wr = create_features(list(map(int, match['radiant_team'].split(',') + match['dire_team'].split(','))))
        roles = ['Escape', 'Initiator', 'Pusher', 'Durable', 'Nuker', 'Carry', 'Disabler', 'Jungler', 'Support']
        match['t1gpm'] = t1gpm
        match['t1xpm'] = t1xpm
        match['t1kpm'] = t1kpm
        match['t1lhpm'] = t1lhpm
        match['t1hdpm'] = t1hdpm
        match['t1hhpm'] = t1hhpm
        match['t1td'] = t1td
        match['t1_complexity'] = t1_complexity
        match['t1_escape'] = t1_escape
        match['t1_initiator'] = t1_initiator
        match['t1_pusher'] = t1_pusher
        match['t1_durable'] = t1_durable
        match['t1_nuker'] = t1_nuker
        match['t1_carry'] = t1_carry
        match['t1_disable'] = t1_disable
        match['t1_jungler'] = t1_jungler
        match['t1_support'] = t1_support
        
        match['t2gpm'] = t2gpm
        match['t2xpm'] = t2xpm
        match['t2kpm'] = t2kpm
        match['t2lhpm'] = t2lhpm
        match['t2hdpm'] = t2hdpm
        match['t2hhpm'] = t2hhpm
        match['t2td'] = t2td
        match['t2_complexity'] = t2_complexity                
        match['t2_escape'] = t2_escape
        match['t2_initiator'] = t2_initiator
        match['t2_pusher'] = t2_pusher
        match['t2_durable'] = t2_durable
        match['t2_nuker'] = t2_nuker
        match['t2_carry'] = t2_carry
        match['t2_disable'] = t2_disable
        match['t2_jungler'] = t2_jungler
        match['t2_support'] = t2_support
        
        match['p1_wr'] = p1_wr
        match['p2_wr'] = p2_wr
        match['p3_wr'] = p3_wr
        match['p4_wr'] = p4_wr
        match['p5_wr'] = p5_wr
        match['p6_wr'] = p6_wr
        match['p7_wr'] = p7_wr
        match['p8_wr'] = p8_wr
        match['p9_wr'] = p9_wr
        match['p10_wr'] = p10_wr
        
    with open("matches/matches_ap.txt","w") as f:
        f.write(str(matches))   
try:
    # save_csv()
    # features_check()
    # create_features([1,2,3,4,5,6,7,8,9,10])
    if int(sys.argv[1]) == 1:
        get_matches()
    else:
        mlp()
        xgb()
    

    
except ApiException as e:
    print("Exception when calling BenchmarksApi->benchmarks_get: %s\n" % e)