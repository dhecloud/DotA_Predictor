import time
import od_python
from od_python.rest import ApiException
# create an instance of the API class
hero_id = 'hero_id_example' # str | Hero ID


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
        time.sleep(400)
        get_ap_matches()
        time.sleep(400)

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
                # print(match['throw'])
                # print(type(match['throw']))
                # print(match['loss'])
                # print(type(match['loss']))
                matches_all_pick.append(match)
                with open("matches/matches_ap.txt","w") as f:
                    f.write(str(matches_all_pick))
        time.sleep(2)
    with open("matches/matches_ap.txt","w") as f:
        f.write(str(matches))

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
    gpm = xpm = kpm = lhpm = hdpm= hhpm= td= 0
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
        
def create_features(heroes):
    gpm, xpm, kpm, lhpm, hdpm, hhpm, td = get_average_benchmarks()
    with open("hero_data/heroes_stats_combined.txt","r") as f:
        heroes_stats_combined = eval(f.readline())
    with open("hero_data/hero_positions.txt","r") as f:
        positions = eval(f.readline())
        
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
    
    t1gpm = (p1['gpm']+p2['gpm']+p3['gpm']+p4['gpm']+p5['gpm'])/(5*gpm)
    t1xpm = (p1['xpm']+p2['xpm']+p3['xpm']+p4['xpm']+p5['xpm'])/(5*xpm)
    t1kpm = (p1['kpm']+p2['kpm']+p3['kpm']+p4['kpm']+p5['kpm'])/(5*kpm)
    t1lhpm = (p1['lhpm']+p2['lhpm']+p3['lhpm']+p4['lhpm']+p5['lhpm'])/(5*lhpm)
    t1hdpm = (p1['hdpm']+p2['hdpm']+p3['hdpm']+p4['hdpm']+p5['hdpm'])/(5*hdpm)
    t1hhpm = (p1['hhpm']+p2['hhpm']+p3['hhpm']+p4['hhpm']+p5['hhpm'])/(5*hhpm)
    t1td = (p1['td']+p2['td']+p3['td']+p4['td']+p5['td'])/(5*td)
    t1_stats = [t1gpm, t1xpm, t1kpm, t1lhpm, t1hdpm, t1hhpm, t1td]
    
    t2gpm = (p6['gpm']+p7['gpm']+p8['gpm']+p9['gpm']+p10['gpm'])/(5*gpm)
    t2xpm = (p6['xpm']+p7['xpm']+p8['xpm']+p9['xpm']+p10['xpm'])/(5*xpm)
    t2kpm = (p6['kpm']+p7['kpm']+p8['kpm']+p9['kpm']+p10['kpm'])/(5*kpm)
    t2lhpm = (p6['lhpm']+p7['lhpm']+p8['lhpm']+p9['lhpm']+p10['lhpm'])/(5*lhpm)
    t2hdpm = (p6['hdpm']+p7['hdpm']+p8['hdpm']+p9['hdpm']+p10['hdpm'])/(5*hdpm)
    t2hhpm = (p6['hhpm']+p7['hhpm']+p8['hhpm']+p9['hhpm']+p10['hhpm'])/(5*hhpm)
    t2td = (p6['td']+p7['td']+p8['td']+p9['td']+p10['td'])/(5*td)
    t2_stats = [t2gpm, t2xpm, t2kpm, t2lhpm, t2hdpm, t2hhpm, t2td]
    
    return t1_stats, t2_stats

try:
    get_matches()
    # create_features([1, 73, 10, 74, 11, 5, 87, 86, 84, 83])
    

    
except ApiException as e:
    print("Exception when calling BenchmarksApi->benchmarks_get: %s\n" % e)