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
        with open("matches/matches.txt","r") as f:
            matches = eval(f.readline())
        f.close()
        api_response = api_instance.public_matches_get()
        
        for i in range(len(api_response)):
            matches.append({"match_id":api_response[i].match_id, "match_seq_num":api_response[i].match_seq_num, "radiant_win":api_response[i].radiant_win, "start_time":api_response[i].start_time, "duration":api_response[i].duration, "radiant_team":api_response[i].radiant_team, "dire_team":api_response[i].dire_team})
        matches = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in matches)]
        print(len(matches))
        with open("matches/matches.txt","w") as f:
            f.write(str(matches))
        f.close()
        time.sleep(20)
        
try:
    api_instance = od_python.MatchesApi()
    with open("matches/matches.txt","r") as f:
        matches = eval(f.readline())
    matches_all_pick = []
    abandon = False
    for match in matches:
        api_response = api_instance.matches_match_id_get(match['match_id'])
        if (api_response.game_mode == 1):
            for i in range(10):
                if (api_response.players[i].abandons == 0):
                    abandon = True
            if not abandon:  
                match["throw"] = api_response.throw
                match["loss"] = api_response.loss
                match['patch'] = api_response.patch
                match['region'] = api_response.region
                match['skill'] = api_response.skill
                match['leaver_status'] = api_response.patch
                match['patch'] = api_response.patch
        matches_all_pick.append(matches)
        time.sleep(1.5)
    with open("matches/matches_ap.txt","w") as f:
        f.write(str(matches))
        
    # api_response = api_instance.matches_match_id_get(3997196903)
    # print(api_response)
    
        
        

    
except ApiException as e:
    print("Exception when calling BenchmarksApi->benchmarks_get: %s\n" % e)