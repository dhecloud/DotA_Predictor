import time
import od_python
from od_python.rest import ApiException
# create an instance of the API class
api_instance = od_python.HeroesApi()
hero_id = 'hero_id_example' # str | Hero ID

try:
    heroes=[]
    # GET /benchmarks
    api_response = api_instance.heroes_get()
    for i in range(len(api_response)):
        hero = api_response[i]
        heroes.append({"id":hero.id, "name":hero.name, "localized_name":hero.localized_name, "primary_attr":hero.primary_attr, "attack_type":hero.attack_type, "roles":hero.roles})
    print(heroes[0])
    with open("heroes.txt","w") as f:
        f.write(str(heroes))
            
    
except ApiException as e:
    print("Exception when calling BenchmarksApi->benchmarks_get: %s\n" % e)