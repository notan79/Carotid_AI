import os
import json



def save_json(save_dict, path = "./json/best_dict.json"):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as outfile: 
        json.dump(save_dict, outfile)
        
def load_json(path = "./json/best_dict.json"):
    if not os.path.exists(path):
        print('No file found')
    else:
        with open(path) as json_file:
            return json.load(json_file)
            
    
    