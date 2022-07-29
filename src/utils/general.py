import json
import jsonlines

def save_jsonl(data:list, path:str):
    if '.jsonl' not in path:
        path = path + '.jsonl'
        
    with jsonlines.open(path, 'w') as writer:
        writer.write_all(data)
        
def load_jsonl(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    return [json.loads(json_str) for json_str in json_list]

def load_json(path:str)->dict:
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data

def flatten(x):
    return [i for j in x for i in j]
