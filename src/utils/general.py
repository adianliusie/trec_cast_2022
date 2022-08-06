import json
import jsonlines
import sys
from os.path import abspath

def save_jsonl(data:list, path:str):
    if '.jsonl' not in path:
        path = path + '.jsonl'
        
    with jsonlines.open(path, 'x') as writer:
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

def save_script_args():
    CMD = f"python {' '.join(sys.argv)}\n"
    
    with open('CMDs', 'a+') as f:
        f.write(CMD)
        
def get_base_dir():
    """automatically gets root dir of framework"""
    
    #gets path of the src folder 
    cur_path = abspath(__file__)
    src_path = '/src'.join(cur_path.split('/src')[:-1])
    
    #can be called through a symbolic link, if so go out one more dir.
    if src_path.split('/')[-1] in ['eval', 'scripts']:
        base_path = '/'.join(src_path.split('/')[:-1])
    else:
        base_path = src_path
        
    return base_path
