import json
import jsonlines
import sys
from os.path import abspath
import os

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

def save_retrieval_results(output_path, results, corpus):
    fout = open(output_path, 'w', encoding='utf-8')
    for query_id in results.keys():
        result_dict = dict()
        result_dict["q_id"] = query_id
        result_dict["results"] = []

        scores_dict = results[query_id]
        scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
        for i in range(len(scores)):
            doc_dict = dict()
            id, score = scores[i]
            doc_dict["result_id"] = id
            doc_dict["text"] = corpus[id].get("text")
            doc_dict["score"] = score
            result_dict["results"].append(doc_dict)
        fout.write(json.dumps(result_dict) + '\n')
    
    fout.close()

def check_output_path(output_path):
    if os.path.exists(output_path):
        exit(f"{output_path} already exists!")

    outdir = os.path.dirname(output_path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)