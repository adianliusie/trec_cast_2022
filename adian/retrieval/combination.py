import jsonlines
import json
import time
import numpy as np
from tqdm import tqdm

def save_jsonl(data:list, path:str):
    if '.jsonl' not in path:
        path = path + '.jsonl'

    with jsonlines.open(path, 'x') as writer:
        writer.write_all(data)

def load_jsonl(path):
    print('loading docs')
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    return [json.loads(json_str) for json_str in json_list]

def load_doc_ids(path):
    print('loading doc ids')
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    output = []  
    for ex in [json.loads(json_str) for json_str in json_list]:
        q_id = ex['q_id']
        doc_ids = [doc['result_id'] for doc in ex['results']]
        output.append((q_id, doc_ids))
    return output
   
def doc_overlap(paths):
    results = [load_doc_ids(path) for path in paths]

    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            system_1 = results[i]
            system_2 = results[j]
            x = _overlap(system_1, system_2) 
            print(paths[i], paths[j], x)

def _overlap(system_1, system_2, N=500):
    """calculates (average) document overlap of N best for 2 systems"""
    results = []
    for query_1, query_2 in zip(system_1, system_2):
        q_id_1, doc_ids_1 = query_1
        q_id_2, doc_ids_2 = query_2

        assert(q_id_1 == q_id_2), "mismathced outputs"
        N_best_1 = doc_ids_1[:N]
        N_best_2 = doc_ids_2[:N]
        overlap = len([i for i in N_best_1 if i in N_best_2])
        results.append(overlap)
    return sum(results)/len(results)

def system_combination(path_1, path_2, output_path='test.jsonl'):
    results_1 = load_jsonl(path_1)
    results_2 = load_jsonl(path_2)
    
    outputs = [_combine_N_best(ex_1, ex_2) for ex_1, ex_2 in zip(results_1, results_2)]
    save_jsonl(outputs, output_path)
 
def _combine_N_best(ex_1, ex_2):
    assert ex_1['q_id'] == ex_2['q_id'], "mismathced outputs"
    q_id = ex_1['q_id']

    results = []
    docs_ids = []
    
    results_1, results_2 = ex_1['results'], ex_2['results']
    for k in range(len(results_1)):
        for doc in [results_1[k], results_2[k]]:
            if doc['result_id'] not in docs_ids:
               results.append(doc)
        
            if len(results) >= 1000:
                 return {'q_id':q_id, 'results':results}
        
system_combination('../outputs/bm25/trec_2021_baseline_ctx-5-3.jsonl', 
                   '../outputs/bm25/trec_2021_baseline_ctx-3-1.jsonl',
                   'ctx3-1_ctx5-3')
"""
doc_overlap(['../outputs/bm25/trec_2021_baseline.jsonl',
             '../outputs/bm25/trec_2021_baseline_ctx-3-1.jsonl', 
             '../outputs/bm25/trec_2021_baseline_ctx-5-3.jsonl'])
"""
