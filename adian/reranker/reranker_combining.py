import jsonlines
import json
import time
from copy import deepcopy

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

def system_combination(path_1, path_2, output_path='test.jsonl'):
    results_1 = load_jsonl(path_1)
    results_2 = load_jsonl(path_2)
    
    outputs = [_combine_scores(ex_1, ex_2) for ex_1, ex_2 in zip(results_1, results_2)]
    save_jsonl(outputs, output_path)
 
def _combine_scores(ex_1, ex_2):
    assert ex_1['q_id'] == ex_2['q_id'], "mismathced outputs"
    q_id = ex_1['q_id']

    results = []

    results_1, results_2 = ex_1['results'], ex_2['results']
 
    id_to_index = {i:k for k, i in enumerate([doc['result_id'] for doc in results_2])}
    for doc_1 in results_1:
        if doc_1['result_id'] in id_to_index:
            index = id_to_index[doc_1['result_id']]
            doc_2 = results_2[index]
            new_doc = deepcopy(doc_1)
            new_doc['score'] = doc_1['score'] * doc_2['score']
            results.append(new_doc)

    assert len(results) > 990, "too many mismatches"
    return {'q_id':q_id, 'results':results}
        
system_combination('../../outputs/rerank_monot5/trec_2021_baseline.jsonl', 
                   '../../outputs/bm25/trec_2021_baseline.jsonl',
                   'test')
