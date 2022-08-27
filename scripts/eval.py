import argparse
import os
import pytrec_eval
import logging
from typing import Dict, List, Tuple

from src.utils.eval import convert_jsonl_to_beir_results, load_q_rel_tsv
from src.utils.general import save_script_args

def evaluate(qrels: Dict[str, Dict[str, int]], 
             results: Dict[str, Dict[str, float]], 
             k_values: List[int],
             ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        
    if ignore_identical_ids:
        logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
        
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
        
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)
        
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        
    for eval in [ndcg, _map, recall, precision]:
        logging.info("\n")
        for k in eval.keys():
            logging.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision

if __name__ == '__main__':    
    save_script_args()
    
    parser = argparse.ArgumentParser(description='rewrites standalone queries for a data set')
    parser.add_argument('--predictions', help='document ordering (either result file, or jsonl file')
    parser.add_argument('--references', help='relevant documents for each query (either q_rel or jsonl file')
    parser.add_argument('--threshold', type=int, default=2, help='conver the relevant score to 1 if it is >= threshold, 2021:2, 2022:1')

    args = parser.parse_args()
    
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # parameters
    k_values = [1,3,5,10,100,500,1000,3000,5000,9000]

    # load referenecs
    logging.info("Loading reference: {}".format(args.references))
    qrels = load_q_rel_tsv(args.references, threshold=args.threshold)
    
    # load resutls
    logging.info("Loading predictions: {}".format(args.predictions))
    results = convert_jsonl_to_beir_results(args.predictions)

    # evaluate
    logging.info("Retriever evaluation for k in: {}".format(k_values))
    ndcg, _map, recall, precision = evaluate(qrels, results, k_values)
    