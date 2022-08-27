from .general import load_jsonl
import os, sys, csv

def convert_jsonl_to_results(path):
    """converts output file into format required to use TREC evaluation tool, and saved in temp dir"""    
    all_q_docs = load_jsonl(path)
    output_path = path.replace('../', '')
    print(output_path)
    output_path = output_path.replace('/', '_')
    print(output_path)

    if not os.path.exists("results"):
        os.path.makedirs("results")

    with open(f'results/{output_path}', 'w+') as f:
        for q_docs in all_q_docs:
            q_id = q_docs['q_id']
            for k, doc in enumerate(q_docs['results']):
                doc_id = doc['result_id']
                doc_score = 2 #doc['score']
                f.write(f'{q_id} Q0 {doc_id} {k} {doc_score} STANDARD\n')
    return output_path

def convert_jsonl_to_q_rel(path):
    queries = load_jsonl(path)
    output_path = path.replace('/', '_')
    
    with open(f'results/{output_path}', 'w+') as f:
        for query in queries:
            q_id = query['q_id']
            
            pass
            #confused as in given examples, there seems to be multiple choices?

def convert_jsonl_to_beir_results(path, top_k=None):
    """ convert bm25/reranker output jsonl file into the results format for evaluation """
    queries = load_jsonl(path)
    out_results = dict()
    for query in queries:
        q_id = query['q_id']
        q_results = query['results']
        out_results[q_id] = dict()
        if top_k:
            q_results = q_results[:top_k]
        for q_result in q_results:
            doc_id = q_result["result_id"]
            score = q_result["score"]
            out_results[q_id][doc_id]  = score

    return out_results

def load_q_rel_tsv(path, threshold=1):
    """ Load the q_rel file in the beir format """
    if '.tsv' not in path:
        sys.exit(f"{qrel} not in tsv format")

    reader = csv.reader(open(path, encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    
    qrels = dict()
    for id, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        if score >= threshold:
            score = 1
        else:
            score = 0
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    return qrels

