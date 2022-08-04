from .general import load_jsonl
import os

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