import argparse
import os

from src.utils.eval import convert_jsonl_to_results

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='rewrites standalone queries for a data set')
    parser.add_argument('--predictions', help='document ordering (either result file, or jsonl file')
    parser.add_argument('--references',  help='relevant documents for each query (either q_rel or jsonl file')
    
    args = parser.parse_args()
    
    if '.jsonl' in args.predictions: 
        args.predictions = convert_jsonl_to_results(args.predictions)
    
    os.system(f'bash trec_eval {args.references} {args.predictions}')
    