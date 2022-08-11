import argparse
import os

from src.utils.eval import convert_jsonl_to_results
from src.utils.general import save_script_args

if __name__ == '__main__':    
    save_script_args()
    
    parser = argparse.ArgumentParser(description='rewrites standalone queries for a data set')
    parser.add_argument('--predictions', help='document ordering (either result file, or jsonl file')
    parser.add_argument('--references',  help='relevant documents for each query (either q_rel or jsonl file')
    
    args = parser.parse_args()
    
    if '.jsonl' in args.predictions: 
        args.predictions = convert_jsonl_to_results(args.predictions)
    
    os.system(f'./trec_eval/trec_eval {args.references} results/{args.predictions}')
    