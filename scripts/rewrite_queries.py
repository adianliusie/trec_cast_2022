import argparse

from src.modules.query_rewriter import QueryRewriter
from src.utils.general import save_script_args

if __name__ == '__main__':
    save_script_args()
    
    parser = argparse.ArgumentParser(description='rewrites standalone queries for a data set')
    parser.add_argument('--data_name',  default='trec_2021', help='data set to generate rewritten queries for')
    parser.add_argument('--output_path', help='output dir for jsonl')
    
    args = parser.parse_args()
    rewriter = QueryRewriter()
    rewriter.rewrite_queries(args.data_name, args.output_path)
    
