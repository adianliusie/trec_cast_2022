import argparse

from src.modules.query_rewriter import QueryRewriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rewrites standalone queries for a data set')
    parser.add_argument('--data_name',  default='trec_2021', help='data set to generate rewritten queries for')
    parser.add_argument('--output_dir', default='../outputs/queries/trec_2021_baseline.jsonl', help='output dir for jsonl')
    
    args = parser.parse_args()
    rewriter = QueryRewriter()
    rewriter.rewrite_queries(args.data_name, args.output_dir)
    
