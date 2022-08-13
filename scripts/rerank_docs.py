import argparse

from src.modules.reranker import SbertReranker, PassageReranker
from src.utils.general import save_script_args

if __name__ == '__main__':
    save_script_args()

    parser = argparse.ArgumentParser(description='reranks documents in terms of relevants for a given query')
    parser.add_argument('--system',  default='passage', help='which reranker to use for reranking')
    parser.add_argument('--query_path',  help='input queries (needs q_id and text)')
    parser.add_argument('--docs_path',   help='input queries (needs doc_ids and text)')
    parser.add_argument('--output_path', help='input queries (needs ids and text)')
    parser.add_argument('--device',     default=None, help='if specific device ordinal is to be specified')

    args = parser.parse_args()
    
    if args.system == 'sbert':
        reranker = SbertReranker(args.device)
    elif args.system == 'passage':
        reranker = PassageReranker(args.device)
    
    reranker.rerank(args.query_path, args.docs_path, args.output_path)
