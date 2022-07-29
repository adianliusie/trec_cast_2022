import argparse

from src.modules.reranker import SentenceReranker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='reranks documents in terms of relevants for a given query')
    parser.add_argument('--system',  default='sbert_reanker', help='which reranker to use for reranking')
    parser.add_argument('--query_path',  help='input queries (needs q_id and text)')
    parser.add_argument('--docs_path',   help='input queries (needs doc_ids and text)')
    parser.add_argument('--output_path', help='input queries (needs ids and text)')

    args = parser.parse_args()
    
    if args.system == 'sbert_reanker':
        reranker = SentenceReranker()
    elif args.system == 't5_reranker':
        pass
    
    reranker.rerank(args.query_path, args.docs_path, args.output_path)