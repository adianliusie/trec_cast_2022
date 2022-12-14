from beir import util, LoggingHandler
# from beir.datasets.data_loader import GenericDataLoader
from src.utils.beir_data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import MonoT5
from beir.reranking import Rerank

import pathlib, os
import logging
import random
import json
import argparse

from src.utils.general import save_script_args, save_retrieval_results, check_output_path

if __name__ == '__main__':
    save_script_args()

    parser = argparse.ArgumentParser(description='retrieval documents using monot5 re-ranker.')
    parser.add_argument('--data_path',  help='the directory for the input files')
    parser.add_argument('--retrieval_result_path',  help='the directory for the results file after retrieval')
    parser.add_argument('--output_path',  help='Output file, retrievaled documents and corresponding scores for each query.')
    parser.add_argument('--top_k', type=int, default=1000, help='Rerank top_k results from the retrieval_results.')
    parser.add_argument('--model_name', type=str, default='castorini/monot5-base-msmarco', help='The model to do the rerank.')
    
    args = parser.parse_args()
    check_output_path(args.output_path)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    # the loaded corpus = {}
    corpus, queries, qrels = GenericDataLoader(args.data_path).load(split="test")

    #### Restore the results from BM25 #####
    logging.info("Loading retrieval results from {}".format(args.retrieval_result_path))
    results = dict()
    with open(args.retrieval_result_path, 'r') as json_file:
        for line in json_file:
            doc_dict = dict()
            bm25_result_line = json.loads(line)
            query_id = bm25_result_line["q_id"]
            # results["q_id"] = query_id
            results_list = bm25_result_line["results"]
            for doc in results_list:
                doc_id = doc["result_id"]
                score = doc["score"]
                doc_dict[doc_id] = score
                
                # mq227: load the corpus from the retrieval results
                doc_text = doc["text"]
                if doc_id in corpus and corpus[doc_id].get("text") != doc_text:
                    print("Multiple versions of text for {}!".format(doc_id))
                else:
                    corpus[doc_id] = {"text": doc_text, "title": ""}

            results[query_id] = doc_dict
    

    ##############################################
    #### (2) RERANK docs using MonoT5 ####
    ##############################################

    #### Reranking using MonoT5 model #####
    # Document Ranking with a Pretrained Sequence-to-Sequence Model 
    # https://aclanthology.org/2020.findings-emnlp.63/

    #### Check below for reference parameters for different MonoT5 models 
    #### Two tokens: token_false, token_true
    # 1. 'castorini/monot5-base-msmarco':             ['???false', '???true']
    # 2. 'castorini/monot5-base-msmarco-10k':         ['???false', '???true']
    # 3. 'castorini/monot5-large-msmarco':            ['???false', '???true']
    # 4. 'castorini/monot5-large-msmarco-10k':        ['???false', '???true']
    # 5. 'castorini/monot5-base-med-msmarco':         ['???false', '???true']
    # 6. 'castorini/monot5-3b-med-msmarco':           ['???false', '???true']
    # 7. 'unicamp-dl/mt5-base-en-msmarco':            ['???no'   , '???yes']
    # 8. 'unicamp-dl/ptt5-base-pt-msmarco-10k-v2':    ['???n??o'  , '???sim']
    # 9. 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2':   ['???n??o'  , '???sim']
    # 10.'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2':['???n??o'  , '???sim']
    # 11.'unicamp-dl/mt5-base-en-pt-msmarco-v2':      ['???no'   , '???yes']
    # 12.'unicamp-dl/mt5-base-mmarco-v2':             ['???no'   , '???yes']
    # 13.'unicamp-dl/mt5-base-en-pt-msmarco-v1':      ['???no'   , '???yes']
    # 14.'unicamp-dl/mt5-base-mmarco-v1':             ['???no'   , '???yes']
    # 15.'unicamp-dl/ptt5-base-pt-msmarco-10k-v1':    ['???n??o'  , '???sim']
    # 16.'unicamp-dl/ptt5-base-pt-msmarco-100k-v1':   ['???n??o'  , '???sim']
    # 17.'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['???n??o'  , '???sim']

    logging.info("Start reranking ...")
    logging.info(f"model name: {args.model_name}")
    cross_encoder_model = MonoT5(args.model_name, token_false='???false', token_true='???true')
    reranker = Rerank(cross_encoder_model, batch_size=128)

    # # Rerank top-100 results using the reranker provided
    rerank_results = reranker.rerank(corpus, queries, results, top_k=args.top_k)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    k_values = [1,3,5,10,100,500,1000]
    logging.info("Re-ranker evaluation for k in: {}".format(k_values))
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, k_values)

    #### Save retrieval results ####
    save_retrieval_results(args.output_path, rerank_results, corpus)

