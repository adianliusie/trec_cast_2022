"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.

Modified from beir/examples/retrieval/evaluation/lexical/evaluate_bm25.py.
Mengjie Qian, 2022-08-02
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import pathlib, os, random
import logging
import json
import argparse

from src.utils.general import save_script_args, save_retrieval_results, check_output_path

if __name__ == '__main__':
    save_script_args()

    parser = argparse.ArgumentParser(description='retrieval documents using BM25.')
    parser.add_argument('--data_path',  help='the directory for the input files')
    parser.add_argument('--output_path',  help='Output file, retrievaled documents and corresponding scores for each query.')

    args = parser.parse_args()
    check_output_path(args.output_path)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #### load data
    # data folder would contain these files: 
    # (1) data_path/corpus.jsonl  (format: jsonlines, note: all the documents to retrieval from)
    # (2) data_path/queries.jsonl (format: jsonlines)
    # (3) data_path/qrels/test.tsv (format: tsv ("\t"), note: the ground truth)
    corpus, queries, qrels = GenericDataLoader(args.data_path).load(split="test")

    #### Lexical Retrieval using Bm25 (Elasticsearch) ####
    #### Provide a hostname (localhost) to connect to ES instance
    #### Define a new index name or use an already existing one.
    #### We use default ES settings for retrieval
    #### https://www.elastic.co/
    hostname = "localhost" #localhost
    index_name = "bm25_cast" # scifact
    timeout = 10000
    retry_on_timeout = True
    
    #### Intialize #### 
    # (1) True - Delete existing index and re-index all documents from scratch 
    # (2) False - Load existing index
    initialize = True # False

    #### Sharding ####
    # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
    # number_of_shards = 1
    # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)
    # (2) For datasets with big corpus ==> keep default configuration
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, timeout=timeout, retry_on_timeout=retry_on_timeout)
    retriever = EvaluateRetrieval(model, k_values=[1,3,5,10,100,500,1000,3000,5000,9000])

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)
    
    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    
    #### Save retrieval results ####
    save_retrieval_results(args.output_path, results, corpus)
