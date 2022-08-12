from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import MonoT5
from beir.reranking import Rerank

import pathlib, os
import logging
import random
import json
import argparse

from src.utils.general import save_script_args

if __name__ == '__main__':
    save_script_args()

    parser = argparse.ArgumentParser(description='retrieval documents using BM25 and monot5 re-ranker.')
    parser.add_argument('--data_path',  help='the directory for the input files')
    parser.add_argument('--output_path',  help='Output file, retrievaled documents and corresponding scores for each query.')

    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    # #### Download trec-covid.zip dataset and unzip the dataset
    # dataset = "trec-covid"
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    # data_path = util.download_and_unzip(url, out_dir)

    # #### Provide the data path where trec-covid has been downloaded and unzipped to the data loader
    # data folder would contain these files: 
    # (1) trec-covid/corpus.jsonl  (format: jsonlines)
    # (2) trec-covid/queries.jsonl (format: jsonlines)
    # (3) trec-covid/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(args.data_path).load(split="test")

    #########################################
    #### (1) RETRIEVE docs using BM25
    #########################################

    #### Provide parameters for Elasticsearch
    hostname = "localhost" #localhost
    index_name = "bm25_cast" # trec-covid
    initialize = True # False

    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model, k_values=[1,3,5,10,100,500,1000])

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    logging.info("BM25 evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    ##############################################
    #### (2) RERANK docs using MonoT5 ####
    ##############################################

    #### Reranking using MonoT5 model #####
    # Document Ranking with a Pretrained Sequence-to-Sequence Model 
    # https://aclanthology.org/2020.findings-emnlp.63/

    #### Check below for reference parameters for different MonoT5 models 
    #### Two tokens: token_false, token_true
    # 1. 'castorini/monot5-base-msmarco':             ['▁false', '▁true']
    # 2. 'castorini/monot5-base-msmarco-10k':         ['▁false', '▁true']
    # 3. 'castorini/monot5-large-msmarco':            ['▁false', '▁true']
    # 4. 'castorini/monot5-large-msmarco-10k':        ['▁false', '▁true']
    # 5. 'castorini/monot5-base-med-msmarco':         ['▁false', '▁true']
    # 6. 'castorini/monot5-3b-med-msmarco':           ['▁false', '▁true']
    # 7. 'unicamp-dl/mt5-base-en-msmarco':            ['▁no'   , '▁yes']
    # 8. 'unicamp-dl/ptt5-base-pt-msmarco-10k-v2':    ['▁não'  , '▁sim']
    # 9. 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2':   ['▁não'  , '▁sim']
    # 10.'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2':['▁não'  , '▁sim']
    # 11.'unicamp-dl/mt5-base-en-pt-msmarco-v2':      ['▁no'   , '▁yes']
    # 12.'unicamp-dl/mt5-base-mmarco-v2':             ['▁no'   , '▁yes']
    # 13.'unicamp-dl/mt5-base-en-pt-msmarco-v1':      ['▁no'   , '▁yes']
    # 14.'unicamp-dl/mt5-base-mmarco-v1':             ['▁no'   , '▁yes']
    # 15.'unicamp-dl/ptt5-base-pt-msmarco-10k-v1':    ['▁não'  , '▁sim']
    # 16.'unicamp-dl/ptt5-base-pt-msmarco-100k-v1':   ['▁não'  , '▁sim']
    # 17.'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['▁não'  , '▁sim']

    cross_encoder_model = MonoT5('castorini/monot5-base-msmarco', token_false='▁false', token_true='▁true')
    reranker = Rerank(cross_encoder_model, batch_size=128)

    # # Rerank top-100 results using the reranker provided
    rerank_results = reranker.rerank(corpus, queries, results, top_k=1000)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Re-ranker evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

    #### Save retrieval results ####
    fout = open(args.output_path, 'w', encoding='utf-8')
    for query_id in rerank_results.keys():
        result_dict = dict()
        result_dict["q_id"] = query_id
        result_dict["results"] = []

        scores_dict = rerank_results[query_id]
        scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
        for i in range(len(scores)):
            doc_dict = dict()
            id, score = scores[i]
            doc_dict["result_id"] = id
            doc_dict["text"] = corpus[id].get("text")
            doc_dict["score"] = score
            result_dict["results"].append(doc_dict)
        fout.write(json.dumps(result_dict) + '\n')
    
    fout.close()

    #### Print top-k documents retrieved ####
    # top_k = 10

    # query_id, ranking_scores = random.choice(list(rerank_results.items()))
    # scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    # logging.info("Query : %s\n" % queries[query_id])

    # for rank in range(top_k):
    #     doc_id = scores_sorted[rank][0]
    #     # Format: Rank x: ID [Title] Body
    #     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))