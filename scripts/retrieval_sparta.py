from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseSearch

import logging
import pathlib, os
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

    # dataset = "scifact"

    # #### Download scifact dataset and unzip the dataset
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    # data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data path where scifact has been downloaded and unzipped to the data loader
    # data folder would contain these files: 
    # (1) scifact/corpus.jsonl  (format: jsonlines)
    # (2) scifact/queries.jsonl (format: jsonlines)
    # (3) scifact/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_folder=args.data_path).load(split="test")

    #### Sparse Retrieval using SPARTA #### 
    model_path = "BeIR/sparta-msmarco-distilbert-base-v1"
    sparse_model = SparseSearch(models.SPARTA(model_path), batch_size=128)
    retriever = EvaluateRetrieval(sparse_model, k_values=[1,3,5,10,100,500,1000])

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Sparta Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    #### Save retrieval results ####
    fout = open(args.output_path, 'w', encoding='utf-8')
    for query_id in results.keys():
        result_dict = dict()
        result_dict["q_id"] = query_id
        result_dict["results"] = []

        scores_dict = results[query_id]
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

    # #### Print top-k documents retrieved ####
    # top_k = 10

    # query_id, ranking_scores = random.choice(list(results.items()))
    # scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    # logging.info("Query : %s\n" % queries[query_id])

    # for rank in range(top_k):
    #     doc_id = scores_sorted[rank][0]
    #     # Format: Rank x: ID [Title] Body
    #     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))