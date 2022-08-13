# !/use/bin/bash
# Take the output from the query rewriter, and prepare the files for bm25.
# Mengjie Qian, 2022-08-02
import json
import argparse
from src.utils.general import save_script_args
import os

# infile = "/home/mq227/rds/hpc-work/trec_cast_2022/outputs/queries/trec_2021_baseline_v2.jsonl"
# qrelfile = "/home/mq227/rds/hpc-work/beir/data/bm25_cast_test1/qrels/test.tsv"
# queryfile = "/home/mq227/rds/hpc-work/beir/data/bm25_cast_test1/queries.jsonl"

if __name__ == '__main__':
    save_script_args()

    parser = argparse.ArgumentParser(description='retrieval documents using BM25.')
    parser.add_argument('--infile', default='../outputs/queries/trec_2021_baseline_v2.jsonl', help='output from rewriter')
    parser.add_argument('--qrelfile', default='../outputs/query4bm25/bm25_cast_test1/qrels/test.tsv', help='output: ground truth')
    parser.add_argument('--queryfile', default='../outputs/query4bm25/bm25_cast_test1/queries.jsonl', help='output: query file')

    args = parser.parse_args()

    outdir = os.path.dirname(args.qrelfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    qrel_lines = []
    query_lines = []
    with open(args.infile, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)    # keys: q_id, text, result_text, result_id
            query_id = data.get("q_id")
            corpus_id = data.get("result_id")
            score = "1"
            qrel_lines.append(f"{query_id}\t{corpus_id}\t{score}")
            query_dict = dict()
            query_dict["_id"] = query_id
            query_dict["text"] = data.get("text")
            query_dict["metadata"] = {}
            query_lines.append(query_dict)

    with open(args.qrelfile, 'w', encoding='utf-8') as fout:
        fout.write("query-id\tcorpus-id\tscore\n")
        for line in qrel_lines:
            fout.write(line + '\n')

    with open(args.queryfile, 'w', encoding='utf-8') as fout:
        for query_dict in query_lines:
            fout.write(json.dumps(query_dict)+'\n')
