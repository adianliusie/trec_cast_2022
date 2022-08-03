import json
import os
import gzip

def prepare_bm25_data(infile, outfile):
    print("Input file: ", infile)
    corpus_JsonList = []
    with gzip.open(infile, 'rb') as f:
        for line in f:
            js = json.loads(line)
            corpus_dict = dict()
            corpus_dict["_id"] = js.get("docid")
            corpus_dict["title"] = js.get("title")
            corpus_dict["text"] = js.get("body")
            corpus_JsonList.append(corpus_dict)

    with open(outfile, 'w', encoding='utf-8') as fout:
        for corpus_dict in corpus_JsonList:
            fout.write(json.dumps(corpus_dict)+'\n')
    print("Output file: ", outfile)
    return

indir = "/home/mq227/rds/hpc-work/trec_cast_2022/data/2022_challenge_data/raw_collection/msmarco_v2_doc/"
outdir = "/home/mq227/rds/hpc-work/trec_cast_2022/data/2022_challenge_data/bm25_data/"
for f in os.listdir(indir):
    if not f.endswith('gz'):
        print("skip file ", f)
        continue
    infile = '{}{}'.format(indir, f)
    fname = f.replace('.gz', '')
    outfile = '{}{}.jsonl'.format(outdir, fname)

    prepare_bm25_data(infile, outfile)

