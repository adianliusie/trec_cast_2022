import logging
from typing import Dict, List

import spacy
from tqdm import tqdm
from time import time

logger = logging.getLogger(__name__)

#Parent class for any reranking model
class Rerank:
    
    def __init__(self, model, batch_size: int = 128, **kwargs):
        self.cross_encoder = model
        self.batch_size = batch_size
        self.rerank_results = {}

        try:
            self.load_spacy()
        except:
            spacy.cli.download("en_core_web_sm")
            self.load_spacy()

    def load_spacy(self):
        self.nlp = spacy.load("en_core_web_sm", exclude=[
                 "parser", "tagger", "ner", "attribute_ruler", "lemmatizer", "tok2vec"])
        self.nlp.enable_pipe("senter")
        self.nlp.max_length = 2000000  # for documents that are longer than the spacy character limit

        
    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        # sentence_pairs, pair_ids = [], []
        pair_ids = []
        rerank_scores = []
        
        for query_id in tqdm(results, total=len(results)):
            if len(results[query_id]) > top_k:
                doc_ids = [doc_id for doc_id, _ in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]]
            else:
                doc_ids = results[query_id].keys()
            
            doc_texts = [corpus[doc_id].get("text") for doc_id in doc_ids]
            proc_docs = self.nlp.pipe(doc_texts, n_process=1)
            
            for doc_id, proc_doc in tqdm(zip(doc_ids, proc_docs), total=len(doc_ids)):
                # t0 = time()
                sentence_pairs = []
                pair_ids.append([query_id, doc_id])
                sents = list(proc_doc.sents)
                passages = self.create_passages(sents)
                # print('processing passages using {:.3f}'.format(time()-t0))

                if len(passages) == 0:
                    print('long passage skipped')
                    rerank_scores.append(0)
                    continue

                # t1 = time()
                for passage in passages:
                    corpus_text = (corpus[doc_id].get("title", "") + " " + passage).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])
                # print('create pairs using {:.3f}'.format(time()-t1))

                # t2 = time()
                rerank_scores_per_doc = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]
                # print('calculate scores using {:.2f}'.format(time()-t2))
                rerank_scores.append(max(rerank_scores_per_doc))

        #### Starting to Rerank using cross-attention
        # logging.info("Starting To Rerank Top-{}....".format(top_k))
        # rerank_scores = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]

        #### Reranking results
        self.rerank_results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[query_id][doc_id] = score

        return self.rerank_results 

    @staticmethod
    def create_passages(sents, max_len=250):
        #create passages using maximum word limit
        passages = []
        cur_passage = ''
        for sent in sents:
            if (cur_passage+sent.text).count(' ') < max_len:
                cur_passage += ' '+sent.text
            elif cur_passage:
                passages.append(cur_passage)
                cur_passage = sent.text
            else:
                passages.append(sent.text)

        #at end of document, if passage add it
        if cur_passage:
            passages.append(cur_passage)
        return passages
