import logging
from typing import Dict, List

import spacy

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
        
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    sentence_pairs = []
                    pair_ids.append([query_id, doc_id])
                    doc_text = corpus[doc_id].get("text")
                    #process doc into sentences
                    proc_docs = self.nlp(doc_text)
                    sents = list(proc_docs.sents)
                    passages = self.create_passages(sents)
        
                    if len(passages)==0:
                        print('long passage skipped')
                        rerank_scores.append(0)
                        continue
                    
                    for passage in passages:
                        # corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                        corpus_text = (corpus[doc_id].get("title", "") + " " + passage).strip()
                        sentence_pairs.append([queries[query_id], corpus_text])

                    rerank_scores_per_doc = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]
                    rerank_scores.append(max(rerank_scores_per_doc))

            
            else:
                for doc_id in results[query_id]:
                    sentence_pairs = []
                    pair_ids.append([query_id, doc_id])
                    doc_text = corpus[doc_id].get("text")
                    #process doc into sentences
                    proc_docs = self.nlp(doc_text)
                    sents = list(proc_docs.sents)
                    passages = self.create_passages(sents)
        
                    if len(passages)==0:
                        print('long passage skipped')
                        rerank_scores.append(0)
                        continue
                    
                    for passage in passages:
                        # corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                        corpus_text = (corpus[doc_id].get("title", "") + " " + passage).strip()
                        sentence_pairs.append([queries[query_id], corpus_text])
                    # corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    # sentence_pairs.append([queries[query_id], corpus_text])
                    rerank_scores_per_doc = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]
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
