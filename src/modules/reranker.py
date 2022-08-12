import numpy as np
import spacy

from typing import List
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util

from ..utils.general import load_jsonl, save_jsonl

class SbertReranker:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/msmarco-distilbert-dot-v5')
        
    def get_doc_scores(self, query:str, docs:List[str])->List[int]:
        """old method, uses entire document as the response"""
        query_emb = self.model.encode(query)
        doc_emb = self.model.encode(docs)

        #Compute dot score between query and all document embeddings
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        return scores
    
    def rerank(self, query_path:str, docs_path:str, output_path:str):
        queries = load_jsonl(query_path)
        queries_docs = load_jsonl(docs_path)

        output = []
        for query, q_docs in tqdm(zip(queries, queries_docs), total=len(queries)):
            #Get similarity score (with query) for every doc
            assert str(query['q_id']) == str(q_docs['q_id'])
            query_text = query['text']
            docs_text = [doc['text'] for doc in q_docs['results']]
            scores = self.get_doc_scores(query_text, docs_text)
            
            #save scores and then reorder
            ordering = np.argsort(scores)[::-1]
            for i in range(len(q_docs['results'])):
                q_docs['results'][i]['score'] =  scores[i]
            new_results = [q_docs['results'][i] for i in ordering]
            q_docs['results'] = new_results
            
            #add new reordered documents
            output.append(q_docs)
        
        save_jsonl(output, output_path)
        
class PassageReranker(SbertReranker):
    def __init__(self):
        super().__init__()
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


    def get_doc_scores(self, query:str, docs:List[str], max_len=250):
        #get query embedding
        query_emb = self.model.encode(query)

        #process doc into sentences
        proc_docs = self.nlp.pipe(docs, n_process=1)
        
        #get scores per passage
        scores = []
        for doc in tqdm(proc_docs):
            sents = list(doc.sents)
            passages = self.create_passages(sents)
            
            if len(passages)==0:
                print('long passage skipped')
                scores.append(0)
                continue
                
            pass_emb = self.model.encode(passages)
            doc_score = util.dot_score(query_emb, pass_emb)[0].cpu().tolist()
            scores.append(max(doc_score))
        return scores
    
    @staticmethod
    def create_passages(sents, max_len=250):
        #create passages using maximum word limit
        passages = []
        cur_passage = ''
        for sent in sents:
            if (cur_passage +sent.text).count(' ') < max_len:
                cur_passage += ' ' + sent.text
            elif cur_passage:
                passages.append(cur_passage)
                cur_passage = sent.text

        if cur_passage:
            passages.append(cur_passage)
        return passages
    