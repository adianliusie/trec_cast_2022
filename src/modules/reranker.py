from typing import List
import numpy as np
from tqdm import tqdm
import os, sys 

from sentence_transformers import SentenceTransformer, util

from ..utils.general import load_jsonl, save_jsonl

class SentenceReranker:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/msmarco-distilbert-dot-v5')

    def get_doc_scores(self, query:str, docs:str)->List[int]:
        query_emb = self.model.encode(query)
        doc_emb = self.model.encode(docs)

        #Compute dot score between query and all document embeddings
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        return scores

    def rerank(self, query_path:str, docs_path:str, output_path:str):
        if os.path.exists(output_path):
            sys.exit(f"{output_path} file exists!")

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
