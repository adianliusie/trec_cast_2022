from typing import List
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util

from ..utils.general import load_jsonl, save_jsonl

class SentenceReranker:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/msmarco-distilbert-dot-v5')

    def get_docs_rank_order(self, query:str, docs:str)->List[int]:
        query_emb = self.model.encode(query)
        doc_emb = self.model.encode(docs)

        #Compute dot score between query and all document embeddings
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        return np.argsort(scores)[::-1]

    def rerank(self, query_path:str, docs_path:str, output_path:str):
        queries = load_jsonl(query_path)
        queries_docs = load_jsonl(docs_path)

        output = []
        for query, q_docs in tqdm(zip(queries, queries_docs), total=len(queries)):
            assert str(query['q_id']) == str(q_docs['q_id'])
            query_text = query['text']
            docs_text = [doc['text'] for doc in q_docs['docs']]
            
            ordering = self.get_docs_rank_order(query_text, docs_text)
            new_docs = [q_docs['docs'][i] for i in ordering]
            q_docs['documents'] = new_docs
            output.append(q_docs)
        
        save_jsonl(output, output_path)