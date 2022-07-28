from pygaggle.pygaggle.rerank.base import Query, Text
from pygaggle.pygaggle.rerank.transformer import MonoT5

class ReRanker:
    def __init__(self):
        self.reranker = MonoT5()
        
    def rerank(self, query:str, passages:List[Tuple['doc_id', 'passage_text']]):
        query_obj = Query(query)
        texts = [Text(p[1], {'docid': p[0]}, 0) for p in passages]
        reranked = reranker.rerank(query, texts)
        return reranked
