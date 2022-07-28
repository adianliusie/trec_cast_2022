from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ReRanker:
    def __init__(self, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco-10k")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("castorini/monot5-base-msmarco-10k")
        self.model.to(device)
        self.device = device

    def format_pairs(query, docs):
        #query = query['text']
        #docs  = [doc['text'] for doc in docs]
        ranking_patterns = [f'{query} {doc}' for doc in docs]
        reranking_ids = [self.tokenizer(pat, return_tensors="pt").input_ids.to(self.device) for pat in ranking_patterns]
        return reranking_ids
    
    def get_score():
        
    
    def rerank_query(query, docs):
        ids = self.format_pairs(query, docs)
        for 
        
    def rerank_passages(query_path, relevant_docs_path):
        
        