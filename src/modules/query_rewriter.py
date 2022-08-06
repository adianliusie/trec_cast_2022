from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from types import SimpleNamespace
from tqdm import tqdm
import torch

from ..utils.data_loader import DataLoader
from ..utils.general import save_jsonl

class QueryRewriter:
    def __init__(self):
        """ load pre-trained T5 model finetuned on canard"""
        
        self.tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")
        if torch.cuda.is_available(): self.model.to('cuda')
        
    def rewrite_query(self, conv_state:dict):
        """ given conversation context, rewrites the final utterance to include all information"""

        text_input = ' ||| '.join(conv_state['context'])
        tokenized_input = self.tokenizer(text_input, return_tensors="pt").input_ids
        if torch.cuda.is_available(): tokenized_input = tokenized_input.to('cuda') 
        
        output = self.model.generate(input_ids=tokenized_input)
        query = self.tokenizer.decode(output[0], skip_special_tokens=True)        
        output = {'q_id': conv_state['query_id'], 
                  'text': query, 
                  'result_text': conv_state['result_text'],
                  'result_id': conv_state['result_id']}
        return output
    
    def rewrite_queries(self, data_name:str, output_path:str):
        """ converts all conv states to rewritten queries """
        
        eval_data = DataLoader(data_name)
        conv_states = eval_data.get_conv_states()
        
        queries = []
        for conv_state in tqdm(conv_states):
            query = self.rewrite_query(conv_state)
            queries.append(query)
        
        save_jsonl(queries, output_path)
    
    def gold_rewrite_queries(self, data_name:str, output_path:str):
        """ prepares the dataset with the gold standard queries """
        
        eval_data = DataLoader(data_name)
        queries = eval_data.get_gold_rewrites()
        
        save_jsonl(queries, output_path)


        
    