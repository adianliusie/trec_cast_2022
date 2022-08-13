from .general import flatten, load_json, get_base_dir

BASE_DIR = get_base_dir()

class TrecConversation:
    def __init__(self, conv_id:int, utts:list, fields:dict=None):
        self.conv_id = str(conv_id)
        
        #reformat utterance fields if necessary
        if fields:
            utts = [self.reformat_utt(utt, fields) for utt in utts]
            
        self.utterances = [TrecUtterance(**utt) for utt in utts]
    
    @staticmethod
    def reformat_utt(utt, fields):
        utt = utt.copy()
        for k, v in fields.items():
            utt[k] = utt.pop(v)
        return utt

class TrecUtterance():
    def __init__(self, utt_id:int, text:str, result_text:str=None, result_id:int=None, gold_rewritten_utt:str=None, **kwargs):
        self.utt_id = str(utt_id)
        self.text = text
        self.result_text = result_text
        self.result_id = result_id
        self.gold_rewritten_utt = gold_rewritten_utt

    @property
    def text_passage(self):
        return [self.text, self.result_text]
        
    def __repr__(self):
        return self.text
    
class DataLoader:
    def __init__(self, data_name):
        if data_name == 'trec_2021': 
            self.convs = self._load_trec_2021()
        elif data_name == 'trec_2022':
            self.convs = self._load_trec_2022()

        else: raise ValueError('Invalid data set provided')
            
    def get_gold_rewrites(self):
        output = []
        for conv in self.convs:
            for utt in conv.utterances:
                cur_ex = {'q_id': f'{conv.conv_id}_{utt.utt_id}', 
                          'text': utt.gold_rewritten_utt,
                          'result_text': utt.result_text,
                          'result_id': utt.result_id}
                output.append(cur_ex)        
        return output
    
    def get_raw_context(self, num_utts:int=None, num_resp:int=None):
        print(num_utts, num_resp)
        if num_resp:
            assert num_resp < num_utts

        output = []
        for conv in self.convs:
            utts = conv.utterances      
            for k in range(len(utts)):
                utt = utts[k]
                utt_text = [utt.text]               

                if num_resp:
                    context_1 = [u.text for u in utts[max(k-num_utts, 0):max(k-num_resp, 0)]]
                    context_2 = [u.text_passage for u in utts[max(k-num_resp, 0):k]]
                    context = context_1 + flatten(context_2) + utt_text
                else:
                    past_context = [u.text_passage for u in utts[max(k-num_utts, 0):k]]
                    context = flatten(past_context) + utt_text
                
                context = ' '.join(context[-200:])
                cur_ex = {'q_id': f'{conv.conv_id}_{utt.utt_id}', 
                          'text': context,
                          'result_text': utt.result_text,
                          'result_id': utt.result_id}
                output.append(cur_ex)         
        return output
    
    def get_conv_states(self):
        output = []
        for conv in self.convs:
            utts = conv.utterances
            for k in range(len(utts)):
                utt = utts[k]
                context_1 = [u.text for u in utts[max(k-4, 0):max(k-2, 0)]]
                context_2 = [u.text_passage for u in utts[max(k-2, 0):k]]
                utt_text = [utt.text]               
                cur_ex = {'query_id':f'{conv.conv_id}_{utt.utt_id}',
                          'context':context_1 + flatten(context_2) + utt_text, 
                          'result_text':utt.result_text, 
                          'result_id':utt.result_id}
                output.append(cur_ex)        
        return output
    
    def _load_trec_2021(self):
        """load trec 2021 eval data """
        
        path=f'{BASE_DIR}/data/treccastweb/2021/2021_automatic_evaluation_topics_v1.0.json'
        json_data = load_json(path)
        
        # fields defines how to map given fields to standard 
        fields = {'utt_id':'number', 'text':'raw_utterance', 'result_text':'passage', 
                  'result_id':'canonical_result_id', 'gold_rewritten_utt':'automatic_rewritten_utterance'}
        
        convs = []
        for conv in json_data:
            convs.append(TrecConversation(conv_id=conv['number'], utts=conv['turn'], fields=fields))
        return convs
    
    def _load_trec_2022(self):
        path=f'{BASE_DIR}/data/treccastweb/2022/2022_evaluation_topics_flattened_duplicated_v1.0.json'
        json_data = load_json(path)
        
        fields = {'utt_id':'number', 'text':'utterance', 'result_text':'response',
                  'result_id':'provenance', 'gold_rewritten_utt':'manual_rewritten_utterance'}
        
        convs = []
        for conv in json_data:
            if conv['number'] == 142: continue
            convs.append(TrecConversation(conv_id=conv['number'], utts=conv['turn'], fields=fields))
        return convs
    