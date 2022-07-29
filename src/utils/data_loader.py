from .general import flatten, load_json

class Conversation:
    def __init__(self, **kwargs):
        conv = kwargs
        self.conv_id = conv['number']
        self.utterances = [Utterance(**utt) for utt in conv['turn']]
        
class Utterance():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.text = self.raw_utterance

    @property
    def text_passage(self):
        return [self.text, self.passage]
        
    def __repr__(self):
        return self.text
    
class DataLoader:
    def __init__(self, data_name):
        if data_name == 'trec_2021': self.load_trec_2021()
        else: raise ValueError('Invalid data set provided')
            
    def load_trec_2021(self):
        path='/home/al826/rds/hpc-work/2022/trec/trec_cast_2022/data/treccastweb/2021/2021_automatic_evaluation_topics_v1.0.json'
        json_data = load_json(path)
        
        self.convs = []
        for conv in json_data:
            self.convs.append(Conversation(**conv))
    
    def get_conv_states(self):
        output = []
        for conv in self.convs:
            utts = conv.utterances
            
            for k in range(len(utts)):
                cur_utt = utts[k]
                context_1 = [utt.text for utt in utts[max(k-5, 0):max(k-2, 0)]]
                context_2 = [utt.text_passage for utt in utts[max(k-2, 0):k]]
                utt_text = [cur_utt.text]               
                cur_ex = {'query_id':len(output),
                          'context':context_1 + flatten(context_2) + utt_text, 
                          'passage':cur_utt.passage, 
                          'result_id':cur_utt.canonical_result_id, 
                          'passage_id':cur_utt.passage_id}
                output.append(cur_ex)        
        return output
    