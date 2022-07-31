# Baseline systems

### Query Rewriter
The query Rewriter is a T5 based neural rewriter trained on CANARD

Input to the seq2seq model is the previous 4 queries, the last 2 responses, and the current query. Each part is separated with '|||'-

```
 <query 1> ||| <query 2> ||| <query 3> ||| <result 3> ||| <query 4> ||| <result 4>  ||| <query 5>  
```

T5 Model is then finetuned on the [CANARD](http://users.umiacs.umd.edu/~jbg/docs/2019_emnlp_sequentialqa.pdf) data set 

*The trained rewriter is taken from the huggingface hub: [model link](https://huggingface.co/castorini/t5-base-canard)*

### Reranker
The baseline reranker is based on [Sentence-BERT](https://arxiv.org/abs/1908.10084), further fine-tuned on MS-MACRO 

The query is converted into a vector embedding. Each result (document/passage) is also converted into a vector embedding. The reranked score is simply the dot product between the query vector and the passage vector. 

*The trained reranker is taken from the huggingface hub: [model link](https://huggingface.co/sentence-transformers/msmarco-distilbert-dot-v5)*

