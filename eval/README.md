Evaluation 
---
Download the official TREC evaluation tool in this directory:

```
git clone https://github.com/usnistgov/trec_eval
```

Then  make the tool using ```make``` in the eval directory 

Evaluating the script can be done using the script :
```
evaluate.py --queries ../outputs/queries/<query_file>.jsonl --documents ../outputs/reranked/<ordered_docs_file>.jsonl  
```

---

*To understand the TREC evaluation tool used under the hood, check out* [this useful blog post](http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system) 


