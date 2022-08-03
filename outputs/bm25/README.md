**bm25 input format:**
```
datadir
└── qrels
│    └── test.csv
└── queries.jsonl
└── corpus.jsonl
```
queries.jsonl
```javascript
[{"_id": "106_1", "text": "What are the most common types of cancer in regards to me?"},
 {"_id": "106_2", "text": "Once the cancer breaks out, how likely is it to spread?"},
  ...
]
```
corpus.jsonl
```javascript
[{"_id": "msmarco_doc_00_0", 
  "title": "0-60 Times - 0, ...", 
  "text": "0-60 Times - 0-60 | 0 to 60 Times & 1/4 Mile Times | Zero to 60 Car Reviews\n0-60 Times\nThere are many ways to measure the power a ..."},
 {"_id": "msmarco_doc_00_4806", 
   "title": "Ethel Percy Andrus Gerontology Center [WorldCat Identities]", 
   "text": "Ethel Percy Andrus Gerontology Center [WorldCat Identities]\nEthel Percy Andrus Gerontology Center\nOverview\nWorks:\n233  works in  338  publication ..."},
  ...
]
```
qrels/test.tsv: ground truth, used in evaluation
```javascript
query-id        corpus-id       score
106_1   MARCO_D59865    1
106_2   MARCO_D684514   1
...
```

**bm25 output format:**
```javascript
[{"q_id": "106_1", 
   "results":[
	   {"reult_id":"MARCO_D49524",
	    "text":"What are the most common Sites for Melanoma ...",
	    "score":71.4},
	    
	   {"reult_id":"MARCO_D59865",
	    "text":"More research is needed. Types Breast cancer...",
	    "score":70.2},    
	   ...
	]},
 {"q_id": "106_2", 
  "results":[...]},
  ...
]
```
