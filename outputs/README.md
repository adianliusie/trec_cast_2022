## Output formats

**queries:**
```javascript
[{ "q_id": "106_1", 
   "text":"What are the most common types of breast cancer",
   "result_text": "More research is needed. Types Breast cancer can be...",
   "result_id":"MARCO_D59865" },
   
  { "q_id": "106_2", 
  "text":"Once the cancer breaks out, how likely is it to spread?",
  "result_text": "Even though this condition doesn’t spread, it’s...",
  "result_id":"MARCO_D684514" }, 
  ...
]
```

**bm25 / reranked:**
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
