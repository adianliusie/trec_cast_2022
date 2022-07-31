# TREC CaST 2022 attempt for HEATWAVE-CAM 

---
### Set Up 
* clone this directory 
	```
	git clone https://github.com/usnistgov/trec_eval
	```
* Install the relevant dependencies
	```
	pip install -r requirements.txt
	```

* In the data directory download the relevant data sets (further instructions can be found in the data directory) 
**Need to add corpora set up (MS MARCO, KILT, WaPo)*

* In the scripts directory run the 3 stage pipeline of query rewriting, BM25, and reranking  
**Need to add BM25 implementation in src and create script for it* 

* Evaluation of overall system performance can be done in the eval folder 
**Not tested yet- need correct Doc-IDs to test the script*


### Repository Structure
The main structure of the repository is as follows:

```
trec_cast_2022
└── src
│    └── modules
│    └── utils
└── scripts
│    └── run_rewrite
│    └── run_bm25
│    └── run_reranking
└── outputs
│    └── queries
│    └── bm25
│    └── reranking
└── eval
└── data
```

- src: all the python code for the 3 stages
- scripts: the scripts to process the data for each stage
- outputs: where the output files of each stage is saved
- data: where all the external data is stored (used by src)
- eval: the evaluation scripts, used to measure overall end-to-end performance

