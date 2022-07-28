# TREC CaST 2022 attempt for HEATWAVE-CAM 

### Relevant links


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

