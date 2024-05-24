# LDM-DA
Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning

Official Implementation of "[Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning](https://arxiv.org/pdf/2405.14170)".

<img src="resources/model.png" width = "800" />

LLM-DA first analyzes historical data to extract temporal rules and utilizes the powerful generative capabilities of LLMs to generate general rules. Subsequently, LLM-DA updates these rules using current data. Finally, the updated rules are applied
to predict future events.

## Requirements
```
pip install -r requirements.txt
```
Set your OpenAI API key in `.env` file

## Mining Rules with LLM-DA
 
1.  Temporal Logical Rules Sampling
```
python rule_sampler.py -d ${DATASET} -m 3 -n 200 -p 16 -s 12 --is_relax_time No
```
4.  Candidate Reasoning
```
python reasoning.py -d ${DATASET} -r confidence.json -l 1 2 3 -p 8 --min_conf 0.01  --weight_0 0.5 --bgkg all --score_type noisy-or --is_sorted True --evaluation_type origin --gpu 0 --top_k 20 --is_return_timestamp No --window 0
```
