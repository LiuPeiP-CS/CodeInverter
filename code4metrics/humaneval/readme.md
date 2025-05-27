Instructions for Use:

1. Re-exe and Re-com:
Run re-exe&re-com/agent_1.3B.py or re-exe&re-com/agent_6.7B.py to obtain decompilation results.

Run re-exe&re-com/recompile.py to calculate the corresponding Re-exe and Re-com metrics.

2. ES Metric:
After obtaining the decompilation results in the previous step, run es/edit_similarity.py to calculate the ES metric. You need to modify the path to the decompilation result in the .py file according to your actual situation.

3. pass@k
Run pass@k/agent_1.3B_pass.py or pass@k/agent_6.7B_pass.py to obtain the results of 20 rounds of decompilation.

Run pass@k/passAT.py to calculate the corresponding passAT metric.

Note: Parts of the code and paths involved in the above .py files need to be modified according to your actual situation.