# trer_research
Research while at university of Tokyo

# How to run:
1. Use pip to install requirements (comming soon)
2. Download open-webtext dataset from external/gpt-2-output-dataset save in data folder
3. Run gpt/datasets.py to create the datasets
-SUBSEQ
4. Install sdls-lite library (in external folder)
5. run ./program path-training-data path-to-queries > path-to-answers in external/SUBSEQ/SUBSEQ/src/
6. edit paths in gpt/test.py
7. Run gpt/test.py to get accuracies.

-GPT
4. Run training.py to train gpt model. (uses the open-webtext dataset unprocessed)
5. Edit gpt/test.py to set GPT type of test.
6. Run test.py 
