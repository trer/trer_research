# trer_research
Research while at university of Tokyo

# How to run:
1. Use pip to install requirements.txt
2. Download open-webtext dataset from external/gpt-2-output-dataset save in data folder
3. edit and run gpt/datasets.py to create the dataset you want.
   
-SUBSEQ
1. Install sdls-lite library (in external folder)
2. In SUBEQ/src run "./program path-training-data path-to-queries > path-to-answers"
3. edit paths in test.py
4. Run test.py to get accuracy.
5. *The answers file will contain training time, total infernce time (needs to be divided by total of inferences), and total memory used.


-GPT
1. Run training.py to train gpt model. (uses the open-webtext/tiny_shakespeare dataset unprocessed)
2. Edit test.py to set GPT type of test.
3. Run test.py to get accuracy and inference time
