# Enhancing the Empathetic Dialogues through Emotional Attention Network

This is the main code for this paper.
## Abstract

This paper proposes a method to use emotional attention networks to improve the empathy capacity of current dialogue models, which will make the responses generated by language models more empathetic. In this approach, we encode the responses and context from the current language model and classify them through an attention network to specific adapters for emotional fine-tuning to generate more empathetic responses. Experiments show that our method can effectively improve the empathy ability of current language models. We also emphasize the importance of the modules we use through the analysis and testing of different modules.

## Main process
![](https://github.com/huibaisedeshijie/Enhancing-the-Empathetic-Dialogues-through-Emotional-Attention-Network/blob/main/1-main-stream(simple).drawio%20(1).png)

The main steps of the system: Step 1 Import the pre-trained model and generate the response. Step 2 encodes the response and context and imports the emotional attention network for classification. Step 3, use sentiment adapters and adapter fusion to fine-tune and generate final responses.

## Environment
Check the packages needed or simply run the command
```
pip install -r requirements.txt
```
Some files are placed on Google Drive because they are too large, download and put in/empathetic-dialogue/
```

```

Pre-trained glove embedding: glove.6B.300d.txt inside folder /vectors/.
```
To be downloaded from http://nlp.stanford.edu/data/glove.6B.zip
```
## Train 
```
python3 main.py --model experts  --label_smoothing --noam --emb_dim 300 --hidden_dim 100 --hop 1 --heads 2 --topk 5 --cuda --pretrain_emb --softmax --basic_learner --schedule 10000 --save_path save/
```
## Test
```
python3 main.py --model experts  --label_smoothing --noam --emb_dim 300 --hidden_dim 100 --hop 1 --heads 2 --topk 5 --cuda --pretrain_emb --softmax --basic_learner --schedule 10000 --save_path save/moel/
```
