# HOW TO USE BERT TO PREDICT A CHINESE SENTIMENT CLASSIFICATION?

### Environment

python = 3.6

tensorflow = 1.10

### Pre-train model
First, you need to clone my code and download chinese pretrained model here:



*   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M
    parameters

### Dataset

You can explore data in the 'data' directory. We use sentiment in this instruction. You can test another one which is about stocks by you interests.



### Sentence (and sentence-pair) classification tasks



This example code fine-tunes `BERT-Chinese-Model` on the Chinese sentiment classification, which only contains 17000 examples.   **[`sim_model`](https://drive.google.com/file/d/1ndSg0Za0Fk5YySaS-ghmJGdUDcZxzg1z/view?usp=sharing)** The classification model is provided by us. So you can use it directly. They should be in the 'sim_model' directory. If you want to make you own classification train, you can execute script below on your shell.


```shell

python run_classifier.py \
  --task_name=Sim \
  --do_train=true \
  --do_eval=true \
  --data_dir=/path/to/data/ \
  --vocab_file=/path/to/bert/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/path/to/bert/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/path/to/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=70 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/sim_model/
```

You should see output like this:

```
***** Eval results *****
  eval_accuracy = 0.89801764
  eval_loss = 0.31444973
  global_step = 1605
  loss = 0.3143656
```

This means that the Dev set accuracy was 89.80%. 

#### Prediction from classifier

You can predict the sentence in inference mode by using the --do_predict=true command. You need to have a file named test.csv in the input folder. We have provided test.csv for you and you can add what you want to explore in it!

In the code directory, use you shell to excute script.

```shell

python run_classifier.py \
  --task_name=Sim \
  --do_predict=true \
  --data_dir=/path/to/data/ \
  --vocab_file=/path/to/bert/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/path/to/bert/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/path/to/fine/tuned/classifier/ \
  --max_seq_length=70 \
  --output_dir=/sim_prediction_output/
```

Output will be created in file called test_results.csv in the output folder. Each line will contain output for each sample, columns are the class probabilities.

### What is the difference between original BERT code and this?

Little. This is the beauty of BERT. I only need to write a processor class to import the data then do a little revise about how to process data. So it is why fine-tune really makes BERT adapts to different NLP tasks.

### What are these labels' meaning? 

0 means neutral. 1 means good. 2 means bad.

### Demo

If you want to see a demo, here it is!
**[`demo`](https://drive.google.com/file/d/1IGW-BGooAzWN_3t0CuHUtBZKQszcVCSo/view?usp=sharing)**:

### Cite
This is the BERT article.

For now, cite [the Arxiv paper](https://arxiv.org/abs/1810.04805):

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

Thanks to bilibili's video creator '迪哥有点愁' to provide the dataset and guidance!
