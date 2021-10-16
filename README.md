# BERT

You can download chinese pretrained model here:



*   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M
    parameters



### Sentence (and sentence-pair) classification tasks

This example code fine-tunes `BERT-Chinese-Model` on the Chinese sentiment classification, which only contains 17000 examples. The classification model is provided by us. So you can use it directly. They are in the 'sim_output' directory.

#### Prediction from classifier

You can predict the sentence in inference mode by using the --do_predict=true command. You need to have a file named test.csv in the input folder. Output will be created in file called test_results.csv in the output folder. Each line will contain output for each sample, columns are the
class probabilities.

```shell

python run_classifier.py \
  --task_name=Sim \
  --do_predict=true \
  --data_dir=/path/to/data \
  --vocab_file=/path/to/bert/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/path/to/bert/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/path/to/fine/tuned/classifier/ \
  --max_seq_length=70 \
  --output_dir=/sim_prediction_output/
```


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

