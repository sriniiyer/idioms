# Learning Programmatic Idioms for Scalable Semantic Parsing


# CONCODE


## Download Dataset

Create a folder named `concode/`

`mkdir concode`

Download the Concode dataset from https://drive.google.com/drive/folders/1kC6fe7JgOmEHhVFaXjzOmKeatTJy1I1W?usp=sharing into the concode folder. Make sure you have the files {train|valid|test}_shuffled_with_path_and_id_concode.json

## Extract idioms

The top 600 idioms are already included in this repo in concode_idioms_full_bpe_600/idioms0.json and you can skip this section if you only wish to train the models. In case you wish to extract idioms again, run the following command:

```
python build.py -train_file concode/train_shuffled_with_path_and_id_concode.json -valid_file concode/valid_shuffled_with_path_and_id_concode.json -test_file concode/test_shuffled_with_path_and_id_concode.json -train_num 100000 -valid_num 2000 -dataset concode -get_idioms -idiom_folder concode_idioms_full_bpe_600 -bpe_steps 600 -threads 10 > concode_idioms_full_bpe_600.print
```

## Build dataset by applying idioms

```
python build.py -train_file concode/train_shuffled_with_path_and_id_concode.json -valid_file concode/valid_shuffled_with_path_and_id_concode.json -test_file concode/test_shuffled_with_path_and_id_concode.json -output_folder /scratch/data_concode_ -train_num 100000 -valid_num 2001 -dataset concode -idioms concode_idioms_full_bpe_600/idioms0.json -threads 10 -color -bpe -max_idioms_to_load 200;
```

## Preprocess the dataset

```
mkdir /scratch/data_concode_/data/

python preprocess.py -use_new_split -bpe_vocab 10000 -train /scratch/data_concode_/train.dataset -valid /scratch/data_concode_/valid.dataset -test /scratch/data_concode_/test.dataset -save_data /scratch/data_concode_/data/concode -train_max 1000000000 -valid_max 4000 -seq2seq_words_min_frequency 6 -seq2seq_words_max_vocab 20000 -next_rules_max_vocab 19000 -tgt_words_min_frequency 2 -names_min_frequency 7 -names_max_vocab 33000 -src_seq_length 200 -tgt_seq_length 150 -dataset concode
```

## Train the concode model

```
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/p36/bin/python train.py -dropout 0.5 -data /scratch/data_concode_/data/concode -save_model /scratch/data_concode_/data/models/concode -epochs 30 -learning_rate 0.001 -learning_rate_decay 0.8 -seed 1126 -enc_layers 2 -dec_layers 2 -tgt_word_vec_size 512 -rnn_size 1024 -decoder_rnn_size 1024 -brnn -attn_type general -gpuid 0 -report_every 50 -encoder_type concode -decoder_type concode -batch_size 40 -src_word_vec_size 512 -prev_divider 4
```

## Predict on dev set

```
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/p36/bin/ipython run.ipy -- --mode predict --start 5 --end 30 --beam 3 --data_dir /scratch/data_concode_/ --tgt_len 500 --batch_size 1 --trunc 2000 --dataset concode --models_dir data/models/ --test_dir /scratch/data_concode_/
```

## Use best dev epoch to predict on test set

```
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/p36/bin/ipython run.ipy -- --mode test --start 5 --end 30 --beam 3 --data_dir /scratch/data_concode_/ --tgt_len 500 --batch_size 1 --trunc 2000 --dataset concode --models_dir data/models/ --test_dir /scratch/data_concode_/ —best_json /scratch/data_concode_/data/models/preds.json
```

The expected output for dev and test is approximately:

```
Dev is:

['Best', 'bleu', 'Epoch:', 24, 23.45, 'bleu', 23.45, 'exact', 9.45]
['Best', 'exact', 'Epoch:', 24, 9.45, 'bleu', 23.45, 'exact', 9.45]

And test is:

['Best', 'bleu', 'Epoch:', 24, 27.37, 'bleu', 27.37, 'exact', 12.4]
['Best', 'exact', 'Epoch:', 24, 12.4, 'bleu', 27.37, 'exact', 12.4]
```

Note that your results may vary slightly owing to randomness.


# ATIS

We have included this dataset for convenience in the atis/ folder. Since the evaluation script needs to execute queries, you need a copy of the atis database. Refer to https://github.com/sriniiyer/nl2sql to download and setup the atis database.

## Extract idioms

The top 400 idioms are already included in this repo in atis/idioms0.json and you can skip this section if you only wish to train the models. In case you wish to extract idioms again, run the following command:

```
~/miniconda3/envs/p36/bin/python build.py -train_file atis/train.json -valid_file atis/valid.json -test_file atis/test.json -train_num 100000 -valid_num 2000 -dataset sql -get_idioms -idiom_folder ./atis/ -bpe_steps 400 -threads 10 > atis.idioms.print
```

## Build the dataset

```
~/miniconda3/envs/p36/bin/python build.py -train_file atis/train.json -valid_file atis/valid.json -test_file atis/test.json -output_folder /scratch/data_atis -train_num 10000 -valid_num 10000 -dataset sql -idioms atis/idioms0.json -threads 10 -color -bpe -max_idioms_to_load 400
```

## Preprocess the dataset

```
mkdir /scratch/data_atis_/data/

python preprocess.py -bpe_vocab -1 -train /scratch/data_atis_/train.dataset -valid /scratch/data_atis_/valid.dataset -test /scratch/data_atis_/test.dataset -save_data /scratch/data_atis_/data/atis -train_max 10000 -valid_max 10000 -seq2seq_words_min_frequency 0 -seq2seq_words_max_vocab 20000 -next_rules_max_vocab 19000 -tgt_words_min_frequency 0 -src_seq_length 200 -tgt_seq_length 152 -dataset sql
```

## Train the model

```
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/p36/bin/python train.py -attbias -dropenc 1 -input_feed -param_init 0.1 -dropout 0.6 -data /scratch/data_atis_/data/atis -save_model /scratch/data_atis_/data/models/atis -epochs 60 -learning_rate 0.001 -learning_rate_decay 0.8 -seed 1123 -enc_layers 2 -dec_layers 2 -src_word_vec_size 1024 -tgt_word_vec_size 1024 -rnn_size 1024 -decoder_rnn_size 1024 -brnn -attn_type general -gpuid 0 -report_every 50 -encoder_type regular -decoder_type prod -batch_size 50 -prev_divider 4
```

## Predict on the dev set


Setup the atis database for this and update run.ipy with the database host, username and password.

```
~/miniconda3/envs/p36/bin/ipython run.ipy -- --mode predict --start 20 --end 60 --beam 5 --data_dir /scratch/data_atis_/ --tgt_len 500 --batch_size 1 --trunc 2000 --dataset sql --models_dir data/models/ --test_dir /scratch/data_atis_/

```

## Use best dev epoch to predict on test set

```
~/miniconda3/envs/p36/bin/ipython run.ipy -- --mode test --start 20 --end 60 --beam 5 --data_dir /scratch/data_atis_/ --tgt_len 500 --batch_size 1 --trunc 2000 --dataset sql --models_dir data/models/ --test_dir /scratch/data_atis_/ --best_json /scratch/data_atis_/data/models/preds.json;
```

The expected output for dev and test is approximately:

```
Dev is:

['Best', 'exact_sql', 'Epoch:', 40, 72.5050916496945, 'exact_sql', 72.5050916496945, 'den', 84.72505091649694]
['Best', 'den', 'Epoch:', 40, 84.72505091649694, 'exact_sql', 72.5050916496945, 'den', 84.72505091649694]

and test is:

['Best', 'exact_sql', 'Epoch:', 40, 0.0, 'exact_sql', 0.0, 'den', 84.59821428571429]
['Best', 'den', 'Epoch:', 40, 84.59821428571429, 'exact_sql', 0.0, 'den', 84.59821428571429]

```

Note that your results may vary slightly owing to randomness.
