#!/bin/bash

args=(
  # data arguments
  --data_dir ../data/europolis/justification
  --label argumentative
  --labels_num 2
  --max_seq_length 256
  --text_col cleaned_comment
  --output_dir ./justificationAttention2
  # model arguments
  --model_name_or_path roberta-base
  # training arguments
  --learning_rate 0.0001
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 16
  #--metric_for_best_model macro_f1
  --save_strategy epoch
  --evaluation_strategy epoch
  --logging_strategy epoch
  --seed 42
  #--weight_decay 0.1
  --save_total_limit 1
  --num_train_epochs 1
  --adapter_name "../FusionModels/DQcorr/withOwn/argumentive/checkpoint-119/argument"
  --load_best_model_at_end True
  --class_weights True
  --pretrained_adapters_file "../adapterPaths/DQcorrAdapters/withOwnAdapter/argumentative.csv"
  --fusion_path "../FusionModels/DQcorr/withOwn/argumentive/checkpoint-119/adapter0,adapter1,adapter2,adapter3,adapter4"
  )
python ./extract_attention_weights.py "${args[@]}" "$@"
