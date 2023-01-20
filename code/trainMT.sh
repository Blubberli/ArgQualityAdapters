#!/bin/bash

args=(
  # data arguments
  --data_dir data/europolis/justification
  --label justification
  --labels_num 4
  --max_seq_length 256
  --text_col cleaned_comment
  --output_dir ./justificationMT
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
  --adapter_name justificationMT
  --load_best_model_at_end True
  --class_weights True
  --pretrained_adapters_file "adapterPaths/pretrainedAdapters.tsv"
  --fusion_path ""
  )
python ./train_MTadapter.py "${args[@]}" "$@"
