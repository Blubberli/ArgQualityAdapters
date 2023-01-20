#!/bin/bash

args=(
  # data arguments
  --data_dir ../data/europolis/justification
  --label moderation
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
  --adapter_name "../FusionModels/moderation/split2"
  --load_best_model_at_end True
  --class_weights True
  --pretrained_adapters_file "../adapterPaths/moderation.tsv"
  --fusion_path "../FusionModels/moderation/adapter0,adapter1,adapter2,adapter3,adapter4,adapter5,adapter6,adapter7,adapter8,adapter9,adapter10,adapter11,adapter12,adapter13,adapter14,adapter15,adapter16,adapter17,adapter18,adapter19,adapter20"
  )
python ./extract_attention_weights.py "${args[@]}" "$@"
