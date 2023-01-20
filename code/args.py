# based on transformers/examples/pytorch/text-classification/run_glue.py

import os
import sys
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    label: Optional[str] = field(metadata={"help": "the column storing the layer for classification"})
    testdata: Optional[str] = field(default="", metadata={"help": "dataset for inference"})
    text_col: Optional[str] = field(default="cleaned_comment", metadata={"help": "the column which stores the text"})
    data_dir: Optional[str] = field(
        default=str('5foldStratified/jlev')
    )

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },

    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    adapter_name: str = field(
        metadata={"help": "name of the adapter to train"}
    )
    fusion_path: Optional[str] = field(
        metadata={"help": "the path to the file specifiying the adapter fusion"}
    )
    pretrained_adapters_file: Optional[str] = field(
        metadata={"help": "the path to the file specifiying the pretrained adapters"}
    )

    labels_num: Optional[int] = field(
        default=2, metadata={"help": "number of labels in the model output"}
    )


@dataclass
class KfoldTrainingArguments(TrainingArguments):
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the seed to base the training on"},
    )
    class_weights: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use class weights for imbalanced data."
        },
    )

    adapter_dropout: Optional[bool] = field(
        default=False,
        metadata={"help": "if true, some adapters are dropped out during training"}
    )


def parse_arguments():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, KfoldTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argumentative to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    for x in (model_args, data_args, training_args):
        pprint(x)
    return model_args, data_args, training_args
