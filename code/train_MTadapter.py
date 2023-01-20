from train_STadapter import run_trainer, run_trainer_regression
from transformers import AutoTokenizer, AutoConfig, AutoModelWithHeads, AdapterTrainer
from data import ClassificationDataset, RegressionDataset
from args import parse_arguments
import pandas as pd
from transformers import set_seed
import os
from utils import get_dynamic_adapter_fusion

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_adapters(path_to_pretrained_adapters):
    return pd.read_csv(path_to_pretrained_adapters, sep="\t")


if __name__ == '__main__':
    # read in arguments
    # model args: all classification details
    # data args: path to dataset etc.
    # training args: learning rate, optimizer etc.
    model_args, data_args, training_args = parse_arguments()
    print("model is trained for %s " % data_args.label)
    set_seed(training_args.seed)

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # init a transformer config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=model_args.labels_num,
    )
    # create a model with adapter-specific head
    model = AutoModelWithHeads.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )
    pretrained_adapters = read_adapters(model_args.pretrained_adapters_file)
    adapter_counter = 0
    print("model is trained for %s " % data_args.label)

    for i in range(len(pretrained_adapters)):
        name = pretrained_adapters.name.values[i]
        path = pretrained_adapters.path.values[i]
        model.load_adapter(path, load_as="adapter%d" % adapter_counter, with_head=False)
        print("loading adapter %s as adapter adapter%d" % (name, adapter_counter))
        adapter_counter += 1
    print("number of adapters trained the fusion with is %d" % (adapter_counter))

    # Add a matching classification head
    model.add_classification_head(
        model_args.adapter_name,
        num_labels=model_args.labels_num
    )

    adapter_setup = get_dynamic_adapter_fusion(adapter_number=adapter_counter)
    # Add a fusion layer for all loaded adapters
    model.add_adapter_fusion(adapter_setup)
    # Activate the adapter
    # Unfreeze and activate fusion setup

    model.set_active_adapters(adapter_setup)
    model.train_adapter_fusion(adapter_setup)

    if model_args.labels_num == 1:

        train = RegressionDataset(path_to_dataset=data_args.data_dir + "/train.csv",
                                  label=data_args.label, tokenizer=tokenizer, text_col=data_args.text_col)
        dev = RegressionDataset(path_to_dataset=data_args.data_dir + "/val.csv",
                                label=data_args.label, tokenizer=tokenizer, text_col=data_args.text_col)
        test = RegressionDataset(path_to_dataset=data_args.data_dir + "/test.csv",
                                 label=data_args.label, tokenizer=tokenizer, text_col=data_args.text_col)
        run_trainer_regression(train_data=train, dev_data=dev, test_data=test, model=model, training_args=training_args,
                               model_args=model_args)
    else:

        training_args.metric_for_best_model = "macro_f1"

        train = ClassificationDataset(path_to_dataset=data_args.data_dir + "/train.csv",
                                      label=data_args.label, tokenizer=tokenizer, text_col=data_args.text_col)
        dev = ClassificationDataset(path_to_dataset=data_args.data_dir + "/val.csv",
                                    label=data_args.label, tokenizer=tokenizer, text_col=data_args.text_col)
        test = ClassificationDataset(path_to_dataset=data_args.data_dir + "/test.csv",
                                     label=data_args.label, tokenizer=tokenizer, text_col=data_args.text_col)
        run_trainer(train_data=train, dev_data=dev, test_data=test, model=model, training_args=training_args,
                    class_weights=training_args.class_weights, model_args=model_args)
