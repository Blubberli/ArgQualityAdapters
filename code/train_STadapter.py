from transformers import AutoTokenizer, AutoConfig, EvalPrediction, AutoModelWithHeads, AdapterTrainer, AdapterConfig, \
    EarlyStoppingCallback, TrainerCallback
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, mean_absolute_error, \
    mean_squared_error
from scipy.stats import pearsonr
from transformers import set_seed
import torch
from args import parse_arguments
import numpy as np
from data import ClassificationDataset, RegressionDataset
from torch import nn
from sklearn.utils import class_weight
from pathlib import Path

# check for GPUs or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU in use:')
else:
    print('using the CPU')
    device = torch.device("cpu")


class CustomTrainer(AdapterTrainer):
    """ This trainer can be used to compute the loss using class weights to tackle imbalanced classes"""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You pass the class weights when instantiating the Trainer
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).float().to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def run_trainer(train_data, dev_data, test_data, model, training_args, class_weights, model_args):
    def assert_dir(path):
        assert not Path(path).exists(), f'output dir {path} already exists!'

    # create a new directory if it doesn't exist
    final_output_dir = Path("%s" % (training_args.output_dir))
    assert_dir(final_output_dir)
    final_output_dir.mkdir()
    print("results are saved to %s" % final_output_dir)

    training_args.output_dir = final_output_dir
    if class_weights:
        class_weights = class_weight.compute_class_weight(classes=np.unique(train_data.labels),
                                                          y=train_data.labels,
                                                          class_weight="balanced")
        print("Using class weights \n")
        print(class_weights)
        trainer = CustomTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=dev_data,
            compute_metrics=compute_metrics_classification,
        )
    else:
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=dev_data,
            compute_metrics=compute_metrics_classification,
        )

    trainer.train()
    model.to(device)
    train_results = trainer.evaluate(train_data)
    # evaluate on dev set
    dev_results = trainer.evaluate(dev_data)
    # evaluate on test set
    test_results = trainer.evaluate(test_data)

    train_report = train_results["eval_report_csv"]
    dev_report = dev_results["eval_report_csv"]

    # save the results and predictions on each dataset
    save_predictions(trainer, dev_data, test_data, final_output_dir, train_results, dev_results, test_results)
    print(train_report)
    print(dev_report)


def run_trainer_regression(train_data, dev_data, test_data, model, training_args, model_args):
    """This trainer can be used for regression tasks"""

    def assert_dir(path):
        assert not Path(path).exists(), f'output dir {path} already exists!'

    final_output_dir = Path("%s" % (training_args.output_dir))
    assert_dir(final_output_dir)
    final_output_dir.mkdir()
    print("results are saved to %s" % final_output_dir)
    training_args.output_dir = final_output_dir

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        compute_metrics=compute_metrics_regression,
    )

    trainer.train()
    model.to(device)

    train_results = trainer.evaluate(train_data)
    # evaluate on dev set
    dev_results = trainer.evaluate(dev_data)
    # evaluate on test set
    test_results = trainer.evaluate(test_data)

    save_predictions_regression(trainer, dev_data, test_data, final_output_dir, train_results, dev_results,
                                test_results)
    print(train_results)
    print(dev_results)


def save_predictions(trainer, dev_data, test_data, output_dir, train_results, dev_results, test_results):
    """Save the predictions and results on each dataset. This method is used with classification tasks"""
    dev_predictions = trainer.predict(dev_data)
    test_predictions = trainer.predict(test_data)
    # generate probabilities over classes and save the test data with predictions as a dataframe into split directory
    dev_data.dataset['predictions'] = F.softmax(torch.tensor(dev_predictions.predictions), dim=-1).tolist()
    dev_data.dataset.to_csv(f'{str(output_dir)}/dev_df_with_predictions.csv', index=False, sep="\t")
    test_data.dataset['predictions'] = F.softmax(torch.tensor(test_predictions.predictions), dim=-1).tolist()
    test_data.dataset.to_csv(f'{str(output_dir)}/test_df_with_predictions.csv', index=False, sep="\t")
    # save classification report for training,  validation and test set in split directory
    with open(f'{str(output_dir)}/train_report.csv', "w") as f:
        f.write(train_results["eval_report_csv"])
    with open(f'{str(output_dir)}/dev_report.csv', "w") as f:
        f.write(dev_results["eval_report_csv"])
    with open(f'{str(output_dir)}/test_report.csv', "w") as f:
        f.write(test_results["eval_report_csv"])


def save_predictions_regression(trainer, dev_data, test_data, output_dir, train_results, dev_results, test_results):
    """Save the predictions and results on each dataset. This method is used with regression tasks"""
    dev_predictions = trainer.predict(dev_data)
    test_predictions = trainer.predict(test_data)
    # generate probabilities over classes and save the test data with predictions as a dataframe into split directory
    dev_data.dataset['predictions'] = np.squeeze(dev_predictions.predictions, axis=1).tolist()
    dev_data.dataset.to_csv(f'{str(output_dir)}/dev_df_with_predictions.csv', index=False, sep="\t")
    test_data.dataset['predictions'] = np.squeeze(test_predictions.predictions, axis=1).tolist()
    test_data.dataset.to_csv(f'{str(output_dir)}/test_df_with_predictions.csv', index=False, sep="\t")
    # save classification report for training,  validation and test set in split directory
    with open(f'{str(output_dir)}/train_report.csv', "w") as f:
        f.write("MSE: %.3f CORR: %3f SIGN: %5f" % (
            train_results["eval_mse"], train_results["eval_pearsonr"], train_results["eval_pearsonp"]))
    with open(f'{str(output_dir)}/dev_report.csv', "w") as f:
        f.write("MSE: %.3f CORR: %3f SIGN: %5f" % (
            dev_results["eval_mse"], dev_results["eval_pearsonr"], dev_results["eval_pearsonp"]))
    with open(f'{str(output_dir)}/test_report.csv', "w") as f:
        f.write("MSE: %.3f CORR: %3f SIGN: %5f" % (
            test_results["eval_mse"], test_results["eval_pearsonr"], test_results["eval_pearsonp"]))


def compute_metrics_classification(pred: EvalPrediction):
    """Compute the metrics relevant for classification"""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1).flatten()
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='macro',
                                                                     zero_division=0)
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    report_dict = classification_report(y_true=labels, y_pred=preds, output_dict=True, zero_division=0)
    report_csv = classification_report(y_true=labels, y_pred=preds, zero_division=0)
    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        "report_dict": report_dict,
        "report_csv": report_csv
    }
    return results


def compute_metrics_regression(pred: EvalPrediction):
    """Compute the metrics relevant for regression, correlation and mean squared error"""
    labels = pred.label_ids
    preds = np.squeeze(pred.predictions, axis=1)
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    pearson, p = pearsonr(labels, preds)

    return {
        "mae": mae,
        "mse": mse,
        "pearsonr": pearson,
        "pearsonp": p
    }


if __name__ == '__main__':
    # read in arguments
    # model args: all classification details
    # data args: path to dataset etc.
    # training args: learning rate, optimizer etc.
    model_args, data_args, training_args = parse_arguments()
    # set a seed
    set_seed(training_args.seed)

    print("model is trained for %s " % data_args.label)
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
        config=config,
    )
    # Add a new adapter
    adapter_config = AdapterConfig.load("pfeiffer")
    print(adapter_config)
    model.add_adapter(model_args.adapter_name, config=adapter_config)

    # Add a matching classification head
    model.add_classification_head(
        model_args.adapter_name,
        num_labels=model_args.labels_num
    )
    # Activate the adapter
    model.train_adapter(model_args.adapter_name)

    if model_args.labels_num == 1:
        # this means it is a regression task

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
