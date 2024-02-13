import itertools

from transformers import AutoTokenizer, AutoConfig, AutoModelWithHeads, AdapterTrainer, AutoAdapterModel
from train_MTadapter import read_adapters
from args import parse_arguments
from tqdm import tqdm
from transformers import set_seed
import numpy as np
import os
import pickle
from utils import get_dynamic_adapter_fusion
from data import InferenceDataset
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import torch.nn.functional as F


def predict(dataloader, model, out_put_path, model_args, data_args):
    total_scores = defaultdict(list)
    file_name = data_args.data_dir.split("/")[-1].replace(".csv", "").replace(".tsv", "")
    probabilities = []
    # test set for predictions
    new_dataset = dataloader.dataset.dataset
    for id, batch in enumerate(tqdm(dataloader)):

        # do forward pass
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        output_adapter_fusion_attentions=True)
        # get predictions
        predictions = outputs.logits
        # covert to probabilities, depending on whether it is regression, binary or mulit-classification
        if model_args.labels_num == 1:
            scores = np.squeeze(predictions, axis=1).tolist()
        elif model_args.labels_num == 2:
            scores = torch.sigmoid(torch.tensor(predictions)).tolist()
        else:
            scores = F.softmax(torch.tensor(predictions), dim=-1).tolist()
        probabilities.append(scores)
        # retrieve attention scores and save them in a dictionary
        attention_scores = outputs.adapter_fusion_attentions
        for name, attention_dic in attention_scores.items():
            for layer_id, attention in attention_dic.items():
                # batchsize + seqlen + number of adapters in fusion
                attention_scores = attention["output_adapter"]
                #print("dimension of attention scores", attention_scores.shape)
                # average over the second dimension to get a matrix of batchsize x number of adapters (each intance in batch size has a score for each adapter)
                average_attention = attention_scores.mean(1)
                total_scores[layer_id].append(average_attention)
    # flatten the list of predicted probabilities
    probabilities = list(itertools.chain(*probabilities))
    final_scores = []
    # get the average attenweights across the adapters for each layer
    for layer, attention in total_scores.items():
        # flatten the list of attention scores, this will result in a list of lists
        attention = list(itertools.chain(*attention))
        # convert to numpy array so we get number of instances x number of adapters
        attention = np.array(attention)
        # append to the list so we collect for each layer
        final_scores.append(attention)
    # convert so we get number of layer x instances x num_adpaters
    final_scores = np.array(final_scores)
    # take the mean over layers so we get number of instances x number of adapters
    final_scores = final_scores.mean(0)
    print("final scores shape", final_scores.shape)

    if not os.path.isdir(out_put_path):
        os.makedirs(out_put_path)
        print("created folder : ", out_put_path)
    else:
        print(out_put_path, "folder already exists.")
    with open(out_put_path + "/%s_attention_scores.pkl" % file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(final_scores, filehandle)
    # add probabilities to dataset
    new_dataset[data_args.label] = probabilities
    new_dataset.to_csv(out_put_path + "/%s_predictions.csv" % file_name, sep="\t", index=False)
    return total_scores


if __name__ == '__main__':
    # read in arguments
    # model args: all classification details
    # data args: path to dataset etc.
    # training args: learning rate, optimizer etc.
    model_args, data_args, training_args = parse_arguments()
    print("model is loaded for %s " % data_args.label)
    set_seed(training_args.seed)

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # init a transformer config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=model_args.labels_num,
    )
    # create a model with adapter-specific head
    model = AutoAdapterModel.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )
    pretrained_adapters = read_adapters(model_args.pretrained_adapters_file)
    adapter_counter = 0

    for i in range(len(pretrained_adapters)):
        name = pretrained_adapters.name.values[i]
        path = pretrained_adapters.path.values[i]
        model.load_adapter(path, load_as="adapter%d" % adapter_counter, with_head=False)
        print("loading adapter %s as adapter adapter%d" % (name, adapter_counter))
        adapter_counter += 1
    print("number of adapters trained the fusion with is %d" % (adapter_counter))

    adapter_setup = get_dynamic_adapter_fusion(adapter_number=adapter_counter)
    # Load a fusion layer for all loaded adapters
    model.load_adapter_fusion(model_args.fusion_path, set_active=True)
    print("loaded adapter fusion")
    fusion_name = model_args.fusion_path.split("/")[-1]

    # load classification head
    model.load_head(
        model_args.adapter_name
    )
    print("loaded fusion trained classification head")

    print(type(model))
    print(model.adapter_summary())

    test = InferenceDataset(path_to_dataset=data_args.data_dir, tokenizer=tokenizer,
                            text_col=data_args.text_col)
    print("loaded test set from %s of size %d" % (data_args.data_dir, len(test)))
    # test
    dataloader = DataLoader(test, batch_size=training_args.per_device_eval_batch_size)
    predict(dataloader=dataloader, model=model, out_put_path=training_args.output_dir, model_args=model_args,
            data_args=data_args)
    print("generated all predictions and attention scores")
