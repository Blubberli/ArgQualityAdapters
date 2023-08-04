from transformers import AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterTrainer
from data import InferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
import argparse
from utils import get_dynamic_parallel


def predict(dataloader, model, out_put_path, task2label, dataset, task2identifier):
    output_dic = {}
    for k, v in task2identifier.items():
        output_dic[k] = []
    for id, batch in enumerate(tqdm(dataloader)):
        # print(batch)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # print(outputs)
        for i in range(len(outputs)):
            task = list(task2identifier.keys())[i]
            predictions = outputs[i].logits
            label_num = task2label[task]
            if label_num == 1:
                scores = np.squeeze(predictions, axis=1).tolist()
                for el in scores:
                    output_dic[task].append(el)
            elif label_num == 2:
                probs = torch.sigmoid(torch.tensor(predictions)).tolist()
                for el in probs:
                    output_dic[task].append(el)
            else:
                probs = F.softmax(torch.tensor(predictions), dim=-1).tolist()
                for el in probs:
                    output_dic[task].append(el)
    for task, predictions in output_dic.items():
        label_num = task2label[task]
        if label_num == 1:
            dataset[task] = predictions
        elif label_num == 2:
            dataset[task] = [el[1] for el in predictions]
        else:
            for i in range(label_num):
                # get list of elements at index i
                dataset[f"{task}_{i}"] = [el[i] for el in predictions]
    dataset.to_csv(out_put_path, sep="\t", index=False)


task2label = {"reasonableness": 1,
              "effectiveness": 1,
              "overall": 1,
              "impact": 3,
              "quality": 1,
              "clarity": 1,
              "justification": 4,
              "interactivity": 4,
              "cgood": 3,
              "story": 2,
              "reference": 2,
              "posEmotion": 2,
              "negEmotion": 2,
              "empathy": 2,
              "argumentative": 2,
              "narration": 2,
              "proposal": 2,
              "QforJustification": 2,
              "cogency": 1,
              "respect": 3}

task2identifier = {"reasonableness": "falkne/reasonableness",
                   "effectiveness": "falkne/effectiveness",
                   "overall": "falkne/overall",
                   "impact": "falkne/impact",
                   "quality": "falkne/ibm_rank",
                   "clarity": "falkne/clarity",
                   "justification": "falkne/justification",
                   "interactivity": "falkne/interactivity",
                   "cgood": "falkne/cgood",
                   "story": "falkne/story",
                   "reference": "falkne/reflexivity",
                   "posEmotion": "falkne/posEmotion",
                   "negEmotion": "falkne/negEmotion",
                   "empathy": "falkne/empathie",
                   "argumentative": "falkne/argumentative",
                   "narration": "falkne/narration",
                   "proposal": "falkne/proposal",
                   "QforJustification": "falkne/QforJustification",
                   "cogency": "falkne/cogency",
                   "respect": "falkne/respect"}

if __name__ == '__main__':
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('testdata', type=str,
                        help='path to the test data')
    parser.add_argument('text_col', type=str, help="column name of text column")
    parser.add_argument('batch_size', type=int)
    parser.add_argument("output_path", type=str, help="path to output file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoAdapterModel.from_pretrained("roberta-base")
    adapter_counter = 0

    for k, v in task2identifier.items():
        print("loading adapter %s as adapter %d" % (k, adapter_counter))
        model.load_adapter(v, load_as="adapter%d" % adapter_counter, with_head=True,
                           set_active=True, source="hf")
        adapter_counter += 1

    # model.set_active_adapters([i for i in range(adapter_counter)])
    print("loaded %d adapters" % adapter_counter)
    adapter_setup = get_dynamic_parallel(adapter_number=adapter_counter)
    model.active_adapters = adapter_setup
    test = InferenceDataset(path_to_dataset=args.testdata, tokenizer=tokenizer, text_col=args.text_col)
    dataloader = DataLoader(test, batch_size=args.batch_size)
    predict(dataloader=dataloader, model=model, dataset=test.dataset,
            out_put_path=args.output_path, task2label=task2label, task2identifier=task2identifier)
