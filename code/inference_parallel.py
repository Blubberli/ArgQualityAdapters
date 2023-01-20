from transformers import AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterTrainer
from train_MTadapter import read_adapters
from data import InferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
import argparse
from transformers import set_seed
from utils import get_dynamic_parallel


def predict(dataloader, model, out_put_path, task2label, pretrained_adapters, dataset):
    output_dic = {}
    for name in pretrained_adapters.name.values:
        output_dic[name] = []
    for id, batch in enumerate(tqdm(dataloader)):
        # print(batch)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # print(outputs)
        for i in range(len(outputs)):
            task = pretrained_adapters.name.values[i]
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
        dataset[task] = predictions
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
              "respect": 3,
              "moderation": 2}

if __name__ == '__main__':
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_adapters_file', type=str,
                        help='path to the pretrained adapter file')
    parser.add_argument("pretrained_model", type=str, help="pretrained LM identifier")
    parser.add_argument('testdata', type=str,
                        help='path to the test data')
    parser.add_argument('text_col', type=str, help="column name of text column")
    parser.add_argument('batch_size', type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("outputpath")
    args = parser.parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoAdapterModel.from_pretrained(args.pretrained_model)
    pretrained_adapters = read_adapters(args.pretrained_adapters_file)
    adapter_counter = 0
    for i in range(len(pretrained_adapters)):
        name = pretrained_adapters.name.values[i]
        print("loading adapter %s as adapter %d" % (name, adapter_counter))
        path = pretrained_adapters.path.values[i]
        model.load_adapter(path, load_as="adapter%d" % adapter_counter, with_head=True)
        adapter_counter += 1
    #model.set_active_adapters([i for i in range(adapter_counter)])
    print("loaded %d adapters" % adapter_counter)
    adapter_setup = get_dynamic_parallel(adapter_number=adapter_counter)
    model.active_adapters = adapter_setup
    test = InferenceDataset(path_to_dataset=args.testdata, tokenizer=tokenizer, text_col=args.text_col)
    dataloader = DataLoader(test, batch_size=args.batch_size)
    predict(dataloader=dataloader, model=model, dataset=test.dataset, pretrained_adapters=pretrained_adapters,
            out_put_path=args.outputpath, task2label=task2label)
