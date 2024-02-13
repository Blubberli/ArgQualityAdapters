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
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoAdapterModel.from_pretrained(args.pretrained_model)
    pretrained_adapters = read_adapters(args.pretrained_adapters_file)
    adapter_counter = 0
    for i in range(len(pretrained_adapters)):
        name = pretrained_adapters.name.values[i]
        print("loading adapter %s as adapter %d" % (name, adapter_counter))
        path = pretrained_adapters.path.values[i]
        model.load_adapter(path, load_as=name, with_head=True)
        print("loaded adapter %s" % name)
        adapter_counter += 1
        # push the model to the hub
        model.push_adapter_to_hub(name, name, adapterhub_tag="argument/quality")
        print("pushed adapter %s to the hub" % name)
