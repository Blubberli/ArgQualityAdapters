## Bridging Argument Quality and Deliberative Quality Annotations with Adapters

This repository contains the code to train and evaluate adapters for argument quality and deliberative quality annotations.
It proveds single and multi-task adapters for the following tasks:

Argument Quality:
- quality (IBM-Rank-30k)
- clarity (SwanRank)
- cogency (GAQ)
- effectiveness (GAQ)
- reasonableness (GAQ)
- overall (GAQ)

Deliberative Quality:
- impact (Kialo)

(THB/BK):
- argumentative
- positive Emotion
- negative Emotion
- narrative
- question for justification
- empathy
- proposal
- reference to other arguments

(Europolis):
- justification
- common good
- respect
- storytelling
- interactivity

### Requirements
all requirements are listed in the requirements.txt file.

### Data
We provide the data for deliberative quality used in our experiments in the data folder. The data is to be used for research purposes only.
Europolis:
Each dimension is stored in a separate directory with
train, dev and test files. The label name equals to the directory name (e.g. argumentative, justification, ...).
The text used for Europolis is stored in *cleaned_comment*.

THF/BK:
The train/dev/test splits are stored in ````data/TBHBK````
the text for THF/BK in *EN_translation* and every dimension is stored in a separate column in each file.

We can provide the splits for the argument quality task on request after having received permission from the authors of the original datasets:
- [SwanRank](https://aclanthology.org/W15-4631/)
- [IBM-Rank-30k](https://ojs.aaai.org/index.php/AAAI/article/view/6285)
- [GAQ](https://aclanthology.org/2020.argmining-1.13/)
- [Kialo](https://aclanthology.org/D19-1568/)

### Training
To train a **single task adapter**, run the following command in the code folder:
```
sh trainST.sh
```
The script takes all possible hyperparameters as arguments. If no hyperparameters are given, the default values are used.
The arguments you need to specify are:
- *--data_dir*: path to the data directory
- *--label*: name of the label to train on (aka dimension)
- *--labels_num*: number of labels for the given dimension (e.g. 2 for binary labels, 1 for pointwise labels)
- *--output_dir*: path to the output directory (this will be created if it does not exist but it won't be overwritten if it exists)
- *--model_name_or_path* (optional): path to the pretrained LM to use as backbone (default: roberta-base)
- *--adapter_name*: the name that the new adapter will be saved under

To train **adapter-fusion** you need to have some pretrained adapters. You can either train them yourself or use the ones we provide in the pretrained folder.
To train adapter-fusion, run the following command:
```sh trainMT.sh```
The script takes all possible hyperparameters as arguments. If no hyperparameters are given, the default values are used.
The arguments you need to specify are as above but additionally you need to specify:
- *--pretrained_adapters_file*: path to the file containing the paths to the pretrained adapters. The file should contain one path per line and each line should contain the name and path to the pretrained adapter for one dimension.
We provide the file for the adapters we trained on all dimensions. The file that contains all single-task adapters with corresponding path
is located at ```code/adapterPaths/pretrainedAdapters.tsv```

### Inference

For simplified inference, we uploaded the adapters to the adapter hub, such they can be loaded and used like any other available adapter.
We provide two scripts to run the adapters (model of one seed per quality dimension, each dimension has the model that was trained on the largest available dataset of the corresponding quality dimension) in /code.
For example, you can run a single quality dimension prediction on a dataset of your choice with the following command:
```
python inference_simple.py input_file.csv text_column batch_size overall output_file.csv
```
In this case it generates predictions for the dimension *overall* (see the following table for availbale task identifiers (adapters)):
| quality dimension | task (identifier) | 
| -------- | -------- |
| quality   | as left col   |
| clarity   | as left col   |
| cogency   | as left col   |
| effectiveness   | as left col   |
| reasonableness   | as left col   |
| overall   | as left col   |
| impact   | as left col   |
| argumentative   | as left col   |
| positive Emotion   | posEmotion   |
| negative Emotion   | negEmotion   |
| narrative   | as left col   |
| question for justification   | QforJustification   |
| empathy   | as left col   |
| proposal   | as left col   |
| reference to other arguments   | reference   |
| justification   | as left col   |
| common good   | cgood   |
| respect   | as left col   |
| storytelling   | story   |
| interactivity   | as left col   |

To run all available adapters on your dataset run:
```
python inference_parallel.py input_file.csv text_column batch_size output_file.csv
```
Note however that this will take some time.

To run all models used in the paper, we provide all single-task adapters (for each dimension) and all fusion models. 
The single-task adapters are compressed into ```compressed_models/AdapterModels.tar.gz``` and the fusion models are compressed into ```compressed_models/FusionModels_*.tar.gz```.
First you need to extract the models. To do so, run the following command in the code folder:
```
tar -xvf AdapterModels.tar.gz 
``` 

You can also load pretrained ST adapters and run **parallel inference** on new data. To do so, run the following command:
```
python inference_parallel.py adapterPaths/pretrainedAdapters.tsv roberta-base data/europolis/respect/test.csv cleaned_comment 16 42 inferenceEuropolis
```
This runs inference on the test set of the respect dimension of the Europolis dataset using all pretrained adapters. The output is stored in the inferenceEuropolis folder and 
in form of probabilities for each label or the pointwise label for regression tasks.

Finally you can also load an adapter-fusion for inference and at the same time extract the attention weights for the adapters.
We provide two example scripts for the **fusion-all** for moderation and the **fusion-all** for **argumentative*.
Each fusion model was trained with different adapters. We provide all fusion models for all setups in ```FusionModels```.
We provide the corresponding tsv files to load the adapters in ```adapterPaths```.
You can run, for example the following command for fusion-all for moderation:
```
sh inferenceModeration.sh
```
Three additional arguments are important:
- *--adapter_name*: the name of the head of the trained adapter-fusion. E.g. for a fusion trained on effectiveness based on only AQ dimensions without its own adapter this is stored in 
```../FusionModels/AQcorr/effectiveness/checkpoint-98/effectiveness```"
- *--pretrained_adapters_file*: path to the file containing the paths to the pretrained adapters for a given adapter-fusion. The file should contain one path per line and each line should contain the name and path to the pretrained adapter for one dimension.
It should only contain the adapters that have been used in the adapter-fusion. E.g. for the above scenario the corresponding file is located at
```../adapterPaths/AQcorrAdapters/effectiveness.csv```
- *--fusion_path*: the path to the fusion-model. It is in the same directory as the head but it has the list of 
adapters as name. E.g. for the above scenario the corresponding path is located at ```../FusionModels/AQcorr/effectiveness/checkpoint-98/adapter0,adapter1,adapter2,adapter3```

### References

If you use any of the data provided in this repository, please cite the following papers:

- Katharina Esau. 2022. **Kommunikationsformen und Deliberationsdynamik.** *Nomos* Verlagsgesellschaft mbH & Co. KG.
- Marlène Gerber, André Bächtiger, Susumu Shikano, Simon Reber, and Samuel Rohr. 2018. **Deliberative abilities and influence in a transnational deliberative poll (europolis).** *British Journal of Political Science*, 48(4):1093–1118.
