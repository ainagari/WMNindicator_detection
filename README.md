# ☝🏻 Toward the Automatic Detection of Word Meaning Negotiation Indicators in Conversation 💬

This repository will contain code for the upcoming paper:

Aina Garí Soler, Matthieu Labeau and Chloé Clavel (2025). Toward the Automatic Detection of Word Meaning Negotiation Indicators in Conversation. Accepted at Findings of the Association for Computational Linguistics: EMNLP 2025.


## Data

The NeWMe corpus can be obtained from [here](https://github.com/gu-wmn/webapp/tree/main). This link contains the standoff annotations and code to download the corresponding corpora.

### COMING SOON (expected: end of November 2025) 
We will soon provide (1) the full code to derive the Indicators dataset from the annotations available at the link above, including some postprocessing and cleaning; as well as (2) the data used to train the models for domain adaptation with an mlm objective and (3) the contrastive pairs used for the development of the model trained in a contrastive setting. 

In the meantime, in `dataset_ids/`, you can find the ids of NeWMe instances and the subset of the Indicators dataset they belong to (train-rd, train-rx, dev or test).


## Code

The fine-tuning experiments were run using the libraries specified in `requirements.txt`.


### Supervised Models

To **fine-tune of a model, or to train it in a contrastive setting**, you can run `finetuning.py` with the arguments described below. The script will run training and save the checkpoints together with their dev results at every epoch. It will also select the epoch with the best dev results and save its predictions on the dev and test sets.

**Data-related arguments**
* `--regex_aware`: add this option to train models on the RX dataset, or omit it for the RD dataset
* `--corpus`: the corpus on which to run training. Options: 'swda','Reddit','BNC','all'. By default, it is "all".
* `--base_dir`: base directory where the trained models and results will be stored. If --model_path is provided, that will be the base_dir.
* `--context`: The kind of context to be used for indicators: '3past', '1p1f', 'no_context' (for utterances), 'sentenceonly'

**Model-related arguments**
* `--model_name`: FacebookAI/roberta-base, FacebookAI/roberta-large, MingZhong/DialogLED-large-5120, MingZhong/DialogLED-base-16384, vinai/bertweet-base
* `--model_path`: if you are fine-tuning a model after running contrastive training or domain adaptation, provide its path instead of --model_name. 
* `--sensitive_arguments`: add this option if you want the script to automatically infer the context type and dataset to use from the provided model_path. The epoch with the best dev set result will automatically be selected.

**Training-related arguments**
* `--learning_rate`
* `--contrastive`: if you want to run the contrastive training step, previous to the general fine-tuning.
* `--use_cpu`
* `--num_train_epochs` (default: 5)
* `--train_batch_size` (default: 8)
* `--seed` (default: 9)


To train a model for **domain adaptation with an mlm objective**, you can use the script `mlm.py`. You can use the arguments `--model_name`, `--base_dir`, `--learning_rate`,`--use_cpu`,`--num_training_epochs`,`--train_batch_size` as described above.


### LLMs

The LLM experiments can be reproduced with the script `llm_calls.py` under `LLMs/`, to be called with the following arguments:

* `--model`: one of ['olmo','llama3B','llama70B'].
* `--llama_token`: if using Llama-3.2-3B-Instruct, you can request your token [here](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
* `--llama70B_path`: provide the local path to llama70B snapshots, e.g. "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.3-70b-Instruct/snapshots/"
* `--path_to_data`: the path to the Indicators dataset
* `--setup`: one of [DEV-A, DEV-B, TEST-BEST]. "DEV-A" runs all prompts with one set of examples (A) on the development set. "DEV-B" is model dependent: it runs the best prompt obtained after DEV-A for a given model but with set of examples B and in a zero-shot setting. "TEST-BEST" is also model-dependent, it runs the overall best prompt on the test set. (The choice of prompts for DEV-B and TEST-BEST is not done automatically. The information was manually inserted into the script after inspection of DEV-A and DEV-B results).

For example:

`python llm_calls.py --setup DEV-A --model olmo --path_to_data indicators_dataset`

The script will save all model outputs and it will calculate multiple evaluation metrics and print their results.


## Contact

For any questions or requests feel free to [contact me](https://ainagari.github.io/menu/contact.html).


