'''

IN PROCESS!! ADAPTING FROM model_training.py....
use with indicators environment

Make sure it works with all the models of interest...


Main steps:

1. Read in config options
2. Instantiate model and tokenizer
3. Load the data with the desired specifications
4. Prepare trainer
5. Run training
6. Store predictions and config information


'''

import math
import transformers
from ledmodel import ModifiedLEDForMLM
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM

from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import json

from transformers import DataCollatorForLanguageModeling


import torch

from argparse import ArgumentParser

transformers.set_seed(9)


def load_mlm_data(debugging_mode=False):
    '''problem_definition can be cc (corpus centric) or pp (ppwu centric)
    regex awereness can be True or false
    simnwt can be negsimnwt or nosimnwt'''
    fn = "mlm_dataset.json"
    with open(fn) as f:
        data = json.load(f)

    pairs = [p[0] for p in data] # dont take the id

    if debugging_mode:
        pairs = pairs[:10]

    return pairs



class IndicatorDatasetForMLM(Dataset):
    def __init__(self, instances, tokenizer):
        self.data = []
        for ins in instances:
            tokenizer.truncation_side = "right"
            full_text = ins
            encoded = tokenizer(full_text, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
            if encoded['input_ids'].shape[1] > tokenizer.model_max_length:
                print("Size of input ids depasses tokenizer's max length")
            for k in encoded:
                encoded[k] = encoded[k][0]  # first batch element
            self.data.append(encoded)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]  # , self.labels[idx]


def save_perplexity(ppl):
    fn = args.out_dir + "/results/ppl.txt"

    with open(fn, "w") as out:
       out.write(str(ppl))


if __name__ == "__main__":
    parser = ArgumentParser()
    ## dataset-related arguments
    # it doesn't matter if it is regexaware or not. I will run it on the nonregexaware though.

    ## model- and training-related arguments
    parser.add_argument("--model_name", default="distilbert/distilbert-base-uncased",
                        choices=["FacebookAI/roberta-base", "FacebookAI/roberta-large",
                                 "distilbert/distilbert-base-uncased", "MingZhong/DialogLED-large-5120","MingZhong/DialogLED-base-16384","vinai/bertweet-base","microsoft/deberta-v3-large"]) #, required=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--use_cpu", action='store_true')
    parser.add_argument("--num_train_epochs", default=2, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--base_dir", default="domain_adapted/", type=str,
                        help="base directory where the trained models and results will be stored")

    parser.add_argument("--debugging_mode", action="store_true")

    args = parser.parse_args()
    torch.cuda.empty_cache()

    ################## CHECKING ARGUMENTS ###################
    # check that argument combinations make sense
    if args.debugging_mode:
        args.use_cpu = True
        args.num_train_epochs = 1
        args.train_batch_size = 4
        args.model_name = "distilbert/distilbert-base-uncased"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    ################################################################

    args.out_dir = args.base_dir + args.model_name

    # Load dataset
    dataset = load_mlm_data(debugging_mode=args.debugging_mode)

    # Instantiate tokenizer
    if "roberta" in args.model_name:
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForMaskedLM
        safe = True
    elif "LED" in args.model_name:
        tokenizer_class = AutoTokenizer
        model_class = ModifiedLEDForMLM # For prediction, run the conditionalgeneration one
        safe = False
    else:
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForMaskedLM
        safe = True

    tokenizer = tokenizer_class.from_pretrained(args.model_name, add_prefix_space=True)

    model = model_class.from_pretrained(args.model_name)


    if "DialogLED" in args.model_name:
        tokenizer.model_max_length = 1024
    elif "tweet" in args.model_name:
        tokenizer.model_max_length = model.config.max_position_embeddings - 2


    # Create a Pytorch dataset out of it with the specified parameters, and prepare a datacollator
    pt_dataset = dict()
    pt_dataset = IndicatorDatasetForMLM(dataset, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, return_tensors="pt")

    model.to(device)

    # warmup steps: https://arxiv.org/pdf/2104.07705
    training_args = TrainingArguments(
        output_dir=args.out_dir + '/results',  # output directory
        num_train_epochs=args.num_train_epochs,  # total # of training epochs
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=4,  # batch size for evaluation
        warmup_ratio=0.02,
        weight_decay=0.01,  # strength of weight decay
        logging_dir=args.out_dir + '/logs',  # directory for storing logs
        use_cpu=args.use_cpu,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        eval_strategy="epoch",
        do_eval=True,
        save_safetensors=safe,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pt_dataset,
        eval_dataset=pt_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("cuda is available:", torch.cuda.is_available())
    trainer.train()

    eval_results = trainer.evaluate()

    ppl = math.exp(eval_results['eval_loss'])
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    save_perplexity(ppl)



