
import transformers
from ledmodel import ModifiedLEDForSequenceClassification, ModifiedTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification
from transformers import DataCollatorWithPadding, DefaultDataCollator
from transformers import Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from sentence_transformers.losses import BatchAllTripletLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import models
import json
import os
import numpy as np
import torch
import evaluate
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import sys
import datetime
import pandas as pd
from datasets import DatasetInfo

from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction


from utils import load_dataset, determine_usernames_for_instance


class IndicatorDataset(Dataset):
    def __init__(self, instances, tokenizer, context_type, add_usernames, contrastive=False):
        if "DialogLED" in modelnametouse or "roberta" in modelnametouse:
            prefix = "Ġ"
        elif "tweet" in modelnametouse:
            prefix = "@@"

        else:
            prefix = tokenizer.decoder.prefix

        if contrastive:
            self.info = DatasetInfo(dataset_name="indicators")
            self.column_names = ["sentence_0_input_ids", "label"]


        include_context = False if context_type == "no_context" else True
        max_length = tokenizer.model_max_length - 3

        self.data = []
        self.labels = []
        for ins in instances:

            tokenizer.truncation_side = "right"

            # first, tokenize indicator utterance.
            if include_context == False:
                encoded = tokenizer(ins['target']['text'], truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
            elif context_type == "sentenceonly":
                encoded = tokenizer(ins['target']['sentence'], truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
            else:
                if add_usernames:
                    # determine the username of each message. Starting from the beginning, S1, S2, S3...
                    username_id_mapping = determine_usernames_for_instance(ins)
                    indicator = username_id_mapping[ins['target']['author']] + ": " + ins['target']['text']

                else:
                    indicator = ins['target']['text']

                tokenized_indicator = tokenizer.tokenize(indicator, truncation=True, max_length=max_length)


                #if include_sep:
                #    what_can_we_fit = max(0, tokenizer.model_max_length - len
                #        (tokenized_indicator) - 3)  # for the 3 special tokens: cls and 2 sep
                #else:
                what_can_we_fit = max(0, tokenizer.model_max_length - len
                    (tokenized_indicator) - 2)  # for the 2 only special token: cls and one sep

                if what_can_we_fit == 0:
                    encoded = tokenizer(indicator, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')

                else:
                    # include context, truncating from the beginning...
                    if context_type == "3past":
                        candidate_contexts = ins['past_context']
                    elif context_type in ["1past" ,"1p1f"]:
                        candidate_contexts = [ins['past_context'][-1]]

                    concatenated_context = ''
                    for ctxt in candidate_contexts:
                        if add_usernames:
                            concatenated_context += username_id_mapping[ctxt['author']] +  ": " + ctxt['text'] + " "
                        elif not add_usernames:
                            concatenated_context += ctxt['text'] + " "

                    # we truncate from the left if necessary
                    tokenizer.truncation_side = "left"
                    full_string = concatenated_context + " " + indicator


                    encoded = tokenizer(full_string, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')

                    # If context_type is 1p1f and future context still fits, include it, but truncating on the right

                    if context_type == "1p1f":
                        if tokenizer.model_max_length - encoded['input_ids'].shape[1] > 0:
                            if ins['future_context']:
                                future_context = ins['future_context'][0]
                            else:
                                future_context = ''
                            if add_usernames and future_context:
                                future_context = username_id_mapping[future_context['author']] +  ": " + future_context['text'] + " "
                            elif future_context:
                                future_context = future_context['text']
                            detokenized_string = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])[1:-1] # remove first and last tokens. they will be added again...
                            detokenized_string = [token.strip(prefix) for token in detokenized_string]
                            new_string = detokenized_string + [token.strip(prefix) for token in tokenizer.tokenize(future_context)]

                            # now truncate on the right!
                            tokenizer.truncation_side = "right"
                            encoded = tokenizer(new_string, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt' ,is_split_into_words=True)

                if encoded['input_ids'].shape[1] > tokenizer.model_max_length:
                    print("Size of input ids depasses tokenizer's max length")

            for k in encoded:
                encoded[k] = encoded[k][0]

            if contrastive:
                if "input_ids" in encoded:
                    encoded["sentence_0_input_ids"] = encoded["input_ids"]
                del encoded["input_ids"]
                encoded['sentence_0_attention_mask'] = encoded['attention_mask']
                del encoded['attention_mask']
                if len(encoded["sentence_0_input_ids"]) < tokenizer.model_max_length:
                    remaining_length = tokenizer.model_max_length - len(encoded["sentence_0_input_ids"])
                    encoded["sentence_0_input_ids"] = torch.cat((encoded["sentence_0_input_ids"], torch.tensor([pad_id] * remaining_length)))
                    encoded["sentence_0_attention_mask"] = torch.cat((encoded["sentence_0_attention_mask"], torch.tensor([0] * remaining_length)))

            encoded['label'] = ins['label']
            self.data.append(encoded)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)  # because classification and not regression
    result = metrics.compute(predictions=preds, references=p.label_ids)
    return result

def save_predictions(predictions, subset):
    pred_dir = args.out_dir + "/predictions/"
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    fn = subset + ".txt"
    with open(pred_dir + fn, "w") as out:
        for p in predictions:
            out.write(str(int(p)) + "\n")


def load_contrastive_pairs(regex_aware, pt_dataset):
    s = "regex" if regex_aware else "random"
    pair_idcs = {'train':[], 'dev':[]}

    pt = [l.strip().split("\t") for l in open("data/contrastive_pairs_" + s + ".tsv").readlines()]
    pair_idcs['train'] = [(int(x), int(y)) for x, y in pt]

    pd = [l.strip().split("\t") for l in open("data/contrastive_pairs_dev.tsv").readlines()]
    pair_idcs['dev'] = [(int(x), int(y)) for x, y in pd]

    # Now select the pairs from the dataset
    pairs = dict()
    for subset in pair_idcs:
        pairs[subset] = []
        for x, y in pair_idcs[subset]:
            try:
                p1, p2 = pt_dataset[subset][x], pt_dataset[subset][y]
                label = 1 if p1['label'] == p2['label'] == 1 else 0
                new_pair = {'sentence_0_input_ids': p1['sentence_0_input_ids'], 'sentence_1_input_ids': p2['sentence_0_input_ids'], 'label': label}
                new_pair['sentence_0_attention_mask'] = p1['sentence_0_attention_mask']
                new_pair['sentence_1_attention_mask'] = p2['sentence_0_attention_mask']
                pairs[subset].append(new_pair)
            except IndexError: # if in debugging mode
                continue

    return pairs

def get_sentences_from_padded_ids(dataset):

    sentences = dict()

    for k in ['sentence_0_input_ids', 'sentence_1_input_ids']:
        sentences[k] = []
        for input_ids in dataset[k]:
            unpadded_list = [i for i in input_ids if i != pad_id]
            decoded = tokenizer.decode(unpadded_list)
            sentences[k].append(decoded)
    return sentences



#### TODO this is for contrastive, polish
def select_best_model_from_path(path): # similar to code later but path is not args.out_path and also we check the binary results dont we??
    dev_results = pd.read_csv(path + "/results/eval/binary_classification_evaluation_indicators_dev_results.csv")
    # pick the row with highest cosine_f1 and then pick its epoch. and then find and load the corresponding model
    idxbestrow = dev_results['cosine_f1'].idxmax()
    best_epoch = dev_results.iloc[idxbestrow]['epoch']
    for checkpoint in os.listdir(path + '/results/'):
        if "checkpoint" not in checkpoint:
            continue
        trainer_state = json.load(open(path + '/results/' + checkpoint + "/trainer_state.json"))
        current_epoch = trainer_state['log_history'][-1]['epoch']
        if int(current_epoch) == int(best_epoch):
            path_to_chosen_model = path + '/results/' + checkpoint + "/"

    return path_to_chosen_model



if __name__ == "__main__":
    parser = ArgumentParser()
    ## dataset-related arguments
    parser.add_argument("--regex_aware", action='store_true')
    parser.add_argument("--corpus", default='all', help="the corpus on which to run training", choices=['swda' ,'Reddit' ,'BNC','all'])
    parser.add_argument("--base_dir", default="Results/", type=str,
                        help="base directory where the trained models and results will be stored. If --model_path is provided, that will be the base_dir too")
    parser.add_argument("--context", default="no_context", choices=['no_context', '3past', '1past', '1p1f',"sentenceonly"])

    ## model- -related arguments
    parser.add_argument("--model_name", choices=["FacebookAI/roberta-base" ,"FacebookAI/roberta-large", "distilbert/distilbert-base-uncased" ,"MingZhong/DialogLED-large-5120"
                                                 ,"MingZhong/DialogLED-base-16384" ,"vinai/bertweet-base"])
    parser.add_argument("--model_path", help="for models trained with an mlm or contrastive objective, provide the path to where they are stored.")
    parser.add_argument("--sensitive_arguments", action="store_true",
                        help="whether the arguments of this script should be chosen based on the model path (important for contrastive models)")


    # training-related arguments
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--contrastive", action='store_true',help="whether we run contrastive training")
    parser.add_argument("--use_cpu", action='store_true')
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--seed", default=9, type=int)



    ## input-related arguments
    parser.add_argument("--debugging_mode", action="store_true")

    args = parser.parse_args()
    torch.cuda.empty_cache()

    transformers.set_seed(9)


    ################### CHECKING INPUT ARGUMENTS ####################
    # check that argument combinations make sense
    if args.add_usernames and not args.context:
        print("Incompatible argument combination")
        sys.exit()
    if args.debugging_mode:
        args.use_cpu = True
        args.num_train_epochs = 1
        args.context = "no_context"
        args.add_usernames = True
        args.train_batch_size = 4
        # using distilbert for debugging certain things
        args.model_name = "distilbert/distilbert-base-uncased"

    if args.sensitive_arguments:
        args.add_usernames = True
        _, ctx, reg, _ = args.model_path.strip("/").split("/")[-1].split("_")
        if ctx == "noctxt":
            args.context = "no_context"
        else:
            args.context = ctx
        if reg == "regex":
            args.regex_aware = True
        else:
            args.regex_aware = False
        args.model_name = select_best_model_from_path(args.model_path)

    if args.model_name and args.model_path:
        sys.exit("Provide only a model name or a path, not both!")



    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    args.add_usernames = True # to simplify things
    ################################################################

    if args.context == "no_context":
        ctxt_str = "noctxt"
    elif args.context == "3past":
        ctxt_str = "ctxt"
    else:
        ctxt_str = args.context


    regex_str = "regex" if args.regex_aware else "noregex"
    corpus_str = "trainedon" + args.corpus
    lr_str = str(args.learning_rate)
    now = datetime.datetime.now()

    base_dir = args.base_dir

    if args.model_path:
        base_dir = args.model_path
        out_dir = base_dir + "/" + ctxt_str + "_" + regex_str + "_" + lr_str + "/finetuning"

    else:
        out_dir = base_dir + args.model_name + "_" + ctxt_str + "_" + regex_str + "_" + lr_str

        if args.corpus == "all":
            args.out_dir = base_dir + args.model_name + "_" + ctxt_str + "_" + regex_str + "_" + lr_str
        else:
            args.out_dir = base_dir + "isolated_corpora/" + args.model_name + "_" + ctxt_str + "_" + regex_str + "_" + lr_str + "_" + corpus_str




    # Load dataset
    dataset = load_dataset(regex_awareness=args.regex_aware, debugging_mode=args.debugging_mode, corpus=args.corpus)

    # Instantiate tokenizer

    modelnametouse = args.model_name if args.model_name else args.model_path

    if not args.contrastive:
        if "roberta" in modelnametouse:
            tokenizer_class = RobertaTokenizer
            model_class = RobertaForSequenceClassification
        elif "LED" in modelnametouse:
            tokenizer_class = AutoTokenizer
            model_class = ModifiedLEDForSequenceClassification
        else:
            tokenizer_class = AutoTokenizer
            model_class = AutoModelForSequenceClassification
    elif args.contrastive:
        model_class = SentenceTransformer
        if "roberta" in args.model_name:
            tokenizer_class = RobertaTokenizer
        else:
            tokenizer_class = AutoTokenizer

    tokenizer = tokenizer_class.from_pretrained(modelnametouse, add_prefix_space=True)

    # Instantiate model
    id2label = {0: "No indicator", 1: "Indicator"}
    label2id = {v: k for k, v in id2label.items()}


    if not args.contrastive:
        model = model_class.from_pretrained(modelnametouse, num_labels=2, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    elif args.contrastive:
       if "DialogLED" in args.model_name:
           transformer = ModifiedTransformer(args.model_name)  # LEDModel.from_pretrained(args.model_name)
       else:
            transformer = models.Transformer(args.model_name)
       embedding_dimension = transformer.get_word_embedding_dimension()
       pooling = models.Pooling(word_embedding_dimension=embedding_dimension, pooling_mode="cls")
       model = model_class(modules=[transformer, pooling])

       sampler = BatchSamplers.BATCH_SAMPLER

       if "distil" in args.model_name or "DialogLED" in args.model_name:
           v = tokenizer.vocab
       else:
           v = tokenizer.encoder

       pad_id = v[tokenizer.special_tokens_map['pad_token']]

    if "DialogLED" in modelnametouse:
        tokenizer.model_max_length = 1024
    elif "tweet" in modelnametouse:
        tokenizer.model_max_length = model.config.max_position_embeddings - 2


    # Create a Pytorch dataset out of it with the specified parameters, and prepare a datacollator
    pt_dataset = dict()
    for subset in dataset: # train, dev, test
        pt_dataset[subset] = IndicatorDataset(dataset[subset], tokenizer, context_type=args.context, add_usernames=args.add_usernames, contrastive=args.contrastive)



    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.contrastive:
        data_collator = DefaultDataCollator()
        data_collator.valid_label_columns = ['label']
        pairs_pt_dataset = load_contrastive_pairs(args.regex_aware, pt_dataset)
        hf_dataset = dict()
        hf_dataset['train'] = HFDataset.from_list(pt_dataset['train'])
        hf_dataset['dev'] = HFDataset.from_list(pairs_pt_dataset['dev'])  # The dev evaluation is done with pairs so we could better compare among the different configurations we tried (different losses)
        tokenizer.model_input_names = ["sentence_0_input_ids", "sentence_0_attention_mask"]

    model.to(device)

    # warmup steps: https://arxiv.org/pdf/2104.07705
    if not args.contrastive:
        training_args = TrainingArguments(
            output_dir= out_dir + '/results',  # output directory
            num_train_epochs=args.num_train_epochs,  # total # of training epochs
            per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=4,  # batch size for evaluation
            warmup_ratio=0.02,
            weight_decay=0.01,  # strength of weight decay
            logging_dir= out_dir +'/logs',  # directory for storing logs
            use_cpu=args.use_cpu,
            learning_rate=args.learning_rate,
            save_strategy="epoch",
            eval_strategy="epoch",
            do_eval=True
        )

    elif args.contrastive:
        training_args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=args.out_dir + '/results',
            # Optional training parameters:
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=4,
            warmup_ratio=0.02,
            weight_decay=0.01,
            logging_dir=args.out_dir + '/logs',
            use_cpu=args.use_cpu,
            learning_rate=args.learning_rate,
            # fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            # bf16=False,  # Set to True if you have a GPU that supports BF16
            batch_sampler=sampler,
            # Optional tracking/debugging parameters:
            eval_strategy="epoch",
            save_strategy="epoch",
            do_eval=True
        )

    metrics = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])

    if not args.contrastive:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=pt_dataset['train'],
            eval_dataset=pt_dataset['dev'],
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    elif args.contrastive:
        loss = BatchAllTripletLoss(model, distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance)
        dev_sentences = get_sentences_from_padded_ids(hf_dataset['dev'])
        dev_evaluator = BinaryClassificationEvaluator(
            sentences1=dev_sentences['sentence_0_input_ids'],
            sentences2=dev_sentences['sentence_1_input_ids'],
            labels=hf_dataset['dev']["label"],
            name="indicators_dev")

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset['train'],  # changed to HF
            data_collator=data_collator,
            loss=loss,
            compute_metrics=compute_metrics,
            evaluator=dev_evaluator
        )

    print("cuda is available:", torch.cuda.is_available())
    trainer.train()


    # Select model from the epoch that got the highest f1 score on the dev set,
    # and load that one
    evalkey = "eval_indicators_dev_cosine_f1" if args.contrastive else "eval_f1"

    previous_best_f1 = -1
    best_epoch = -1
    for checkpoint in os.listdir(args.out_dir + '/results/'):
        if "checkpoint" not in checkpoint:
            continue
        trainer_state = json.load(open(args.out_dir + '/results/' + checkpoint + "/trainer_state.json"))
        current_f1 = trainer_state['log_history'][-1][evalkey]
        if current_f1 >= previous_best_f1:  # take the later epoch if results are the same.
            path_to_chosen_model = args.out_dir + '/results/' + checkpoint + "/"
            best_epoch = trainer_state['log_history'][-1]['epoch']
            previous_best_f1 = current_f1
    model_to_eval = model_class.from_pretrained(path_to_chosen_model, use_safetensors=True)
    model_to_eval.to(device)

    print("BEST EPOCH WAS", best_epoch)
    print("best f1", previous_best_f1)

    if args.contrastive:
        sys.exit() # if the model is being trained in a contrastive setting stop here, , do not run classification evaluation

    ################ EVALUATION
    for subset in ["dev", "test"]:
        print(subset)
        eval_dataloader = DataLoader(pt_dataset[subset], batch_size=4, collate_fn=data_collator)

        accumulating_predictions = []
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model_to_eval(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            accumulating_predictions.extend(predictions)
            metrics.add_batch(predictions=predictions, references=batch["labels"])
        complete_result = metrics.compute()
        print(complete_result)


        save_predictions(accumulating_predictions, subset)












