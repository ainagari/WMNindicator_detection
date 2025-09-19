

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from argparse import ArgumentParser
import generating_llm_prompts
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import os
from operator import itemgetter

import random

random.seed(9)


def transform_prediction(prediction):
	if prediction.strip().lower() == "yes":
		return 1
	elif prediction.strip().lower() == "no":
		return 0
	else:
		return None



def evaluate(transformed_predictions, gold):
    # We will report two versions of the metrics: one assuming missing values are 0 and another omitting the missing values
    # Missing values are cases where the model did not reply Yes or No. This was rare: it only happened in 2 instances out of 4410 (test set size) for one model.
	# Calculate all metrics with what is available, and print the number of missing predictions.

	def0_metrics = {} # "default to 0" (the f1 obtained with this is what we report in the paper)
	notmiss_metrics = {} # ignoring missing values

	# def0 data:
	def0_predictions = [p if p != None else 0 for p in transformed_predictions]

	notmissing_gold = []
	notmissing_predictions = []
	number_missing_predictions = 0

	for p, g in zip(transformed_predictions, gold):
		if p != None:
			notmissing_gold.append(g)
			notmissing_predictions.append(p)
		else:
			number_missing_predictions += 1

	for metric in [accuracy_score, f1_score, precision_score, recall_score]:
		def0_metrics[metric.__name__] = metric(gold, def0_predictions)
		notmiss_metrics[metric.__name__] = metric(notmissing_gold, notmissing_predictions)
	notmiss_metrics['#missing_instances'] = number_missing_predictions
	notmiss_metrics['total_instances'] = len(gold)
	notmiss_metrics['pct_missing_instances'] = number_missing_predictions / len(gold)*100

	all_metrics = {'def0':def0_metrics, 'notmiss':notmiss_metrics}

	return all_metrics




if __name__ == "__main__":


    parser = ArgumentParser()	
    parser.add_argument("--setup", choices=['DEV-A','DEV-B','TEST-BEST'])
    parser.add_argument("--model", choices=['olmo','llama3B','llama70B'])
    parser.add_argument("--path_to_data")
    parser.add_argument("--llama_token", help="Request your token for the 3B model here: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--llama70B_path", help="provide the local path to llama70B snapshots, e.g. ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.3-70b-Instruct/snapshots/")
    args = parser.parse_args()

    ### interpret arguments
    if args.setup in ["DEV-A","DEV-B"]:
        subset = "dev"
    else:
        subset = "test"

    if args.setup == "DEV-A":
        prompt_types = list(range(1, 21))
    elif args.setup == "DEV-B" and args.model in ["olmo","llama7B"]:
       prompt_types = [21, 23]
    elif args.setup == "DEV-B" and args.model == "llama3B":
        prompt_types = [22,23]
    elif args.setup == "TEST-BEST" and args.model == "olmo":
        prompt_types = [19]
    elif args.setup == "TEST-BEST" and args.model == "llama3B":
        prompt_types = [20]
    elif args.setup == "TEST-BEST" and args.model == "llama7B":
        prompt_types = [21]



    #######################################################################


    # Load model

    if args.model == "olmo":
        llm = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
    elif args.model == "llama3B":
        llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token=args.llama_token)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token=args.llama_token)
    elif args.model == "llama70B":
        quant_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
        model_path = args.llama70B_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant_config, device_map="auto", trust_remote_code=True)
    

    if args.model != "llama70B":
        llm = llm.to('cuda')

    
    llm.eval()

    #######################################################################

    # Load the dataset

    with open(args.path_to_data) as f:
        data = json.load(f)


	# Loop over types of prompt that we want to try
    f1_by_prompttype = dict()

    for prompt_type in prompt_types:
        print(prompt_type)
        predictions = []
        gold = []
        out_dir = args.model + "_" + args.setup + "_" + str(prompt_type) + "/"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

		
        # Generate prompt introduction
        prompt_intro = generating_llm_prompts.create_prompt_intro(prompt_type)
		
        # Loop over the dataset to generate continuation with concrete example

        prompt_chars = generating_llm_prompts.prompt_dict[prompt_type]

        for i, instance in enumerate(data[subset]):
            gold.append(instance['label'])
            instance_text = ''
            if not "none" in prompt_chars['context']:
                instance_text = 'Context: '
                username_id_mapping = generating_llm_prompts.determine_usernames_for_instance(instance)
                context_candidates = instance['past_context'] if prompt_chars['context'] == '3past' else [instance['past_context'][-1]]
                for ctxt in context_candidates:
                    ctxt_str = username_id_mapping[ctxt['author']] + ": \"" + ctxt['text'] + "\""
                    instance_text += ctxt_str + "\n"

            if not "none" in prompt_chars['context']: # add username
                utt_str = username_id_mapping[instance['target']['author']] + ": \"" + instance['target']['text'] + "\""
                instance_text += "Target utterance: "
            else:
                if "utterance" in prompt_chars['context']:
                    utt_str = instance['target']['text'] + "\""
                    instance_text += "Target utterance: "

                elif "sentence" in prompt_chars['context']:
                    utt_str = instance['target']['sentence'] + "\""
                    instance_text += "Target sentence: "

            instance_text += utt_str +"\n"

            if prompt_chars['context'] == "1p1f":
                if 'future_context' in instance and instance['future_context']:
                    future_context_str = username_id_mapping[instance['future_context'][0]['author']] + ": \"" + instance['future_context'][0]['text'] + "\""
                    instance_text += future_context_str + "\n"


            instance_text += "Your response (Yes or No):"

            full_prompt = prompt_intro + "\n\n" + instance_text

            # Tokenize the prompt for the llm
            message = [{"role": "user", "content":full_prompt}]
            inputs = tokenizer.apply_chat_template(message, return_tensors='pt', add_generation_prompt=True, tokenize=True)


            inputs = inputs.to('cuda')


            # Feed prompt to the llm
            with torch.no_grad():
                response = llm.generate(inputs, max_new_tokens=1, do_sample=False)

            # Collect output
            readable_output = tokenizer.batch_decode(response, skip_special_tokens=True)[0]	    	

            # Save full output as txt (one per instance)
            with open(out_dir + str(i) + "_response.txt", 'w') as out:
                out.write(readable_output)

            # "Interpret" it (pick the last token, which should be Yes/No)
            last_token = readable_output.split()[-1]
            predictions.append(last_token)

        ###### Saving results: predictions and evaluation results

        # Save predictions
        with open(out_dir + "_predictions.txt", 'w') as out:
            for p in predictions:
                out.write(p + "\n")

        # Calculate evaluation metrics

        transformed_predictions = [transform_prediction(p) for p in predictions]

        results_all_metrics = evaluate(transformed_predictions, gold)

        print(results_all_metrics)
        f1_by_prompttype[prompt_type] = results_all_metrics['def0']['f1_score']

        # Save evaluation metric results
        json.dump(results_all_metrics, open(out_dir + "results.json",'w'))

    # Determine what the best prompt was and print the result

    print("results by prompt:", f1_by_prompttype)
    best_prompt = max(list(f1_by_prompttype.items()), key=itemgetter(1))[0]
    print("the best prompt was:", best_prompt)





