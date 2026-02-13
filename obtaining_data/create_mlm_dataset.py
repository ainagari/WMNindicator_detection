
import json
import os
import random
from create_indicators_dataset import simplify
random.seed(9)
from convokit import Corpus, download
import re


bnc_dir = "bnc_considered_simplified_spoken_conversations/" 

def check_unavailable_myutt(utt):
    if utt['text'] in ['[deleted]', '[removed]', '<INAUDIBLE>'] or utt['author'] in ['[deleted]', '[removed]', '<INAUDIBLE>']:
            return True
    return False
    
def check_unavailable(conv):
    for utt in conv.iter_utterances():
        if utt.text in ['[deleted]', '[removed]', '<INAUDIBLE>'] or utt.speaker.id in ['[deleted]', '[removed]', '<INAUDIBLE>']:
            return True
    return False

def remove_cmv_extra_text(utterance):
    # Remove anything that comes after "*Hello, users of CMV! This is a footnote"
    try:
        match = re.search("\n\n&gt; *Hello, users of CMV! This is a", utterance)
        if match:
            new_utterance = utterance[:match.span()[0]]
            #print('found')
        else:
            new_utterance = utterance
    except:
        new_utterance = utterance
    try:
        match = re.search("Hello, users of CMV! This is a footnote", new_utterance)
        if match:
            nnew_utterance = new_utterance[:match.span()[0]]
            #print('found')
        else:
            nnew_utterance = new_utterance
    except:
        nnew_utterance = new_utterance
        
    return nnew_utterance
        
def modify_for_citation(utterance_text): 
    citation_matches = re.finditer("&gt;.*?\n\n", utterance_text)
    if citation_matches:
        filtered_text = ''
        citation_spans = [(m.span()[0], m.span()[1]) for m in citation_matches]
        if not citation_spans:
            return utterance_text
        else:
            decalage = 0
            next_first_index = 0
            for cit_span in citation_spans:
                filtered_text += utterance_text[next_first_index:cit_span[0]] + '[STA-CITE]' + utterance_text[cit_span[0]:cit_span[1]] + '[END-CITE]'
                decalage += len('[STA-CITE]') + len('[END-CITE]')
                next_first_index = len(filtered_text) - decalage
            filtered_text += utterance_text[next_first_index:]
            return filtered_text


if __name__ == '__main__':
	    
	dataset_rx = json.load(open("../data/indicators_dataset_regexaware.json"))

	all_conv_ids = {"BNC":set(),'swda':set(),'Reddit':set()}


	for subset in dataset_rx:
	    for ins in dataset_rx[subset]:
	        if ins['corpus'] in ["BNC","swda"]:
	            conv_id = ins['id'].split("_")[0]
	        elif ins['corpus'] == "Reddit":
	            conv_id = "_".join(ins['id'].split("_")[0:2])

	        all_conv_ids[ins['corpus']].add(conv_id)


	############## BNC
	print("BNC")


	conversations = dict()

	for fn in os.listdir(bnc_dir):
	    with open(bnc_dir + fn) as f:
	        conv_name = fn[:-4]
	        if conv_name not in all_conv_ids['BNC']:
	            conversations[conv_name] = []
	            for l in f:
	                author, uttnum, utt = l.split("\t")
	                conversations[conv_name].append({'author':author,'text':utt.strip(), 'id': uttnum})
	                
	empty_keys = []
	for k in conversations:
	    if not conversations[k]:
	        empty_keys.append(k)
	for k in empty_keys:
	    del conversations[k]




	bnc_pairs = []
	i=0
	for conv in conversations:    
	    if i >= 50:
	        break
	    if len(conversations[conv]) % 2 != 0:
	        conversations[conv].append({'text':'', 'author':''})
	    for j in range(0, len(conversations[conv]), 2):
	        if conversations[conv][j+1]:
	            bnc_pairs.append((conversations[conv][j], conversations[conv][j+1], conv + "_" + str(j) + "--" + str(j+1)))
	        else: # if empty
	            bnc_pairs.append((conversations[conv][j], conversations[conv][j+1], conv + "_" + str(j) + "--"))
	    i+=1

	ready_bnc_pairs = []
	for pair in bnc_pairs:
	    if pair[0]['author'] == pair[1]['author']:
	        num = str(random.randint(1,5))
	        text = "Speaker " + num + ": " + pair[0]['text'] + " Speaker " + num + ": " + pair[1]['text']
	    elif pair[1]['author'] == '': # last empty utterance
	        num = str(random.randint(1,5))
	        text = "Speaker " + num + ": " + pair[0]['text']
	    else:
	        num1 = str(random.randint(1,5))
	        num2 = str(num1)
	        while num2 == num1:
	            num2 = str(random.randint(1,5))
	        text = "Speaker " + num1 + ": " + pair[0]['text'] + " Speaker " + num2 + ": " + pair[1]['text']
	    ready_bnc_pairs.append((text, pair[2])) # pair 2 is the conv id with the two utt numbers...
	        
	        
	    

	################# SWDA
	print("SWDA")
	corpus = Corpus(filename=download("switchboard-corpus"))

	conversations = dict()

	for conv in corpus.iter_conversations():
	    ### check if there are unavailable posts & ignore such conversations
	    unav = check_unavailable(conv)
	    if unav:
	        continue
	    if conv.id in all_conv_ids['swda']:
	        continue         
	        
	        
	    utts_this_conv = []
	    for uttnum, utt in enumerate(conv.iter_utterances()):
	        text = utt.text
	        utt_dict = {'id': utt.id, "text": simplify(text), "utt_order_num": uttnum, 'author-plain': utt.speaker.id}
	        utt_dict['author'] = utt_dict['author-plain']
	        utts_this_conv.append(utt_dict)
	    conversations[conv.id] = utts_this_conv



	swda_pairs = []
	i=0
	for conv in conversations:    
	    if i >= 60:
	        break
	    if len(conversations[conv]) % 2 != 0:
	        conversations[conv].append({'text':'', 'author':''})
	    for j in range(0, len(conversations[conv]), 2):
	        if conversations[conv][j+1]:
	            swda_pairs.append((conversations[conv][j], conversations[conv][j+1], conv + "_" + str(j) + "--" + str(j+1)))
	        else: # if empty
	            swda_pairs.append((conversations[conv][j], conversations[conv][j+1], conv + "_" + str(j) + "--"))
	    i+=1

	ready_swda_pairs = []
	for pair in swda_pairs:
	    if pair[0]['author'] == pair[1]['author']:
	        num = str(random.randint(1,2))
	        text = "Speaker " + num + ": " + pair[0]['text'] + " Speaker " + num + ": " + pair[1]['text']
	    elif pair[1]['author'] == '': # last empty utterance
	        num = str(random.randint(1,2))
	        text = "Speaker " + num + ": " + pair[0]['text']
	    else:
	        num1 = str(random.randint(1,2))
	        num2 = str(num1)
	        while num2 == num1:
	            num2 = str(random.randint(1,2))
	        text = "Speaker " + num1 + ": " + pair[0]['text'] + " Speaker " + num2 + ": " + pair[1]['text']
	    ready_swda_pairs.append((text, pair[2]))
	        
	        
	############## REDDIT    
	print("REDDIT")
	corpus = Corpus(filename=download("winning-args-corpus"))

	conversations = dict()
	pairs_by_conv = dict()
	not_ids_present_num_cases = 0

	for conv in corpus.iter_conversations():
	    
	    if conv.id in all_conv_ids['Reddit']:
	        continue            
	    
	    #utts_this_conv = []
	    utts_this_conv = [{'author':'TITLE', 'text':conv.meta['op-title'],'id': conv.id + "_title", "reply_to":None}]
	    for uttnum, utt in enumerate(conv.iter_utterances()):        
	        text = utt.text
	        text = modify_for_citation(text)
	        text = remove_cmv_extra_text(text)

	        utt_dict = {'id': utt.id, "text": text, "utt_order_num": uttnum, 'author-plain': utt.speaker.id}


	        # for reddit corpora, keep track of what post is being replied to
	        if "reply_to" in dir(utt):
	            utt_dict["reply_to"] = utt.reply_to
	            reply_txt = "_##_rt-" + utt.reply_to if utt.reply_to else ""
	            utt_dict['author'] = utt.speaker.id + "_##_" + utt.id + reply_txt
	        else:
	            utt_dict['author'] = utt_dict['author-plain']
	        utts_this_conv.append(utt_dict)

	    
	    reply_chain_almost = dict()
	    
	    all_ids = [u['id'] for u in utts_this_conv]
	    all_ids_are_present = True
	    ### first check
	    for utt_idx, utt in enumerate(utts_this_conv):
	        if utt['reply_to'] is not None and not utt['reply_to'] in all_ids:
	            all_ids_are_present = False
	        reply_chain_almost[utt['id']] = utt['reply_to']
	        if utt_idx > 0 and utt_idx < len(utts_this_conv)-1:
	            if not utts_this_conv[utt_idx+1]['reply_to'] == utt['id']:
	                continue # this was just for debugging

	    if not all_ids_are_present:
	        not_ids_present_num_cases +=1
	        continue # we skip this conv
	   
	    
	    reply_chain = dict() # from utterance id to the tree of message ids that it replies to
	    for k in reply_chain_almost:
	        reply_chain[k] = [reply_chain_almost[k]]
	        finished = False
	        while not finished:
	            if None in reply_chain[k]:
	                finished = True
	            for other_k in reply_chain_almost:
	                if other_k != k and other_k in reply_chain[k] and reply_chain_almost[other_k] not in reply_chain[k]:
	                    if reply_chain_almost[other_k] == None:
	                        finished = True
	                        break
	                    else:
	                        reply_chain[k].append(reply_chain_almost[other_k])                      
	                        

	    ordered_utts_this_conv = []
	    for utt in utts_this_conv:
	        if utt['reply_to'] == None:
	            ordered_utts_this_conv.append(utt)
	        else:
	            last_other_possible_reply_idx = None
	            for utt_idx, possible_parent_utt in enumerate(ordered_utts_this_conv):
	                if possible_parent_utt['id'] in reply_chain[utt['id']]:
	                    last_other_possible_reply_idx = utt_idx

	            if last_other_possible_reply_idx is not None:
	                ordered_utts_this_conv = ordered_utts_this_conv[:last_other_possible_reply_idx+1] + [utt] + ordered_utts_this_conv[last_other_possible_reply_idx+1:]
	                
	                
	    pairs = []
	    used_utterance_ids = []
	    for final_reply_id in reply_chain:
	        replied_to = reply_chain[final_reply_id][0] 
	        if final_reply_id and replied_to:
	            if final_reply_id not in used_utterance_ids and replied_to not in used_utterance_ids:
	                pair1 = [utt for utt in ordered_utts_this_conv if utt['id'] == replied_to][0]
	                pair2 = [utt for utt in ordered_utts_this_conv if utt['id'] == final_reply_id][0]            
	                if not check_unavailable_myutt(pair1) and not check_unavailable_myutt(pair2):
	                    pairs.append((pair1, pair2, conv.id + "_" + final_reply_id + "--" + replied_to))
	                    used_utterance_ids.append(final_reply_id)
	                    used_utterance_ids.append(replied_to)
	            if len(reply_chain[final_reply_id]) > 2:
	                pair1 = [utt for utt in ordered_utts_this_conv if utt['id'] == reply_chain[final_reply_id][2]][0]
	                pair2 = [utt for utt in ordered_utts_this_conv if utt['id'] == reply_chain[final_reply_id][1]][0]            
	                if not check_unavailable_myutt(pair1) and not check_unavailable_myutt(pair2):
	                    pairs.append((pair1, pair2, conv.id + "_" + reply_chain[final_reply_id][1] + "--" + reply_chain[final_reply_id][2]))
	                    used_utterance_ids.append(reply_chain[final_reply_id][1])
	                    used_utterance_ids.append(reply_chain[final_reply_id][2])
	                

	        elif not replied_to or (final_reply_id not in used_utterance_ids and replied_to in used_utterance_ids):
	            pair1 = [utt for utt in ordered_utts_this_conv if utt['id'] == final_reply_id][0]
	            pair2 = {'text':'','author':''}
	            if not check_unavailable_myutt(pair1):
	                pairs.append((pair1, pair2, conv.id + "_" + final_reply_id + "--"))
	                used_utterance_ids.append(final_reply_id)
	        
	        
	    pairs_by_conv[str(conv.id)] = pairs


	    utts_this_conv = ordered_utts_this_conv	        
	    conversations[str(conv.id)] = utts_this_conv
	reddit_pairs = []
	i=0
	for conv in pairs_by_conv:    
	    if i >= 17:
	        break
	    for p in pairs_by_conv[conv]:
	        reddit_pairs.append(p)
	    i+=1

	ready_reddit_pairs = []
	for pair in reddit_pairs:
	    if pair[0]['author'] == pair[1]['author']:
	        num = str(random.randint(1,5))
	        text = "Speaker " + num + ": " + pair[0]['text'] + " Speaker " + num + ": " + pair[1]['text']
	    elif pair[1]['author'] == '': # last empty utterance
	        num = str(random.randint(1,5))
	        text = "Speaker " + num + ": " + pair[0]['text']
	    else:
	        num1 = str(random.randint(1,5))
	        num2 = str(num1)
	        while num2 == num1:
	            num2 = str(random.randint(1,5))
	        text = "Speaker " + num1 + ": " + pair[0]['text'] + " Speaker " + num2 + ": " + pair[1]['text']
	    ready_reddit_pairs.append((text, pair[1]))
	        
	        

	all_pairs = ready_bnc_pairs + ready_swda_pairs + ready_reddit_pairs
	random.shuffle(all_pairs)


	json.dump(all_pairs, open("../data/mlm_dataset.json", "w"))

