
### script to generate prompts for LLMs
### It is used in llm_calls.py



prompt_dict = {1: {'context':'3past', 'explanation':True, 'examples':'A', 'mention_freq':False},
        2: {'context':'3past', 'explanation':False, 'examples':'A','mention_freq':False},
        3: {'context':'3past', 'explanation':True, 'examples':'A', 'mention_freq':True},
        4: {'context':'3past', 'explanation':False, 'examples':'A','mention_freq':True},
        5: {'context':"none:utterance", 'explanation':False, 'examples':'A','mention_freq':False},
        6: {'context':"none:utterance", 'explanation':True, 'examples':'A','mention_freq':False},
        7: {'context':"none:utterance", 'explanation':False, 'examples':'A','mention_freq':True},
        8: {'context':"none:utterance", 'explanation':True, 'examples':'A','mention_freq':True},
        9: {'context':'1p', 'explanation':True, 'examples':'A','mention_freq':False},
        10: {'context':'1p', 'explanation':False, 'examples':'A','mention_freq':False},
        11: {'context':'1p', 'explanation':True, 'examples':'A','mention_freq':True},
        12: {'context':'1p', 'explanation':False, 'examples':'A','mention_freq':True},
        13: {'context':'1p1f', 'explanation':True, 'examples':'A','mention_freq':False},
        14: {'context':'1p1f', 'explanation':False, 'examples':'A','mention_freq':False},
        15: {'context':'1p1f', 'explanation':True, 'examples':'A','mention_freq':True},
        16: {'context':'1p1f', 'explanation':False, 'examples':'A','mention_freq':True},
        17: {'context':"none:sentence", 'explanation':True, 'examples':'A','mention_freq':False},
        18: {'context':"none:sentence", 'explanation':False, 'examples':'A','mention_freq':False},
        19: {'context':"none:sentence", 'explanation':True, 'examples':'A','mention_freq':True}, # This was the final best prompt for OLMo in our experiments
        20: {'context':"none:sentence", 'explanation':False, 'examples':'A','mention_freq':True}, # This was the final best prompt for Llama-3.2-3B in our experiments
21: {'context':"none:sentence", 'explanation':True, 'examples':'B','mention_freq':True}, # This was the final best prompt for Llama-3.3-7B in our experiments
22: {'context':"none:sentence", 'explanation':False, 'examples':'B','mention_freq':True},
23: {'context':"none:sentence", 'explanation':False, 'examples':None,'mention_freq':True}} # zero-shot prompt
        


# task description when providing full utterance
td = "You are tasked with determining whether an utterance signals the need to discuss or clarify a word's meaning. This may take the form of a clarification request or a challenge to the appropriateness or meaning of a word or short phrase.\n"

# task description when providing sentence only
td_sentence = "You are tasked with determining whether a sentence signals the need to discuss or clarify a word's meaning. This may take the form of a clarification request or a challenge to the appropriateness or meaning of a word or short phrase.\n"

# instructions with full utterance
instructions = "Respond with \"Yes\" if the target utterance signals the need to discuss or clarify a word's meaning, or \"No\" if it does not."

# instructions with sentence only
instructions_sentence = "Respond with \"Yes\" if the target sentence signals the need to discuss or clarify a word's meaning, or \"No\" if it does not."


examples_explanation = "Below are some examples with the expected answer and an additional explanation."
examples = "Below are some examples with the expected answer." 
freq_info = "In most cases, the response should be \"No\"."

giving_context = "To help you decide, you are provided with at most three previous utterances as context, followed by the target utterance.\n"
giving_context_1p = "To help you decide, you are provided with the previous utterance as context, followed by the target utterance.\n"
giving_context_1p1f = "To help you decide, you are provided with the previous utterance as context, followed by the target utterance and the subsequent turn, if present.\n"


# The assignment below was done manually

task_descriptions = dict()

# Context, examples and explanations
task_descriptions[1] = td + giving_context + instructions + " " + examples_explanation
# Context, examples but no explanations
task_descriptions[2] = td + giving_context + instructions + " " + examples
# Same as 1 but with frequency info
task_descriptions[3] = task_descriptions[1] + " " + freq_info 
# Same as 2 but with frequency info
task_descriptions[4] = task_descriptions[2] + " " + freq_info 

# Now no context
# Context, examples and explanations
task_descriptions[5] = td + instructions + " " + examples_explanation
# Context, examples but no explanations
task_descriptions[6] = td + instructions + " " + examples
# Same as 5 but with frequency info
task_descriptions[7] = task_descriptions[5] + " " + freq_info
# Same as 6 but with frequency info
task_descriptions[8] = task_descriptions[6] + " " + freq_info

# Now 1past
# Context, examples and explanations
task_descriptions[9] = td + giving_context_1p + instructions + " " + examples_explanation
# Context, examples but no explanations
task_descriptions[10] = td + giving_context_1p + instructions + " " + examples
# Same as 9 but with frequency info
task_descriptions[11] = task_descriptions[9] + " " + freq_info
# Same as 10 but with frequency info
task_descriptions[12] = task_descriptions[10] + " " + freq_info

# Now 1p1f
# Context, examples and explanations
task_descriptions[13] = td + giving_context_1p1f + instructions + " " + examples_explanation
# Context, examples but no explanations
task_descriptions[14] = td + giving_context_1p1f + instructions + " " + examples
# Same as 13 but with frequency info
task_descriptions[15] = task_descriptions[13] + " " + freq_info
# Same as 14 but with frequency info
task_descriptions[16] = task_descriptions[14] + " " + freq_info

# now sentenceonly
# Context, examples and explanations
task_descriptions[17] = td_sentence + instructions_sentence + " " + examples_explanation
# Context, examples but no explanations
task_descriptions[18] = td_sentence + instructions_sentence + " " + examples
# Same as 17 but with frequency info
task_descriptions[19] = task_descriptions[17] + " " + freq_info
# Same as 18 but with frequency info
task_descriptions[20] = task_descriptions[18] + " " + freq_info

# now with example set B (using the best configuration obtained for each model with examples A)
task_descriptions[21] = task_descriptions[19] ## 11/09 fixed DONE
task_descriptions[22] = task_descriptions[20] ## 11/09 fixed DONE

# zero-shot
task_descriptions[23] = td_sentence + instructions_sentence


############# EXAMPLES A

examples = {'A':{'non':dict(), 'din':dict(), 'okocr':dict()}, 'B':{'non':dict(), 'din':dict(), 'okocr':dict()}}

#### DIN


example_din_A_context = "Context: Speaker 1: \"It is not uncommon to hear in sports press conferences of a player whose team is Super Bowl-bound, playoff-bound, or otherwise enjoying great success that they are \"humbled\" for the opportunity to play at that level. Or a winning candidate from an election to be \"humbled\" by the turnout in their favor. The problem is, when they say that, what they are really feeling is great pride and sense of accomplishment, which is decisively not humility. \"Humbled\" often goes hand-in-hand with \"Humiliated\", which is the opposite of what most people who describe themselves as \"humbled\" have gone through. If someone wants to say \"Although I am experiencing great success, I am still humble\", that is ok. However, even that can be a problem as people describing themselves as humble very often imply that that is some great characteristic about themselves, defeating the word (the ironic phrase \"I\'m probably the most humble person on Earth\" comes to mind). The only other context someone might describe themselves as \"humbled\" would be if they actually *were* properly humbled; e.g. they lost their house, car, and job and had to actually humble themselves by resorting to begging or a less dignified job, thus becoming \"humbled\". Outside of that context, I believe every usage of a person describing themselves as being \"humbled\" by some opportunity to be not only inaccurate, but polar opposite to what is meant by the word. This degrades the meaning of the word, and turns it into what is now an effectively useless proxy to say that one is proud of their accomplishments but wishes to remain coy about it. CMV!\n_____\n\n&gt; *\""
example_din_A_target_username = "Target utterance: Speaker 2: \"[STA-CITE]&gt;The problem is, when they say that, what they are really feeling is great pride and sense of accomplishment, which is decisively not humility.\n\n[END-CITE]No, what they are expressing is a sense of being unworthy and the recipient of a gift, a gift of talent, a gift of opportunity, a gift of trust from voters.  Now, it\'s possible you don\'t believe what they are saying, but that is what they are saying.  \"I\'m not worthy, thank you, I\'ll try not to let it go to my head, to stay humble, to keep my feet on the ground despite all these wonderful accolades.\"  That\'s what it means to say \"I feel humbled.\"\n\nHowever, even if the meaning were changing, what\'s wrong with language changing?  The word \"literally\" now means both \"literally\" and \"figuratively,\" because so many people have used it to mean \"figuratively\" that it\'s become part of the language.  Language changes, get over it.  \"Terrible\" used to mean \"inspiring terror,\" which was a good thing.  \"Awful\" used to mean \"inspiring awe,\" which was a good thing.  Same with \"Fearful.\"  Which is why the King of England, when St. Paul\'s Cathedral opened in London in the 1600s after the Great Fire, called it awful, terrible, and fearful -- and those were compliments.  Over time, they became insults.  There\'s no such thing as \"denigrating the meaning of a word.\"  There may be a lot of hypocrisy out there that should stop, but it has nothing to do with policing language.\""
example_din_A_target_nousername = "Target utterance: \"[STA-CITE]&gt;The problem is, when they say that, what they are really feeling is great pride and sense of accomplishment, which is decisively not humility.\n\n[END-CITE]No, what they are expressing is a sense of being unworthy and the recipient of a gift, a gift of talent, a gift of opportunity, a gift of trust from voters.  Now, it\'s possible you don\'t believe what they are saying, but that is what they are saying.  \"I\'m not worthy, thank you, I\'ll try not to let it go to my head, to stay humble, to keep my feet on the ground despite all these wonderful accolades.\"  That\'s what it means to say \"I feel humbled.\"\n\nHowever, even if the meaning were changing, what\'s wrong with language changing?  The word \"literally\" now means both \"literally\" and \"figuratively,\" because so many people have used it to mean \"figuratively\" that it\'s become part of the language.  Language changes, get over it.  \"Terrible\" used to mean \"inspiring terror,\" which was a good thing.  \"Awful\" used to mean \"inspiring awe,\" which was a good thing.  Same with \"Fearful.\"  Which is why the King of England, when St. Paul\'s Cathedral opened in London in the 1600s after the Great Fire, called it awful, terrible, and fearful -- and those were compliments.  Over time, they became insults.  There\'s no such thing as \"denigrating the meaning of a word.\"  There may be a lot of hypocrisy out there that should stop, but it has nothing to do with policing language.\""
example_din_A_explanation_username = "Explanation: Speaker 2 expresses a disagreement on the meaning of the word \"humble\"."
example_din_A_explanation_nousername = "Explanation: The speaker expresses a disagreement on the meaning of the word \"humble\"."
example_din_A_future_context = "Speaker 3: \"I would agree with you if people said something like, \"I am a humble person.\"  But not with the usage you are using, \"I am humbled to be voted MVP...\"  I think you are being a bit to literal with the definition of humbled.\""
example_din_A_target_sentence = "Target sentence: [END-CITE]No, what they are expressing is a sense of being unworthy and the recipient of a gift, a gift of talent, a gift of opportunity, a gift of trust from voters."

example_din_A_response = "Expected response: Yes"

examples['A']['din']['context'] = example_din_A_context
examples['A']['din']['context1'] = example_din_A_context # because there is only one turn
examples['A']['din']['future_context'] = example_din_A_future_context
examples['A']['din']['target_sentence'] = example_din_A_target_sentence
examples['A']['din']['expected_response'] = example_din_A_response
examples['A']['din']['target_username'] = example_din_A_target_username
examples['A']['din']['target_nousername'] = example_din_A_target_nousername
examples['A']['din']['explanation_username'] = example_din_A_explanation_username
examples['A']['din']['explanation_nousername'] = example_din_A_explanation_nousername


## NON
# 3 previous utterances
example_non_A_context = "Context: Speaker 1: \"enough . They 'll complain about an individual flat ,\"\nSpeaker 2: \"Yeah , what are their con\"\nSpeaker 1: \"and I mean things like disrepair , erm inadequate heating , erm noise from other tenants , erm...\""
# single previous utterance
example_non_A_context1 = "Context: Speaker 1: \"and I mean things like disrepair , erm inadequate heating , erm noise from other tenants , erm...\""
example_non_A_target_username = "Target utterance: Speaker 2: \"But noise in what sense ? [UNCLEAR] what kind of noise are they talking about ? Are they talking about\""
example_non_A_target_nousername = "Target utterance: \"But noise in what sense ? [UNCLEAR] what kind of noise are they talking about ? Are they talking about\""
example_non_A_explanation_username = "Explanation: Speaker 2 asks for clarification about the word \"noise\"."
example_non_A_explanation_nousername = "Explanation: The speaker asks for clarification about the word \"noise\"."
example_non_A_future_context = "Speaker 1: \"Well , it would depend but I mean there are various of nuisance from noise in the flats , or anywhere where you 've got a lot of people put together all living in a s fairly small area . But erm but they do n't , they do n't really complain about the complex as a whole . They 'll complain about their own i individual bit of it .\""
example_non_A_target_sentence = "Target sentence: But noise in what sense ? [UNCLEAR] what kind of noise are they talking about ?"

example_non_A_response = "Expected response: Yes"

examples['A']['non']['context'] = example_non_A_context
examples['A']['non']['context1'] = example_non_A_context1
examples['A']['non']['future_context'] = example_non_A_future_context
examples['A']['non']['expected_response'] = example_non_A_response
examples['A']['non']['target_sentence'] = example_non_A_target_sentence
examples['A']['non']['target_username'] = example_non_A_target_username
examples['A']['non']['target_nousername'] = example_non_A_target_nousername
examples['A']['non']['explanation_username'] = example_non_A_explanation_username
examples['A']['non']['explanation_nousername'] = example_non_A_explanation_nousername

## negative example: OKOCR

example_okocr_A_context = "Context: Speaker 1: \"Both, they're kind of the same view.\n\nI'm arguing that it's no longer reasonable in today's world and people should think before saddling themselves with debt like that.\"\nSpeaker 2: \"[STA-CITE]&gt; I'm arguing that it's no longer reasonable in today's world and people should think before saddling themselves with debt like that.\n\n[END-CITE]Is the problem that you can't support yourself above first world poverty standards on a blue collar job?\n\nor \n\nIs the problem that blue collar people try to do it at all?\n\nI think this is where a lot of the confusion about your view comes from.\"\nSpeaker 1: \"That they try to do it at all\""
example_okocr_A_context1 = "Context: Speaker 1: \"That they try to do it at all\""
example_okocr_A_target_sentence = "Target sentence: I'm not trying to lampoon you, but that's what I'm getting from your posts and I figure you'd like to clarify that statement."
example_okocr_A_target_username = "Target utterance: Speaker 2: \"So your view is that poor people should stop reproducing because they can't afford cell phones?\n\nI'm not trying to lampoon you, but that's what I'm getting from your posts and I figure you'd like to clarify that statement.\""
example_okocr_A_target_nousername = "Target utterance: \"So your view is that poor people should stop reproducing because they can't afford cell phones?\n\nI'm not trying to lampoon you, but that's what I'm getting from your posts and I figure you'd like to clarify that statement.\""
example_okocr_A_explanation_username = "Explanation: Speaker 2 asks for a clarification, but it involves a broader semantic content rather than a specific word's meaning."
example_okocr_A_explanation_nousername = "Explanation: The speaker asks for a clarification, but it involves a broader semantic content rather than a specific word's meaning."
example_okocr_A_future_context = "Speaker 3: \"Japan is a modern economy whose society is mostly a large middle class, and many of these jobs are blue collar (rice farmer, fisherman, repairman, shop owner, etc), especially in rural areas.\""

example_okocr_A_response = "Expected response: No"

examples['A']['okocr']['context'] = example_okocr_A_context
examples['A']['okocr']['context1'] = example_okocr_A_context1
examples['A']['okocr']['future_context'] = example_okocr_A_future_context
examples['A']['okocr']['expected_response'] = example_okocr_A_response
examples['A']['okocr']['target_sentence'] = example_okocr_A_target_sentence
examples['A']['okocr']['target_username'] = example_okocr_A_target_username
examples['A']['okocr']['target_nousername'] = example_okocr_A_target_nousername
examples['A']['okocr']['explanation_username'] = example_okocr_A_explanation_username
examples['A']['okocr']['explanation_nousername'] = example_okocr_A_explanation_nousername


############# EXAMPLES B

# DIN

example_din_B_context = "Context: Speaker 1: \"CMV: I don't believe a person can be truly compassionate before having experienced a prolonged state of egolessness.\nSpeaker 1: I have a general feeling about whether a person has experienced living without ego or not by observation of where they place the importance of their own wants against the needs of others. From this, it is my belief that a person cannot be actively compassionate in every day life when they have not experienced the falseness of self because the ego is the only self they are truly aware of.\n\nCompassion being defined as being able to possess a love for all beings that is as great as that for the self and the closest people in their lives.\n\nTruly compassionate being defined as actively compassionate in every day life for all those that they can have an effect on.\""
example_din_B_context1 = "Context: Speaker 1: I have a general feeling about whether a person has experienced living without ego or not by observation of where they place the importance of their own wants against the needs of others. From this, it is my belief that a person cannot be actively compassionate in every day life when they have not experienced the falseness of self because the ego is the only self they are truly aware of.\n\nCompassion being defined as being able to possess a love for all beings that is as great as that for the self and the closest people in their lives.\n\nTruly compassionate being defined as actively compassionate in every day life for all those that they can have an effect on.\""
example_din_B_target_username = "Target utterance: Speaker 2: \"[STA-CITE]&gt;Compassion being defined as being able to possess a love for all beings that is as great as that for the self and the closest people in their lives.\n\n[END-CITE]I don't believe anyone who is not either mentally disturbed or deficient is actually capable of doing this, and I wouldn't want him to be.  This isn't a functional or desirable definition of compassion.\""
example_din_B_target_nousername = "Target utterance: \"[STA-CITE]&gt;Compassion being defined as being able to possess a love for all beings that is as great as that for the self and the closest people in their lives.\n\n[END-CITE]I don't believe anyone who is not either mentally disturbed or deficient is actually capable of doing this, and I wouldn't want him to be.  This isn't a functional or desirable definition of compassion.\""
example_din_B_explanation_username = "Explanation: Speaker 2 expresses a disagreement on the meaning of the word \"compassion\"."
example_din_B_explanation_nousername = "Explanation: The speaker expresses a disagreement on the meaning of the word \"compassion\"."
example_din_B_target_sentence = "Target sentence: This isn't a functional or desirable definition of compassion."
example_din_B_future_context = "Speaker 3: \"situational compassion just tossed your logic right out the window. a person can have true compassion if they better know the situation or have been in someone's shoes.\""

example_din_B_response = "Expected response: Yes"

examples['B']['din']['context'] = example_din_B_context
examples['B']['din']['context1'] = example_din_B_context1
examples['B']['din']['target_username'] = example_din_B_target_username
examples['B']['din']['target_nousername'] = example_din_B_target_nousername
examples['B']['din']['target_sentence'] = example_din_B_target_sentence
examples['B']['din']['explanation_username'] = example_din_B_explanation_username
examples['B']['din']['explanation_nousername'] = example_din_B_explanation_nousername
examples['B']['din']['expected_response'] = example_din_B_response # because there is only one turn!
examples['B']['din']['future_context'] = example_din_B_future_context

# NON


example_non_B_context = "Context: Speaker 1: \"Uh , some of the tracking control things and skidding control things for up north .\"\nSpeaker 2: \"Uh-huh .\"\nSpeaker 1: \"The C D and the premium sound system .\""
example_non_B_context1 = "Context: Speaker 1: \"The C D and the premium sound system .\""
example_non_B_target_username = "Target utterance: Speaker 2: \"Skidding control , you mean the antilock brake system ?\""
example_non_B_target_nousername = "Target utterance: \"Skidding control , you mean the antilock brake system ?\""
example_non_B_explanation_username = "Explanation: Speaker 2 asks for clarification about the phrase \"skidding control\"."
example_non_B_explanation_nousername = "Explanation: The speaker asks for clarification about the word \"skidding control\"."
example_non_B_target_sentence = "Target sentence: Skidding control , you mean the antilock brake system ?"
example_non_B_future_context = "Speaker 1: \"Yeah , it 's kind of a traction control , I think they call it .\""
example_non_B_response = "Expected response: Yes"


examples['B']['non']['context'] = example_non_B_context
examples['B']['non']['context1'] = example_non_B_context1
examples['B']['non']['target_username'] = example_non_B_target_username
examples['B']['non']['target_nousername'] = example_non_B_target_nousername
examples['B']['non']['target_sentence'] = example_non_B_target_sentence
examples['B']['non']['explanation_username'] = example_non_B_explanation_username
examples['B']['non']['explanation_nousername'] = example_non_B_explanation_nousername
examples['B']['non']['expected_response'] = example_non_B_response 
examples['B']['non']['future_context'] = example_non_B_future_context 


# negative example: OKOCR

example_okocr_B_context = "Context: Speaker 1: \"It received a lot of interest , the N F U were there er and the [UNCLEAR] area , that area was very very keen on getting something together\"\nSpeaker 2: \"Well\"\nSpeaker 2: \"The way we tried to turn them er despite Bill [ANONYMIZATION] we decided to try to turn them to become involved in neighbourhood watch schemes , which there are plenty of already , and erm they were quite interested in that the farmers that were there .\""
example_okocr_B_context1 = "Context: Speaker 2: \"The way we tried to turn them er despite Bill [ANONYMIZATION] we decided to try to turn them to become involved in neighbourhood watch schemes , which there are plenty of already , and erm they were quite interested in that the farmers that were there .\""
example_okocr_B_target_username = "Target utterance: Speaker 3: \"What do you mean despite them ?\""
example_okocr_B_target_nousername = "Target utterance: \"What do you mean despite them ?\""
example_okocr_B_explanation_username = "Explanation: Speaker 3 asks for a clarification, but it involves a broader semantic content rather than a specific word's meaning."
example_okocr_B_explanation_nousername = "Explanation: The speaker asks for a clarification, but it involves a broader semantic content rather than a specific word's meaning."
example_okocr_B_target_sentence = "Target sentence: What do you mean despite them ?"
example_okocr_B_future_context = "Speaker 4: \"Well Bill stood up and was a bit negative really to the whole idea of parish constables , farm watch and everything else , it was quite disappointing\""

example_okocr_B_response = "Expected response: No"

examples['B']['okocr']['context'] = example_okocr_B_context
examples['B']['okocr']['context1'] = example_okocr_B_context1
examples['B']['okocr']['target_username'] = example_okocr_B_target_username
examples['B']['okocr']['target_nousername'] = example_okocr_B_target_nousername
examples['B']['okocr']['target_sentence'] = example_okocr_B_target_sentence
examples['B']['okocr']['explanation_username'] = example_okocr_B_explanation_username
examples['B']['okocr']['explanation_nousername'] = example_okocr_B_explanation_nousername
examples['B']['okocr']['expected_response'] = example_okocr_B_response
examples['B']['okocr']['future_context'] = example_okocr_B_future_context 


#############
#############



your_turn = "\nNow it's your turn:"


def determine_usernames_for_instance(instance):
    context_usernames = []
    indicator_username = instance['target']['author']
    if instance['future_context']:
        future_context_username = instance['future_context'][0]['author']
    for context in instance['past_context']:
        context_usernames.append(context['author'])
    username_id_mapping = dict()
    idi = 1
    if instance['future_context']:
        alls = context_usernames + [indicator_username] + [future_context_username]
    else:
        alls = context_usernames + [indicator_username]

    for cu in alls:
        if cu not in username_id_mapping:
            username_id_mapping[cu] = "Speaker " + str(idi)
            idi +=1

    return username_id_mapping



def create_prompt_intro(prompt_type):
    prompt_chars = prompt_dict[prompt_type]

    ###### Creating intro to prompt based on desired prompt characteristics
    prompt_intro = task_descriptions[prompt_type]

    if prompt_chars['examples']:
        if prompt_chars['context'] in ['1p','1p1f']:
            context_key = "context1"
        elif prompt_chars['context'] == '3past': #True:
            context_key = "context"
        if "none" in prompt_chars['context']:
            target_k = "target_nousername" if prompt_chars['context'] == "none:utterance" else "target_sentence"
        if (not "none" in prompt_chars['context'])  and prompt_chars['explanation']: 
                prompt_intro += "\nExample 1:\n"
                prompt_intro += examples[prompt_chars['examples']]['non'][context_key] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['non']['target_username'] + "\n"
                if prompt_chars['context'] == "1p1f":
                        prompt_intro += examples[prompt_chars['examples']]['non']['future_context'] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['non']['expected_response'] + "\n"

                prompt_intro += examples[prompt_chars['examples']]['non']['explanation_username'] + "\n"
                prompt_intro += "\nExample 2:\n"
                prompt_intro += examples[prompt_chars['examples']]['din'][context_key] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['din']['target_username'] + "\n"
                if prompt_chars['context'] == "1p1f":
                        prompt_intro += examples[prompt_chars['examples']]['din']['future_context'] + "\n"

                prompt_intro += examples[prompt_chars['examples']]['din']['expected_response'] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['din']['explanation_username'] + "\n"
                prompt_intro += "\nExample 3:\n"
                prompt_intro += examples[prompt_chars['examples']]['okocr'][context_key] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['okocr']['target_username'] + "\n"
                if prompt_chars['context'] == "1p1f":
                        prompt_intro += examples[prompt_chars['examples']]['okocr']['future_context'] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['okocr']['expected_response'] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['okocr']['explanation_username'] + "\n"

        elif (not "none" in prompt_chars['context']) and not prompt_chars['explanation']: 
                prompt_intro += "\nExample 1:\n"
                prompt_intro += examples[prompt_chars['examples']]['non'][context_key] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['non']['target_username'] + "\n"			
                if prompt_chars['context'] == "1p1f":
                        prompt_intro += examples[prompt_chars['examples']]['non']['future_context'] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['non']['expected_response'] + "\n"

                prompt_intro += "\nExample 2:\n"
                prompt_intro += examples[prompt_chars['examples']]['din'][context_key] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['din']['target_username'] + "\n"			
                if prompt_chars['context'] == "1p1f":
                        prompt_intro += examples[prompt_chars['examples']]['din']['future_context'] + "\n"

                prompt_intro += examples[prompt_chars['examples']]['din']['expected_response'] + "\n"                
                prompt_intro += "\nExample 3:\n"
                prompt_intro += examples[prompt_chars['examples']]['okocr'][context_key] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['okocr']['target_username'] + "\n"
                if prompt_chars['context'] == "1p1f":
                        prompt_intro += examples[prompt_chars['examples']]['okocr']['future_context'] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['okocr']['expected_response'] + "\n"

        elif ("none" in prompt_chars['context']) and prompt_chars['explanation']: 
                prompt_intro += "\nExample 1:\n"
                prompt_intro += examples[prompt_chars['examples']]['non'][target_k] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['non']['expected_response'] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['non']['explanation_nousername'] + "\n"
                prompt_intro += "\nExample 2:\n"			
                prompt_intro += examples[prompt_chars['examples']]['din'][target_k] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['non']['expected_response'] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['din']['explanation_nousername'] + "\n"
                prompt_intro += "\nExample 3:\n"			
                prompt_intro += examples[prompt_chars['examples']]['okocr'][target_k] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['non']['expected_response'] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['okocr']['explanation_nousername'] + "\n"
        elif ("none" in prompt_chars['context']) and not prompt_chars['explanation']: 
                prompt_intro += "\nExample 1:\n"			
                prompt_intro += examples[prompt_chars['examples']]['non'][target_k] + "\n"			
                prompt_intro += examples[prompt_chars['examples']]['non']['expected_response'] + "\n"
                prompt_intro += "\nExample 2:\n"			
                prompt_intro += examples[prompt_chars['examples']]['din'][target_k] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['din']['expected_response'] + "\n"
                prompt_intro += "\nExample 3:\n"			
                prompt_intro += examples[prompt_chars['examples']]['okocr'][target_k] + "\n"
                prompt_intro += examples[prompt_chars['examples']]['okocr']['expected_response'] + "\n"

    prompt_intro += your_turn

    if not prompt_chars['examples']:
        prompt_intro += "\n"


    return prompt_intro





















