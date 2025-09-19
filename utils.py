import json

def load_dataset(regex_awareness=True, debugging_mode=False, corpus='all'):

    regex_str = "regexaware" if regex_awareness else "random"

    fn = "indicators_dataset_" + regex_str + ".json"

    with open(fn) as f:
        data = json.load(f)

    if debugging_mode:
        data['train'] = data['train'][:100]
        data['dev'] = data['dev'][:10]
        data['test'] = data['test'][:100]

    if corpus != "all":
        new_training = [x for x in data['train'] if x['corpus'] == corpus]
        data['train'] = new_training

    return data


def determine_usernames_for_instance(instance):
    # PENDING TO DEBUG
    context_usernames = []
    indicator_username = instance['target']['author']
    if instance['future_context']:
        future_context_username = instance['future_context'][0]['author']
    for context in instance['past_context']: # + instance['future_context']:
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
            idi += 1

    return username_id_mapping


