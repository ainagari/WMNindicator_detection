import json
from simplifying_swda import simplify
import random
import pdb
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from copy import copy, deepcopy
from matplotlib import pyplot as plt
from scipy.stats import describe
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


random.seed(0)

prop_train = 0.8
prop_dev = 0.05


if __name__ == "__main__":



    dataset_nr = json.load(open("../data/indicators_dataset_noregex.json"))
    dataset_rx = json.load(open("../data/indicators_dataset_regex.json"))
    datasets = [('random', dataset_nr), ('regex', dataset_rx)]


    print("Checking dataset possibilities:")

    print('TRAIN SET')
    for dsn, ds in datasets:
        print(dsn)
        subset = "train"        
        print("total instances", len(ds[subset]))
        numpos = len([x for x in ds[subset] if x['label'] == 1])
        numneg = len([x for x in ds[subset] if x['label'] == 0])    
        print('positives:', numpos)
        print('negatives:', numneg)
        print('total possible positive pairs:', numpos*numpos)
        print('total possible negative pairs:', numpos*numneg)
        


    print('DEV SET')
    for dsn, ds in datasets:
        print(dsn)
        subset = "dev"        
        print("total instances", len(ds[subset]))
        numpos = len([x for x in ds[subset] if x['label'] == 1])
        numneg = len([x for x in ds[subset] if x['label'] == 0])    
        print('positives:', numpos)
        print('negatives:', numneg)
        print('total possible positive pairs:', numpos*numpos)
        print('total possible negative pairs:', numpos*numneg)
        


    selected_pairs = dict()

    for dsn, ds in datasets:
        positive_pairs = set()
        negative_pairs = set()
        selected_pairs[dsn] = dict()

        maxpos = 10000
        maxneg = 20000

        while len(positive_pairs) < maxpos or len(negative_pairs) < maxneg:
            rd1 = random.randrange(0, len(ds['train']))
            rd2 = random.randrange(0, len(ds['train']))
            if rd1 == rd2:
                continue  
                
            lb1 = ds['train'][rd1]['label']        
            lb2 = ds['train'][rd2]['label']
            
            if lb1 == lb2 == 0:
                continue
                
            else:
                if lb1 == 1:
                    srds = [rd1, rd2]
                elif lb2 == 1:
                    srds = [rd2, rd1]                
            
            srds  = tuple(srds)        
            
            if lb1 == lb2 == 1 and srds not in positive_pairs and (srds[1], srds[0]) not in positive_pairs and len(positive_pairs) < maxpos:
                positive_pairs.add(srds)
            elif ((lb1 == 1 and lb2 == 0) or (lb1 == 0 and lb2 == 1)) and srds not in negative_pairs and (srds[1], srds[0]) not in negative_pairs and len(negative_pairs) < maxneg:
                negative_pairs.add(srds)
                
        selected_pairs[dsn]['positive'] = positive_pairs
        selected_pairs[dsn]['negative'] = negative_pairs
        



    # This will be the same for regex and random
    selected_pairs_dev = dict()
    dsn = "regex"

    ds = [ds for dn, ds in datasets if dn == dsn][0]
    positive_pairs = set()
    negative_pairs = set()

    maxpos = 300
    maxneg = 700

    while len(positive_pairs) < maxpos or len(negative_pairs) < maxneg:
        rd1 = random.randrange(0, len(ds['dev']))
        rd2 = random.randrange(0, len(ds['dev']))
        if rd1 == rd2:
            continue  

        lb1 = ds['dev'][rd1]['label']        
        lb2 = ds['dev'][rd2]['label']

        if lb1 == lb2 == 0:
            continue

        else:
            if lb1 == 1:
                srds = [rd1, rd2]
            elif lb2 == 1:
                srds = [rd2, rd1]
        
        srds  = tuple(srds)        

        if lb1 == lb2 == 1 and srds not in positive_pairs and (srds[1], srds[0]) not in positive_pairs and len(positive_pairs) < maxpos:
            positive_pairs.add(srds)
        elif ((lb1 == 1 and lb2 == 0) or (lb1 == 0 and lb2 == 1)) and srds not in negative_pairs and (srds[1], srds[0]) not in negative_pairs and len(negative_pairs) < maxneg:
            negative_pairs.add(srds)

    selected_pairs_dev['positive'] = positive_pairs
    selected_pairs_dev['negative'] = negative_pairs


    # Save the indices and the finetuning script will take care of it
    print("Saving...")

    for dsn in selected_pairs:
        all_pairs = list(selected_pairs[dsn]['positive'])+ list(selected_pairs[dsn]['negative'])
        random.shuffle(all_pairs)
        with open("../data/contrastive_pairs_" + dsn + ".tsv", "w") as out:
            for p1, p2 in all_pairs:
                out.write(str(p1) + "\t" + str(p2) + "\n")
                

    # and save dev pairs:
    all_pairs = list(selected_pairs_dev['positive'])+ list(selected_pairs_dev['negative'])
    random.shuffle(all_pairs)
    with open("../data/contrastive_pairs_dev.tsv", "w") as out:
        for p1, p2 in all_pairs:
            out.write(str(p1) + "\t" + str(p2) + "\n")





    # Number of positive and negative labels, in total, per subset and per corpus

    print("Getting dataset statistics:")


    selected_regex = ['you mean', 'can you define','definition of'] 

    for dsn, ds in datasets:
        sel = 0
        nonsel = 0
        for ins1, ins2 in selected_pairs[dsn]['negative']:        
            ins1, ins2 = ds['train'][ins1], ds['train'][ins2]
            negins = ins1 if ins1['label'] == 0 else ins2                
            s = False
            for rg in negins['regex']:
                if rg in selected_regex:
                    s = True
                if s:
                    sel+=1
                else:
                    nonsel+=1
        print("pct of selected regex among negative instances in", dsn, ":", sel/(sel+nonsel)*100)



    for dsn in selected_pairs: 
        print(dsn)
        all_idcs_p = [x for p in selected_pairs[dsn]['positive'] for x in p]
        all_idcs_n = [x for p in selected_pairs[dsn]['negative'] for x in p]
        cp = Counter(all_idcs_p)
        cn = Counter(all_idcs_n)
        
        print("positive", cp.most_common(20))
        print("negative", cn.most_common(20))
        



    for dsn, ds in datasets:
        print(dsn)
        for subset in ds:
            print(subset)
            print("total instances", len(ds[subset]))
            print('positives:', len([x for x in ds[subset] if x['label'] == 1]))
            print('negatives:', len([x for x in ds[subset] if x['label'] == 0]))
            print('nothings:', len([x for x in ds[subset] if x['wmn-label'] == 'Nothing']))
            print('distractors:', len([x for x in ds[subset] if x['wmn-label'] in ['reference/NE', 'Other kinds of clarification requests']]))
            for corpus in ['BNC','swda','Reddit']:
                print(corpus + ' positive:', len([x for x in ds[subset] if x['corpus'] == corpus and x['label'] == 1]))
                print(corpus + ' negative:', len([x for x in ds[subset] if x['corpus'] == corpus and x['label'] == 0]))
            
            print()
        
