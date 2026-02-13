

# Obtaining the Indicators Dataset

Here we explain how to reconstruct the complete NeWMe corpus from its standoff annotations and public corpora, and how to derive the Indicators dataset from it.

## Obtaining NeWMe

The annotation data is distributed in a standoff format that contains only corpus identifiers, dialogue IDs, span positions and label metadata.

The full text is reconstructed bu downloading the original public corpora and matching the standoff positions. The code to do so is provided in [the NeWMe repository](https://github.com/gu-wmn/webapp/tree/main). However, you need to replace the file `webapp/src/newme/annotation/wmn_annotations.json` in that repository with the version provided in this directory, which, as opposed to the one in that repository, contains also all instances labeled as "Nothing".

After having cloned that repository and changed the standoff annotation file, run `flask --app src/newme/app run` as indicated in the instructions in the NeWMe repository. 
If this is the first time this is run, this will download the underlying corpora (Switchboard, Winning Args and the BNC), will extract the relevant dialogues (which will be saved to `corpora/extracted_corpora.json`) and will start the web interface.

You can stop the Flask server (Ctrl+C) once you see "Saving ./corpora/extracted_corpora.json": the extraction is complete.

Once the corpora have been downloaded, to reconstruct the NeWMe corpus from the standoff annotations, run the script `extract_newme_streaming.py` provided here. This will create the file `extracted_newme.jsonl` with the full NeWMe corpus.


## Deriving the Indicators dataset

To obtain the Indicators dataset from `extracted_newme.jsonl`, run the script `create_indicators_dataset.py`. This will place to new files under `../data/`: `indicators_dataset_random.json` and `indicators_dataset_regexaware.json`


# Obtaining the data for domain adaptation / MLM experiments and contrastive pairs


# Data for Domain adaptation


Since we can't share the original corpora directly (BNC, Switchboard, Reddit CMV), they need to be downloaded from their source (or using the flask webapp presented above).

* The BNC can also be downloaded from the [corpus website](https://www.natcorp.ox.ac.uk/getting/index.xml). 
 
Then, run the `search_bnc_spoken.py` script on it (modify the variable ``folder`` in it if necessary), which extracts from the BNC the spoken conversations that were considered for the NeWMe corpus construction and puts them in a simple format into the folder `bnc_considered_simplified_spoken_conversations/`.
Place this folder in this directory as shown here.

* For switchboard and Reddit CMV, you simply need to install the [convokit](https://convokit.cornell.edu/) library:

```pip3 install convokit```

Then you'll be ready to run the script `create_mlm_dataset.py`.
This will generate the `mlm_dataset.json` file that was used for training the models in the domain adaptation setting. It will be placed under ``../data``.



# Contrastive pairs

To obtain the pairs used for the contrastive learning setting, simply run: 

``python create_contrastive_dataset.py``

The data will be saved in ``../data/``.