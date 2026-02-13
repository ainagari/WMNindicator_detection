
# spoken texts have <stext> instead of <text>


import os
import re
import pdb
import xml.etree.ElementTree as ET


conversations = dict()

skipped_utts = []

folder = "ota_20.500.12024_2554/2554/download/Texts/"


for letter in os.listdir(folder):
	print(letter)
	for secondletter in os.listdir(folder+letter+"/"):
		for fn in os.listdir(folder+letter+"/"+secondletter+"/"):			
			tree = ET.parse(folder+letter+"/"+secondletter+"/"+fn)
			root = tree.getroot()
			neighbors = [x for x in root.iter()]
			found_stext = [x for x in neighbors if x.tag == "stext"]
			if found_stext:
				assert len(found_stext) == 1
				stext = found_stext[0]
				clean_utterances = [] # (author, text)
				uttnum = -1
				for utterance in stext:							
					if "who" not in utterance.attrib:
						skipped_utts.append((fn, utterance))
						continue
					uttnum +=1
					author = utterance.attrib['who']
					words = []
					for sentence in utterance:
						if sentence.tag != "s":
							if sentence.tag == "unclear":
								words.append("[UNCLEAR]")
						else:
							for word in sentence:

								if word.tag in ["w","c"]:
									words.append(word.text.strip())
								elif word.tag in ["align","pause","shift","event","vocal"]: 
									continue
								elif word.tag == "unclear":
									words.append("[UNCLEAR]")
								elif word.tag == "gap":
									if "reason" in word.attrib and word.attrib['reason'] == "anonymization":
										words.append("[ANONYMIZATION]")
								elif word.tag in ["trunc","mw","corr"]:
									for truncword in word:
										if truncword.tag == "mw": 
											for otherword in truncword:
												words.append(otherword.text.strip())		
										elif truncword.tag == "w":
											words.append(truncword.text.strip())
										else:
											continue 
								else:
									pdb.set_trace()
					clean_utterances.append((author, uttnum, words))
				conversations[fn] = clean_utterances


for fn in conversations:
	with open("bnc_considered_simplified_spoken_conversations/"+fn[:-4]+".tsv","w") as out:
		for author, uttnum, utt in conversations[fn]:
			out.write(author + "\t" + str(uttnum) + "\t" + " ".join(utt) + "\n")



print("number of skipped utterances", len(skipped_utts))
