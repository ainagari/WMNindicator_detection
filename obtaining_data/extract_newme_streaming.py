#!/usr/bin/env python3
"""
Extract browsable data from the webapp into JSONL format (streaming, memory-efficient).

This script combines the standoff annotations with corpus dialogues to create
enriched data, writing one sequence per line to avoid loading everything into memory.
"""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from newme.annotation import Annotation


def main():
    """Main extraction function - streaming write to JSONL."""
    
    data_path = Path(__file__).parent
    
    print("=" * 70)
    print("Extracting Browsable Data (Streaming Mode)")
    print("=" * 70)
    print()
    
    # Load annotations
    print("Loading annotations...")
    annotation = Annotation(data_path=data_path)
    
    annotation_file_path = data_path / "src/newme/annotation/wmn_annotations.json"
    with open(annotation_file_path) as f:
        raw_annotations = json.load(f)
    
    print(f"  Found {len(raw_annotations)} WMN sequences")
    print()
    
    # Get corpus object
    corpus_obj = annotation._Annotation__corpus
    
    # Output file
    output_file = data_path / "extracted_newme.jsonl"
    print(f"Writing to: {output_file}")
    print("  Format: JSONL (one sequence per line)")
    print()
    
    sequences_written = 0
    sequences_error = 0
    
    # Stream-write sequences
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for idx, wmn_data in enumerate(raw_annotations):
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(raw_annotations)} sequences...", end='\r')
            
            try:
                # Get dialogue utterances from corpus
                dialogue_utts = corpus_obj.get_dialogue(
                    wmn_data['corpus_codename'],
                    wmn_data['dialogue_id']
                )
                
                # Extract utterance data (minimal fields)
                utterances = []
                for utt in dialogue_utts:
                    utterances.append({
                        "author": utt.author,
                        "text": utt.text,
                        "id": getattr(utt, 'id', None),
                        "author_plain": getattr(utt, 'author_plain', None),
                        "reply_to": getattr(utt, 'reply_to', None),
                        "utt_order_num": getattr(utt, 'utt_order_num', None)
                    })
                
                # Create minimal sequence entry
                sequence_entry = {
                    "wmn_id": wmn_data['wmn_id'],
                    "corpus_codename": wmn_data['corpus_codename'],
                    "dialogue_id": wmn_data['dialogue_id'],
                    "wmn_type": wmn_data['wmn'],
                    "prediction": wmn_data.get('prediction'),
                    "regex_which_matched": wmn_data.get('regex_which_matched', []),
                    "utterances": utterances
                }
                
                # Write immediately (compact JSON)
                outfile.write(json.dumps(sequence_entry, separators=(',', ':'), ensure_ascii=False) + '\n')
                sequences_written += 1
                
            except Exception as e:
                # Write error entry
                error_entry = {
                    "wmn_id": wmn_data['wmn_id'],
                    "corpus_codename": wmn_data['corpus_codename'],
                    "dialogue_id": wmn_data['dialogue_id'],
                    "wmn_type": wmn_data['wmn'],
                    "prediction": wmn_data.get('prediction'),
                    "utterances": [],
                    "error": str(e)
                }
                outfile.write(json.dumps(error_entry, separators=(',', ':'), ensure_ascii=False) + '\n')
                sequences_error += 1
                print(f"\n  Warning: Error processing {wmn_data['wmn_id']}: {e}")
    
    print(f"\n")
    print("=" * 70)
    print("Extraction Complete!")
    print("=" * 70)
    print(f"  Successfully written: {sequences_written} sequences")
    print(f"  Errors: {sequences_error} sequences")
    print(f"  Output: {output_file}")
    print(f"  Format: JSONL (one JSON object per line)")
    print()


if __name__ == "__main__":
    main()
