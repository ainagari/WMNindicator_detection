#!/usr/bin/env python3
"""
Create Indicators Dataset from NeWMe Corpus

This script processes the extracted NeWMe corpus and creates the Indicators dataset
used in the paper. It applies preprocessing, sentence selection, and creates
train/dev/test splits based on pre-determined id lists.

Two versions are created:
- indicators_dataset_random.json: Using train-rd_ids.txt (random training selection)
- indicators_dataset_regexaware.json: Using train-rx_ids.txt (regex-aware training selection)

DEPENDENCIES:
- stanza: For sentence tokenization (matching original preprocessing)
- Install: pip install stanza
- Download models: python -c "import stanza; stanza.download('en')"
"""

import json
import re
from pathlib import Path
import stanza
#from nltk import word_tokenize


def simplify(utterance):
    """Simplify Switchboard utterances by removing annotations and special characters."""
    # Remove curly brackets annotations like {D So, }
    curly_brackets_pattern = re.compile(r"\{[a-z|A-Z] .*?\}")
    out = re.finditer(curly_brackets_pattern, utterance)
    spans_to_remove = []  # second index will not be included
    for match in out:
        beginning = (match.span()[0], match.span()[0]+3)
        end = (match.span()[1]-2, match.span()[1])
        spans_to_remove.append(beginning)
        spans_to_remove.append(end)
    new_utterance = ''
    prev_idx = 0
    for span_to_remove in spans_to_remove:
        new_utterance += utterance[prev_idx:span_to_remove[0]]
        prev_idx = span_to_remove[1]
    new_utterance += utterance[prev_idx:]

    current_utterance = new_utterance

    # Remove laughter and other codes like <laughter>
    laughter_pattern = re.compile(r"\<(.*?)\>")
    out = re.finditer(laughter_pattern, current_utterance)
    spans_to_remove = [match.span() for match in out]

    if spans_to_remove:
        new_utterance = ''
        prev_idx = 0
        for span_to_remove in spans_to_remove:
            new_utterance += current_utterance[prev_idx:span_to_remove[0]]
            prev_idx = span_to_remove[1]
        new_utterance += current_utterance[prev_idx:]
    else:
        new_utterance = current_utterance

    current_utterance = new_utterance

    # Strip punctuation
    new_utterance = current_utterance.replace("/", "")
    new_utterance = new_utterance.replace("+", "")
    new_utterance = new_utterance.replace("--", "")
    new_utterance = new_utterance.replace(" -", "")
    new_utterance = new_utterance.replace("]", " ")
    new_utterance = new_utterance.replace("[", " ")
    new_utterance = new_utterance.replace("#", " ")
    new_utterance = new_utterance.replace("((", " ")
    new_utterance = new_utterance.replace("))", " ")
    new_utterance = new_utterance.replace("  ", " ")
    new_utterance = new_utterance.replace("  ", " ")

    words = word_tokenize(new_utterance)
    new_utterance = " ".join(words)

    return new_utterance

# Initialize Stanza pipeline for sentence tokenization
print("Loading Stanza NLP pipeline...")
nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False, download_method=None)
print("Stanza loaded.")


def selecting_sentence(utterance, indicator_span):
    """
    Select the sentence containing the indicator span using Stanza.
    
    If the indicator is contained in one sentence, return that sentence.
    If the indicator spans multiple sentences, return the sentence with maximum coverage.
    
    Args:
        utterance: Full utterance text
        indicator_span: Tuple of (start_offset, end_offset) for the indicator
    
    Returns:
        str: The selected sentence containing the indicator
    """
    # Use Stanza for sentence tokenization
    doc = nlp(utterance)
    
    # Try to find a sentence that fully contains the indicator
    chosen_sentence = ''
    for sentence in doc.sentences:
        first_char = sentence.tokens[0].start_char
        last_char = sentence.tokens[-1].end_char
        if first_char <= indicator_span[0] and last_char >= indicator_span[1]:
            chosen_sentence = sentence.text
            break
    
    # If indicator spans multiple sentences, find the one with maximum coverage
    if not chosen_sentence:
        max_coverage = 0
        for sentence in doc.sentences:
            first_char = sentence.tokens[0].start_char
            last_char = sentence.tokens[-1].end_char
            
            # Calculate how many characters of the indicator are covered
            if first_char <= indicator_span[0] and last_char < indicator_span[1]:
                # Sentence contains beginning of indicator
                covered_chars = last_char - indicator_span[0]
            elif first_char > indicator_span[0] and last_char >= indicator_span[1]:
                # Sentence contains end of indicator
                covered_chars = indicator_span[1] - first_char
            elif first_char >= indicator_span[0] and last_char <= indicator_span[1]:
                # Sentence is fully contained within indicator
                covered_chars = last_char - first_char
            else:
                continue
            
            if covered_chars > max_coverage:
                max_coverage = covered_chars
                chosen_sentence = sentence.text
    
    # If still no sentence found, return the entire utterance
    if not chosen_sentence:
        chosen_sentence = utterance
    
    return chosen_sentence


def remove_cmv_extra_text(utterance):
    """
    Remove CMV (ChangeMyView) footer text from Reddit utterances.
    
    Removes text after "Hello, users of CMV! This is a footnote"
    which appears at the end of many CMV posts.
    
    Args:
        utterance: The utterance text
    
    Returns:
        str: Cleaned utterance text
    """
    # Remove anything after the CMV footnote (with HTML entity)
    try:
        match = re.search(r"\n\n&gt; \*Hello, users of CMV! This is a", utterance)
        if match:
            utterance = utterance[:match.span()[0]]
    except:
        pass
    
    # Remove anything after the CMV footnote (plain text)
    try:
        match = re.search(r"Hello, users of CMV! This is a footnote", utterance)
        if match:
            utterance = utterance[:match.span()[0]]
    except:
        pass
    
    return utterance


def get_context(sequence, start_turn_num, indicator_turn, number_previous_turns=3):
    """
    Extract past and future context for an indicator.
    
    For non-Reddit: Simple chronological context (previous N turns + next turn)
    For Reddit: Thread-aware context (follows reply_to chain + finds replies)
    
    Args:
        sequence: The full sequence dict from NeWMe corpus
        start_turn_num: Index of the indicator utterance
        indicator_turn: The indicator utterance dict
        number_previous_turns: Number of previous turns to include (default: 3)
    
    Returns:
        tuple: (past_context list, future_context list)
    """
    corpus = sequence.get('corpus', {}).get('codename', sequence.get('corpus_codename', ''))
    utterances = sequence['utterances']
    
    if corpus != 'winning-args-corpus':
        # Simple chronological context for Switchboard and BNC
        start_idx = max(0, start_turn_num - number_previous_turns)
        past_context = utterances[start_idx:start_turn_num]
        
        future_context = []
        if start_turn_num + 1 < len(utterances):
            future_context = [utterances[start_turn_num + 1]]
        
        return past_context, future_context
    
    else:
        # Reddit: Thread-aware context following reply_to chain
        ####### PAST CONTEXT #######
        current_replying_to = indicator_turn.get('reply_to')
        
        context = []
        for i in range(1, number_previous_turns + 1):
            if not current_replying_to:
                # Add title if this doesn't reply to anything
                context.append(utterances[0])
                break
            
            # Find parent utterance
            parent = [utt for utt in utterances if utt.get('id') == current_replying_to or utt.get('utt_order_num') == current_replying_to]
            if not parent:
                break
            parent = parent[0]
            context.append(parent)
            
            current_replying_to = parent.get('reply_to')
        
        context.reverse()
        
        # Clean up author fields
        for element in context:
            if 'author_plain' in element:
                element['author'] = element['author_plain']
        
        # Fix TITLE author to match first commenter
        for i, element in enumerate(context):
            if element.get('author') == "TITLE" and len(context) > i + 1:
                element['author'] = context[i + 1]['author']
        
        past_context = context
        
        ####### FUTURE CONTEXT #######
        # Find utterances that reply to the indicator
        current_utt_id = indicator_turn.get('id') or indicator_turn.get('utt_order_num')
        
        future_context_candidates = []
        for utt in utterances:
            if utt.get('reply_to') == current_utt_id:
                if 'author_plain' in utt:
                    utt['author'] = utt['author_plain']
                future_context_candidates.append(utt)
        
        if not future_context_candidates:
            future_context = []
        else:
            # Prefer response from same speaker as previous context
            previous_speaker = past_context[-1]['author'] if past_context else ''
            
            future_context = []
            if previous_speaker:
                for utt in future_context_candidates:
                    if utt['author'] == previous_speaker:
                        future_context = [utt]
                        break
            
            if not future_context:
                future_context = [future_context_candidates[0]]
        
        return past_context, future_context


def create_instance(sequence, wmn_id, subset_name):
    """
    Create an Indicators dataset instance from a NeWMe sequence.
    
    Args:
        sequence: A sequence from the NeWMe corpus
        wmn_id: The WMN ID for this instance
        subset_name: 'train', 'dev', or 'test'
    
    Returns:
        dict: Formatted instance for the Indicators dataset, or None if invalid
    """
    corpus_name = sequence.get('corpus_codename', '')
    
    # Map corpus codenames to display names
    corpus_display = {
        'switchboard-corpus': 'swda',
        'bnc': 'BNC',
        'winning-args-corpus': 'Reddit'
    }
    corpus = corpus_display.get(corpus_name, corpus_name)
    
    utterances = sequence.get('utterances', [])
    if not utterances:
        return None
    
    # Get the prediction (contains the indicator span)
    prediction = sequence.get('prediction')
    
    # Get wmn_type early for Nothing instance check
    wmn_type = sequence.get('wmn_type', 'Nothing')
    
    # Check if this is a Nothing instance
    is_nothing = wmn_type == 'Nothing'
    
    if not prediction or 'label' not in prediction:
        # No indicator span - use first utterance as target
        target_idx = 0
        target_utt = utterances[target_idx]
        span = [0, len(target_utt['text'])]
        sentence = target_utt['text']
    else:
        pred_label = prediction['label']
        start_idx = pred_label['start_index']
        end_idx = pred_label['end_index']
        start_offset = pred_label['start_offset']
        end_offset = pred_label['end_offset']
        
        target_idx = start_idx
        
        if target_idx >= len(utterances):
            return None
        
        target_utt = utterances[target_idx]
        
        # Store original text for Switchboard
        original_target_text = target_utt['text']
        
        # Apply simplification to Switchboard
        if corpus == 'SWDA':
            target_utt['text'] = simplify(target_utt['text'])
        
        # Apply CMV text cleaning for Reddit
        if corpus == 'Reddit':
            target_utt['text'] = remove_cmv_extra_text(target_utt['text'])
        
        # For Nothing instances, use full utterance span
        if is_nothing:
            span = [0, len(target_utt['text'])]
            indicator_text = target_utt['text']
        else:
            # Extract indicator span for non-Nothing instances
            if start_idx == end_idx:
                # Single utterance span
                if corpus == 'SWDA':
                    # Re-find indicator in simplified text
                    indicator_text_original = original_target_text[start_offset:end_offset]
                    simplified_indicator = simplify(indicator_text_original)
                    match = re.search(re.escape(simplified_indicator), target_utt['text'])
                    if match:
                        span = list(match.span())
                    elif simplified_indicator == target_utt['text']:
                        span = [0, len(simplified_indicator)]
                    elif "On a like that 's not a bad size fish" in simplified_indicator:
                        span = [0, len(target_utt['text'])]
                    else:
                        # Fallback: use full utterance
                        span = [0, len(target_utt['text'])]
                else:
                    # Non-SWDA corpus, use offsets directly
                    span = [start_offset, end_offset]
            else:
                # Multi-utterance span - use full target utterance
                span = [0, len(target_utt['text'])]
            indicator_text = target_utt['text'][span[0]:span[1]]
        
        # Select the sentence containing the indicator
        sentence = selecting_sentence(target_utt['text'], span)
        if not sentence:
            sentence = target_utt['text']
    
    # Get context
    past_context, future_context = get_context(sequence, target_idx, target_utt)
    
    # Calculate displayed_position (utterance's position in dialogue)
    target_displayed_position = target_idx
    
    # Clean context utterances
    past_context_clean = []
    for utt in past_context:
        # Handle BNC IDs (None → use displayed_position)
        utt_id = utt.get('id')
        if utt_id is None:
            utt_id = utt.get('displayed_position', utt.get('utt_order_num', ''))
        
        utt_clean = {
            'author': utt.get('author', ''),
            'text': utt.get('text', ''),
            'id': str(utt_id)
        }
        if corpus == 'SWDA':
            utt_clean['text'] = simplify(utt_clean['text'])
        if corpus == 'Reddit':
            utt_clean['text'] = remove_cmv_extra_text(utt_clean['text'])
        past_context_clean.append(utt_clean)
    
    future_context_clean = []
    for utt in future_context:
        # Handle BNC IDs (None → use displayed_position)
        utt_id = utt.get('id')
        if utt_id is None:
            utt_id = utt.get('displayed_position', utt.get('utt_order_num', ''))
        
        utt_clean = {
            'author': utt.get('author', ''),
            'text': utt.get('text', ''),
            'id': str(utt_id)
        }
        if corpus == 'SWDA':
            utt_clean['text'] = simplify(utt_clean['text'])
        if corpus == 'Reddit':
            utt_clean['text'] = remove_cmv_extra_text(utt_clean['text'])
        future_context_clean.append(utt_clean)
    
    # Get regex patterns (if available)
    regex_patterns = []
    if prediction and 'regex_which_matched' in sequence:
        regex_patterns = sequence.get('regex_which_matched', [])
    
    # Determine label (1 for positive WMN types, 0 for negative/Nothing)
    positive_labels = [
        "WMN: non-understanding", "WMN: disagreement", "WMN: other",
        "SIMN", "Non-pursued", "Without trigger"
    ]
    label = 1 if wmn_type in positive_labels else 0
    
    # Generate ID for utterances without one (BNC)
    target_id = target_utt.get('id')
    if target_id is None:
        # BNC format: just the position number as string
        target_id = str(target_displayed_position)
    
    # Convert utt_order_num to int
    target_utt_order_num = target_utt.get('utt_order_num', target_idx)
    if isinstance(target_utt_order_num, str):
        try:
            target_utt_order_num = int(target_utt_order_num)
        except (ValueError, TypeError):
            target_utt_order_num = target_idx
    
    # Create instance
    instance = {
        'target': {
            'author': target_utt.get('author', ''),
            'text': target_utt['text'],
            'id': str(target_id),
            'displayed_position': target_displayed_position,
            'utt_order_num': target_utt_order_num,
            'span': span,
            'sentence': sentence
        },
        'past_context': past_context_clean,
        'future_context': future_context_clean,
        'corpus': corpus,
        'wmn-label': wmn_type,
        'regex': regex_patterns,
        'id': wmn_id,
        'label': label,
        'subset': subset_name
    }
    
    return instance


def load_id_file(filepath):
    """Load WMN IDs from a text file (one ID per line)."""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def create_indicators_dataset(newme_corpus_file, ids_dir, output_file, train_ids_filename):
    """
    Create Indicators dataset from NeWMe corpus and ID splits.
    
    Args:
        newme_corpus_file: Path to extracted_browsable_data_full.json
        ids_dir: Directory containing the ID split files
        output_file: Output path for the Indicators dataset
        train_ids_filename: Filename for training IDs ('train-rd_ids.txt' or 'train-rx_ids.txt')
    """
    print("=" * 70)
    print(f"Creating Indicators Dataset: {output_file}")
    print("=" * 70)
    print()
    
    # Load NeWMe corpus (JSONL format - streaming to avoid memory issues)
    print(f"Loading NeWMe corpus from: {newme_corpus_file}")
    print("  Processing in streaming mode (JSONL format)...")
    
    sequences_by_id = {}
    total_sequences = 0
    
    with open(newme_corpus_file) as f:
        for line in f:
            if line.strip():
                seq = json.loads(line)
                sequences_by_id[seq['wmn_id']] = seq
                total_sequences += 1
                if total_sequences % 1000 == 0:
                    print(f"    Loaded {total_sequences} sequences...", end='\r')
    
    print(f"\n  Loaded {total_sequences} sequences")
    print()
    
    # Load ID splits
    print(f"Loading ID splits from: {ids_dir}")
    train_ids = load_id_file(Path(ids_dir) / train_ids_filename)
    dev_ids = load_id_file(Path(ids_dir) / 'dev_ids.txt')
    test_ids = load_id_file(Path(ids_dir) / 'test_ids.txt')
    
    print(f"  Train IDs: {len(train_ids)}")
    print(f"  Dev IDs: {len(dev_ids)}")
    print(f"  Test IDs: {len(test_ids)}")
    print()
    
    # Create dataset
    print("Processing instances...")
    dataset = {
        'train': [],
        'dev': [],
        'test': []
    }
    
    # Process each split
    for subset_name, id_list in [('train', train_ids), ('dev', dev_ids), ('test', test_ids)]:
        print(f"  Processing {subset_name}...")
        skipped = 0
        
        for wmn_id in id_list:
            if wmn_id not in sequences_by_id:
                print(f"    Warning: ID '{wmn_id}' not found in corpus")
                skipped += 1
                continue
            
            sequence = sequences_by_id[wmn_id]
            instance = create_instance(sequence, wmn_id, subset_name)
            
            if instance:
                dataset[subset_name].append(instance)
            else:
                skipped += 1
        
        print(f"    Created: {len(dataset[subset_name])} instances")
        if skipped > 0:
            print(f"    Skipped: {skipped} instances")
    
    print()
    
    # Label distribution
    print("Label distribution:")
    for subset_name in ['train', 'dev', 'test']:
        instances = dataset[subset_name]
        label_counts = {0: 0, 1: 0}
        for inst in instances:
            label_counts[inst['label']] += 1
        print(f"  {subset_name}:")
        print(f"    Label 0 (negative): {label_counts[0]}")
        print(f"    Label 1 (positive): {label_counts[1]}")
    print()
    
    # Save dataset
    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  Saved {sum(len(v) for v in dataset.values())} total instances")
    print()
    
    print("=" * 70)
    print("Complete!")
    print("=" * 70)


def main():
    """Main function to create both dataset versions."""
    
    # Paths
    newme_corpus_file = Path(__file__).parent / 'extracted_newme.jsonl'
    ids_dir = 'dataset_ids'
    output_dir = "../data/"
    
    # Check if input file exists
    if not newme_corpus_file.exists():
        print(f"Error: NeWMe corpus file not found: {newme_corpus_file}")
        print("Please run extract_browsable_data.py first.")
        return
    
    # Create train-rd version
    print("\n")
    create_indicators_dataset(
        newme_corpus_file,
        ids_dir,
        output_dir / 'indicators_dataset_random.json',
        'train-rd_ids.txt'
    )
    
    # Create train-rx version
    print("\n" * 2)
    create_indicators_dataset(
        newme_corpus_file,
        ids_dir,
        output_dir / 'indicators_dataset_regexaware.json',
        'train-rx_ids.txt'
    )


if __name__ == '__main__':
    main()
