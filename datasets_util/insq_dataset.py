
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import string
from glob import glob
import json

import pandas as pd
from convokit import Corpus, download

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))


def load_json_data(path):
    with open(path) as f:
        json_objs = json.load(f)
        return json_objs

def extract_longest_sublist(lst, input_id, moderator):
    max_length = 0
    start_index = -1
    end_index = -1
    n = len(lst)

    for i in range(n):
        if lst[i][0] == input_id:
            last_seen_input_id = i
            for j in range(i, n):
                if lst[j][0] in [input_id, moderator]:
                    if lst[j][0] == input_id:
                        last_seen_input_id = j
                        current_length = last_seen_input_id - i + 1
                        if current_length > max_length:
                            max_length = current_length
                            start_index = i
                            end_index = last_seen_input_id
                else:
                    break
    if start_index != -1 and end_index != -1:
        return lst[start_index:end_index+1]
    else:
        return []


import re
from collections import defaultdict

def identify_watermark_strings(text_list, mini_gram=3, mini_count=50):
    """
    Identifies watermark strings in a list of text.

    Args:
        text_list (list of str): The list of text to search for watermark strings.

    Returns:
        list of str: A list of identified watermark strings.
    """
    # Concatenate all texts into one string
    full_text = ' '.join(text_list)

    # Split the text into sentences
    # This regex splits on '.', '?', '!', and handles multiple occurrences and abbreviations.
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    sentences = sentence_endings.split(full_text)
    ngram_counts = defaultdict(int)

    for sent in sentences:
        tokens = sent.split(" ")
        tokens_len = len(tokens)
        substrings = ""
        gram_count = 0
        last_t = ""

        for t in tokens:
            clear_t = t.strip().strip(string.punctuation + '“' + '”')
            if clear_t == "":
                if substrings:
                    substrings += " " + t
            elif clear_t[0].isupper():
                if substrings:
                    substrings += " " + t
                    gram_count += 1
                else:
                    if clear_t.lower() not in stops or clear_t == "The":
                        substrings = t
                        gram_count += 1
            else:
                if substrings:
                    if last_t != "" and last_t[0].isupper() and clear_t in stops:
                        substrings += " " + t
                        gram_count += 1

                    else:
                        gram_count = 0
                        substrings = ""
                        continue

            if substrings and gram_count >= mini_gram:
                ngram_counts[substrings] += 1
            last_t = clear_t

    # Count the frequency of each sentence
    max_value = max(ngram_counts.values())

    if max_value < mini_count:
        return None, max_value

    # Find all keys that have this maximum value
    watermarks_candidates = [key for key, value in ngram_counts.items() if value == max_value]
    watermark = max(watermarks_candidates, key=len)
    return watermark, max_value

def extract_introduction_conclusion(dialogue, debaters, debaters_dict, moderator):
    intros = [(d["speaker"], d["text"]) for d in dialogue if d["segment"] == 0]
    conclusions = [(d["speaker"], d["text"]) for d in dialogue if d["segment"] == 2]

    for debater in debaters:

        stance = None
        if debater in debaters_dict["against"]:
            stance = "against"
        else:
            stance = "for"

        opening = extract_longest_sublist(intros, debater, moderator)
        closing = extract_longest_sublist(conclusions, debater, moderator)
        debaters_dict[stance][debater]["introduction"] = opening
        debaters_dict[stance][debater]["conclusion"] = closing

    return debaters_dict

def get_conversation_meta(convo, extract_intro_conclu = False, short_bio = True):
    debaters_dict = {"against": {}, "for": {}}
    speakers = {}
    moderator = ""
    title = convo.meta["title"]
    speaker_options = ["0 (Unknown)", "1 (Self)", "2 (Everyone)", "3 (Audience)"]

    dialogue_text = [d.text for d in convo.iter_utterances()]
    watermark, watermark_count = identify_watermark_strings(dialogue_text)

    for utt in convo.iter_utterances():

        if utt.meta['speakertype'] in ["for", "against"] and utt.speaker_.id not in speakers :
            debaters_dict[utt.meta['speakertype']][utt.speaker_.id] = {"introduction": [], "conclusion": []}
            speaker = utt.speaker_
            if short_bio:
                bio = short_bio = speaker.meta["bio_short"]
            else:
                bio = speaker.meta["bio"]
            speakers[speaker.id] = {"bio": bio, "stance": utt.meta['speakertype']}

        # if sentence_level and utt.meta['segment'] == 1 and utt.meta['speakertype'] == "mod":
        if utt.meta['speakertype'] == "mod":
            moderator = utt.speaker_.id

    for s in debaters_dict["for"]:
        s_ind = len(speaker_options)
        option_string = f"{str(s_ind)} ({s}- for)"
        speaker_options.append(option_string)

    for s in debaters_dict["against"]:
        s_ind = len(speaker_options)
        option_string = f"{str(s_ind)} ({s}- against)"
        speaker_options.append(option_string)

    meta = {"title": title, "watermark": watermark, "speakers": speakers, "moderator": moderator}

    return meta

def load_annotation_labels():
    split_dict = {}
    data_source = glob(f"../cache/insq_labels/*.json")
    for d in data_source:
        data = load_json_data(d)
        key = d.split("/")[-1].replace(".json", "")
        topics = list(set([d["topic"] for d in data]))
        for t in topics:
            split_dict[t] = key
    return split_dict

def meta_to_string(meta):
    meta_string = f"title: {meta['title']} \n"
    meta_string += f"moderator: {meta['moderator']} \n"
    meta_string += f"speakers: \n"
    for s, prof in meta["speakers"]:
        meta_string += f"name: {s}, stance: {prof['stance']}, bio: {prof['bio']} \n"
    meta_string += "\n\n"
    return meta_string

def load_insq(context_len=5, splits=["dev", "train", "test"], remove_water_mark = True):
    corpus = Corpus(filename=download("iq2-corpus"))
    splits_dict = load_annotation_labels()
    dataset = []
    for i, convo in enumerate(corpus.iter_conversations()):

        dialogue = []
        meta = get_conversation_meta(convo)
        split = splits_dict.get(meta["title"], "train")
        if split not in splits:
            continue

        for i, utt in enumerate(convo.iter_utterances()):
            text = utt.text
            if meta["watermark"] and remove_water_mark:
                if meta["watermark"] in text:
                    text = text.replace(meta["watermark"] + " ", "")

            if context_len == -1:
                context = dialogue
            elif  context_len == 0:
                context = []
            else:
                context = dialogue[-context_len:]
            context_text = ""

            if len(context) != 0:
                for d in context:
                    context_text += f"{d['speaker']} ({d['role']}): {d['text']} \n"
            d = {
                "utt_id": utt.id,
                "topic": meta["title"],
                "split": split,
                "segment": utt.meta['segment'],
                "index": i,
                "speaker": utt.speaker_.id,
                "role": utt.meta['speakertype'],
                "context text": context_text,
                "text": text
            }
            dialogue.append(d)
        dataset.extend(dialogue)
    df = pd.DataFrame(dataset, columns=["utt_id", "topic", "split", "segment", "index", "speaker", "role", "context text", "text"])
    return df


def get_preprocessed_samsum(dataset_config, tokenizer, split):
    dataset = load_insq(split)

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset


if __name__ == "__main__":

    load_insq(3, splits=["dev"])