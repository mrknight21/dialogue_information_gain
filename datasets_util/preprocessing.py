import pandas as pd
import numpy as np
import json
from nltk.tokenize import sent_tokenize
import os
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Alignment
import glob
import random
import shutil
import string

from convokit import Corpus, download
corpus = Corpus(filename=download("iq2-corpus"))
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))


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

def identify_watermark_strings(text_list, mini_gram=3, mini_count=40):
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
        return None

    # Find all keys that have this maximum value
    watermarks_candidates = [key for key, value in ngram_counts.items() if value == max_value]
    watermark = max(watermarks_candidates, key=len)
    return watermark



def get_insq_data():

    for i, convo in enumerate(corpus.iter_conversations()):
        dialogue = []
        speaker_options = ["0 (Unknown)", "1 (Self)", "2 (Everyone)", "3 (Audience)"]
        debaters_dict = {"against": {}, "for": {}}
        debaters = []
        moderator = ""

        dialogue_text = [d.text for d in convo.iter_utterances()]
        watermark = identify_watermark_strings(dialogue_text)


        for utt in convo.iter_utterances():

            text = utt.text
            if watermark:
                if watermark in text:
                    text = text.replace(watermark + " ", "")

            if utt.meta['speakertype'] in ["for", "against"] and utt.speaker_.id not in debaters:
                debaters_dict[utt.meta['speakertype']][utt.speaker_.id] = {"introduction":[], "conclusion": []}
                debaters.append(utt.speaker_.id)

            # if sentence_level and utt.meta['segment'] == 1 and utt.meta['speakertype'] == "mod":
            if utt.meta['speakertype'] == "mod":
                moderator = utt.speaker_.id

            d = {
                "id": utt.id,
                "speaker": utt.speaker_.id,
                "role": utt.meta['speakertype'],
                "segment": utt.meta['segment'],
                "text": text,
                "non-text": utt.meta["nontext"],
                "informational motive": 0,
                "social motive": 0,
                "coordinative motive": 0,
                "dialogue act": "",
                "target speaker": ""
            }
            dialogue.append(d)

        for s in debaters_dict["for"]:
            s_ind = len(speaker_options)
            option_string = f"{str(s_ind)} ({s}- for)"
            speaker_options.append(option_string)

        for s in debaters_dict["against"]:
            s_ind = len(speaker_options)
            option_string = f"{str(s_ind)} ({s}- against)"
            speaker_options.append(option_string)

        intros = [(d["speaker"], d["text"]) for d in dialogue  if d["segment"] == 0]
        conclusions =  [(d["speaker"], d["text"]) for d in dialogue  if d["segment"] == 2]


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

        pass


if __name__ == "__main__":

    get_insq_data()

    # preprocess_dialogue()

