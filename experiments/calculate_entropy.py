import numpy as np
import os, sys
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
import itertools

import argparse
from types import SimpleNamespace
import logging
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
# Functions for saving during training already exist
from transformers import AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, DatasetDict

from utils.dataloaders import _add_to_text, create_context, create_full_context, create_context_dataset_from_df, create_context_dataset, get_perplexity_encodings, create_data_modeltrain, _anytype_context
from utils.entropy_computation import batch_predict_logits_rnn, batch_predict_logits_lm, compute_perplexity
from utils.entropy_computation import sentence_predict, test_predict_entropy, batch_predict_entropy, results_to_df
from utils.entropy_computation import pivot_results_df
from datasets_util.insq_dataset import load_insq

DEVICE = "mps"

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", type=str, required=True,
                        help="Path to the file containing testing arguments & data info")

    args = parser.parse_args()
    assert os.path.exists(args.conf_file)
    with open(args.conf_file, "r") as f:
        test_config = json.load(f)

    # Model
    args.model = test_config['Model']['model_name']
    args.tokenizer = args.model if test_config['Model'].get('tokenizer', None) is None else test_config['Model'][
        'tokenizer']
    args.tok_maxlength = 1024 if test_config['Model'].get('tok_maxlength', None) is None else test_config['Model'][
        'tok_maxlength']
    args.load_from_checkpoint = test_config['Model'].get('load_from_checkpoint', None)  # path to checkpoint
    args.local_only = test_config['Model'].get('local_files_only', False)

    # Data
    args.data = test_config['Dataset']['path']
    args.corpus = test_config['Dataset']['corpus']
    args.column_option = test_config['Dataset']['column_option']
    # DataLoader
    args.dataloader_kwargs = test_config.get("DataLoader", {"sep_token": "eos", "max_length": 150})

    # Experiments to run
    args.experiments = test_config.get("Computation", ["full_context"])
    # possible keys: "progressive_entropy" (ie context 0 to 8), "full_context" (ie 0 and full_context)
    # each experiment will generate different files
    # which sep is loaded in DataLoader config

    return args


def create_context_dataset(dataframe: pd.DataFrame, tokenizer, files_train: list = None, files_test: list = None,
                           text_col: str = 'text', file_col: str = 'file', index_col: str = 'index',
                           speaker_col='speaker',
                           max_length: int = 1024, batch_size: int = 8,
                           sep_token: str = ' ', sep_context_sent: bool = True, **text_kwargs):
    """All-in-one function: Create context and dataset all at once dataset

    Input:
    --------
    sep_token: str, default ' ' (space), can be tokenizer.eos_token
    sep_context_sent: bool, default False ie no specific separator between context and sentence; if True only one separator between context and last sentence, context is concat
    """
    # Copying df to avoid issues rerunning the function
    df = dataframe.copy(deep=True)

    # Parametrize
    join_sep = ' ' if sep_context_sent else sep_token  # default token or space (ie no token) if only separating test sentence from context
    # Same but with tokenizes id
    sep_tok = tokenizer.encode(sep_token) if sep_token != ' ' else []  # tokenized sep_token
    join_st = sep_tok if not sep_context_sent else []  # tokenized join_sep: default token / no token if only separating test sentence from context
    rev_join_st = sep_tok if sep_context_sent else []  # token to add between context and test sentence - ie, if sep_tok has been added to globally, no need to add it for final concat

    # Update text with params
    df[f'{text_col}_u'] = _add_to_text(df, text_col=text_col, speaker_col=speaker_col, file_col=file_col,
                                       **text_kwargs)  # whether adding speaker tokens
    text_col = f'{text_col}_u'  # for further usage
    # Create ids with sep & params
    df['text_input_ids'] = df[text_col].apply(
        lambda x: join_st + tokenizer(x, truncation=True, padding=False)['input_ids'])  # tokenizing and adding sep
    df['length'] = df.text_input_ids.apply(len) - len(
        join_st)  # length errors with 1rst line since sep_token = tokenizer.eos_token adds 1 in length

    df['file'] = df.file.apply(lambda x: " ".join(x.split("_")[2:]))

    # def prepend_ll(x:list, join_pat:list):
    #    """Add this pattern to each list element"""
    #    return [join_pat + y for y in x]

    c = df.groupby(file_col).agg({
        text_col: lambda x: [join_sep + join_sep.join(list(x)[i-1:i]) for i in range(len(x))],
        # if using +1 then not adding the last line (concat)
        'text_input_ids': lambda x: [(list(itertools.chain(*list(x)[i-1:i]))) for i in range(len(x))],
        # if using +1 then not adding the last line (concat)
        index_col: list
    })
    if pd.__version__ >= '1.3.0':
        c = c.explode([text_col, index_col, 'text_input_ids']).reset_index(drop=False)
    else:  # for earlier version list is not accepted
        # chaining explodes doesn't work (n**3) >>> taking index i from the list - if issue then data is not correctly indexed
        c = c.explode(index_col).reset_index(drop=False)
        c[text_col] = c.apply(lambda x: x[text_col][x[index_col]], axis=1)
        c['text_input_ids'] = c.apply(lambda x: x['text_input_ids'][x[index_col]], axis=1)
    df = pd.merge(left=df, left_on=[file_col, index_col], right=c, right_on=[file_col, index_col], how='outer',
                  suffixes=('', '_full'))  # merging bc index might not be ordered identically bc of groupby
    # full_text is currently only context, adding text and separator:
    df[f'{text_col}_full'] = (df[f'{text_col}_full'] + sep_token + df[text_col]).apply(
        lambda x: x.strip().replace('  ', ' ').replace(sep_token + sep_token, sep_token))
    df['input_ids'] = df.apply(lambda x: (x['text_input_ids_full'] + rev_join_st + x['text_input_ids'])[-max_length:],
                               axis=1)  # keep only the correct number of elements

    # Exception made for 1rst line of file if sep_token == ' ': - one token missing
    if sep_token == ' ':
        df.loc[df[index_col] == 0, 'input_ids'] = df.loc[df[index_col] == 0].input_ids.apply(
            lambda x: (tokenizer.encode(tokenizer.eos_token) + x)[-max_length:])

    df['start_idx'] = df.apply(lambda x: len(x.input_ids) - x.length, axis=1)
    df['attention_mask'] = df.input_ids.apply(lambda x: [1] * len(x))

    # Dataset
    if files_train is not None:
        dataset_c = DatasetDict({
            'train': Dataset.from_pandas(df[df[file_col].isin(files_train)]),
            'test': Dataset.from_pandas(df[df[file_col].isin(files_test)])
        })
    else:
        dataset_c = Dataset.from_pandas(df)

    # drop extra columns from dataframe
    # df.drop(['text_input_ids', 'text_input_ids_full', 'attention_mask'], axis=1, inplace = True)
    return dataset_c, df

def apply_pred_steps(df3, dataset_a, model, tokenizer):
    """
    df3: test pd.DataFrame
    dataset_a: finalised Dataset structure
    """
    dataset_a.set_format(type='torch', columns=['input_ids', 'start_idx', 'attention_mask'])
    test_dataloader = DataLoader(dataset_a, collate_fn=data_collator, batch_size=1, worker_init_fn=SEED)
    sent_avg_logp, tokens_logp, sent_length, sentence_tokens = test_predict_entropy(model, test_dataloader, tokenizer, DEVICE, batch_predict_logits_lm)
    test_dataframe = results_to_df(df3, sent_avg_logp, tokens_logp, sent_length, sentence_tokens = sentence_tokens, out_file_name = None)
    return test_dataframe


if __name__ == '__main__':
    args = parse_arguments()
    # default arguments
    # args = SimpleNamespace()
    start = datetime.now()
    MODEL_NAME = args.model

    ###### Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tok_cont_kwargs = {'truncation':True, 'padding':'max_length', 'max_length': args.tok_maxlength}

    logging.info('Loading data...')
    df = load_insq(-1, ["dev"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    df = pd.read_csv("../cache/insq_dev.csv", keep_default_na=False, na_values=[''], index_col=0)
    logging.warning(f"Dropping {df.text.isna().sum() + (df.text == '').sum()} rows")
    df = df[~(df.text.isna()) & (df.text != "")]
    dl_function = create_context_dataset
    dataset_a, df3 = dl_function(df, tokenizer, **args.dataloader_kwargs, **args.column_option)

    ###### Model
    logging.info("Loading model")

    if args.local_only:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # if args.load_from_checkpoint is not None:
    #     model, _, epoch, _ = checkpoint_load(args.load_from_checkpoint, model)
    model.to(DEVICE)  # loading model

    # saving df
    ptest_dataframe = []
    out_file_name = "../cache/insq_dev_c1_gpt2_entropy.csv"

    # No context outside of loop, always returned
    logging.info(f'\n------------- PARAM NO CONTEXT -------------')
    # only need to recreate dataset, not df, since using eos parametrized
    # df3['input_ids'] = df3['text_input_ids']  # no context
    df3['start_idx'] = df3.apply(lambda x: len(x.input_ids) - x.length, axis=1)
    df3['attention_mask'] = df3.input_ids.apply(lambda x: [1] * len(x))
    dataset_a = Dataset.from_pandas(df3)
    # dataset_a.set_format(type='torch', columns=['input_ids', 'start_idx', 'attention_mask'])
    # test_dataloader = DataLoader(dataset_a, collate_fn=data_collator, batch_size=1, worker_init_fn=SEED)
    # sent_avg_logp, tokens_logp, sent_length, sentence_tokens = test_predict_entropy(model, test_dataloader, tokenizer, DEVICE, batch_predict_logits_lm)
    # test_dataframe = results_to_df(df3, sent_avg_logp, tokens_logp, sent_length, sentence_tokens = sentence_tokens, out_file_name = None)
    test_dataframe = apply_pred_steps(df3, dataset_a, model, tokenizer)

    test_dataframe['mode'] = 'no-context'
    ptest_dataframe.append(test_dataframe.copy())

    # Output
    logging.info(f'Saving dataframe to {out_file_name}')
    test_dataframe['model'] = MODEL_NAME
    ptest_dataframe = pd.concat(ptest_dataframe, axis=0, ignore_index=True)
    ptest_dataframe = ptest_dataframe[[col for col in ptest_dataframe if
                                       col not in ['text_input_ids_full', 'text_input_ids', 'input_ids',
                                                   'attention_mask']]]
    ptest_dataframe.to_csv(out_file_name, index=False)