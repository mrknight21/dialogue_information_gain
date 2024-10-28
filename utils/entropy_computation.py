import numpy as np
import os
import pandas as pd
import time
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm

from torch import nn

LOG_2 = torch.log(torch.tensor(2.))


def sentence_predict(model, tokenizer, text, next_words=20):
    # note: model must not be on GPU
    x = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0)
    model.eval()

    for _ in range(0, next_words):
        y_pred = model(x)

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        x = torch.cat((x,torch.LongTensor([[word_index]])), dim=1) # concat despite batch axis

    words = tokenizer.decode(x[0])
    return x, words


def test_predict_entropy(lm, dataloader, tokenizer, device, batch_predict_logits):
    tokens_logp = []
    sent_avg_logp = []
    tokens = []
    iterator = tqdm(dataloader, desc='Iteration')
    for _, batch in enumerate(iterator):
        # call function
        batch_logp, batch_avg_logp, batch_tokens = batch_predict_entropy(lm, batch, tokenizer, device, batch_predict_logits)
        tokens_logp.extend(batch_logp)
        sent_avg_logp.extend(batch_avg_logp)
        tokens.extend(batch_tokens)

    iterator.close()
    sent_length = [len(x) for x in tokens_logp]
    return sent_avg_logp, tokens_logp, sent_length, tokens

def batch_predict_logits_lm(lm, batch):
    with torch.no_grad():
        outputs = lm(**{'input_ids':batch['input_ids'], 'attention_mask':batch['attention_mask']})
    return outputs.logits

def batch_predict_logits_rnn(lm, batch):
    """Use with non HuggingFace model - otherwise other arguments in the dictionary will generate TypeError, 
    also return argument is a dict not an object
    """
    with torch.no_grad():
        outputs = lm(batch['input_ids'])
    return outputs



def batch_predict_entropy(lm, batch, tokenizer, device, batch_predict_logits): # might add logger here
    """ For one batch, get the entropy of expected words 

    Input:
        lm: model
        batch: dataloader batch with 'input_ids', 'attention_mask' keys, 'start_idx' (optional)
        batch_predict_logits: function, which predict function to use 
    """
    batch_length, max_sent_length = batch['input_ids'].shape
    for k in ['input_ids','attention_mask']:
        batch[k] = batch[k].to(device) # put data on gpu
    if 'start_idx' not in batch:
        batch['start_idx'] = [0 for _ in range(batch_length)]
    
    # returns
    batch_avg_logp = []
    batch_logp = []
    batch_tokens = []

    # get predictions
    lm.eval()
    outputs_logits = batch_predict_logits(lm, batch)
    logp_w = log_softmax(outputs_logits, dim=-1)
    logp_w /= LOG_2 # TODO: check

    # for every sentence
    for s_id in range(batch_length): 
        sentence = batch['input_ids'][s_id,:]
        sentence_logp = []
        sentence_tokens = []
        # for every token
        # XXX: issue with a sentence with no context - 1rst token will not be planned
        for token_index in range(max(0,batch['start_idx'][s_id] - 1), max_sent_length-1): # -1 bc looking ahead
            pred_index = token_index + 1
            w_id = sentence[pred_index]
            # skip special tokens (BOS, EOS, PAD) + speaker token # TODO: add speaker tokens
            if w_id in tokenizer.all_special_ids: # and w_id != unk_id:
                # print('w_id in tokenizer.all_special_ids')
                continue
            # increase non-normalised log probability of the sentence
            token_logp = logp_w[s_id, token_index, w_id].item()
            sentence_logp.append(token_logp)
            sentence_tokens.append(tokenizer.decode([w_id]))
            #print(f"Sentence {s_id} word idx {token_index} next word id {w_id} is `{tokenizer.decode([w_id])}` - word entropy {token_logp}")
        # append to batch
        batch_logp.append(sentence_logp)
        sentence_logp = np.array(sentence_logp)
        batch_avg_logp.append(- sentence_logp.sum()/sentence_logp.shape[0])
        batch_tokens.append(sentence_tokens)
    
    return batch_logp, batch_avg_logp, batch_tokens


def results_to_df(dataframe:pd.DataFrame, sent_avg_logp:list, tokens_logp:list, sent_length:list,
                        out_file_name:str=None, sentence_tokens:list=None, column_post:str=None):
    # TODO: check all lengths
    dataframe['normalised_h'] = sent_avg_logp
    dataframe['length'] = sent_length
    dataframe['tokens_h'] = tokens_logp
    dataframe['sum_h'] = dataframe.normalised_h * dataframe.length
    # could add tokens to make sure which tokens
    if sentence_tokens is not None:
        dataframe['tokens'] = sentence_tokens

    h_bar = dataframe.groupby('length').agg({"normalised_h": "mean"}).to_dict()['normalised_h']
    dataframe['xu_h'] = dataframe.apply(lambda x: np.nan if x.length not in h_bar else x.normalised_h/h_bar[x.length], axis=1)

    if column_post is not None:
        dataframe.rename(columns={col:f'{col}{column_post}' for col in ['normalised_h','length','tokens_h','sum_h','xu_h']}, 
                            inplace=True)

    if out_file_name is not None:
        dataframe.to_csv(f'{out_file_name}.csv',index=False)
    return dataframe

def pivot_results_df(df:pd.DataFrame, post_patterns:list) -> pd.DataFrame:
    """Adapting results_to_df dataframe in case of multiple instances of one test set into several columns. 
    Cannot be done during inference since it changes the number of examples.
    """
    main_columns = [col for col in df.columns if 
            not any([pat in col for pat in ['xu_h','normalised_h','length','tokens_h','sum_h']])]
    pivot_df = []
    for pat in post_patterns:
        pat_columns = [f'normalised_h{pat}', f'length{pat}', f'tokens_h{pat}', 
                                    f'sum_h{pat}', f'xu_h{pat}']
        tmp = df[ main_columns + pat_columns]
        tmp.rename(columns={col:col.replace(pat,'') for col in pat_columns}, inplace=True)
        if 'model' in tmp.columns:
            tmp['model'] = tmp['model']+pat
        else:
            tmp['model'] = pat
        pivot_df.append(tmp)
    pivot_df = pd.concat(pivot_df)
    return pivot_df


#%% Perplexity
def compute_perplexity(model, encodings, device, stride:int=8, 
        model_is_lm:bool=True, 
        max_length:int=None, criterion = nn.CrossEntropyLoss() # if model is not LM
    ):
    """From HuggingFace documentation - https://huggingface.co/docs/transformers/perplexity
    Adapted for non HF models
    """
    if model_is_lm and max_length is None:
        max_length = model.config.n_positions

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            if model_is_lm:
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len
            else:
                outputs = model(input_ids[:,:-1]) 
                loss = criterion(outputs.transpose(1, 2), target_ids[:,1:])
                neg_log_likelihood = loss * trg_len # no .item(), need tensor

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()