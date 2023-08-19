import json
import torch
import os
import numpy as np

import random
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForLanguageModeling
from tools import print_rank
from collections import namedtuple, defaultdict


STOPWORDS = {'', ",", ";", ".", "(", "，", "。", "?", "？", "；", "【", "】", "：", "“", "”", "[UNK]"}


class MLMPlugDFormatter:
    def __init__(self, config, mode, *args, **params):
        self.ctx_len = config.getint("train", "ctx_len")
        self.query_len = config.getint("train", "query_len")
    
        self.mode = mode
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "pretrained_model"))
        # self.masker = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_prob, return_tensors="pt")


    def get_candidate_span_clusters(self, stokens, max_span_length, include_sub_clusters=False, validate=True):
        token_to_indices = defaultdict(list)
        for sid, sent in enumerate(stokens):
            for i, token in enumerate(sent):
                token_to_indices[token].append((sid, i))

        recurring_spans = []
        for token, indices in token_to_indices.items():
            for i, (sidx1, tidx1) in enumerate(indices):
                for j in range(i + 1, len(indices)):
                    sidx2, tidx2 = indices[j]
                    assert sidx1 < sidx2 or (sidx1 == sidx2 and tidx1 < tidx2)

                    max_recurring_length = 1
                    for length in range(1, max_span_length):
                        if include_sub_clusters:
                            recurring_spans.append(((sidx1, tidx1), (sidx2, tidx2), length))
                        if (tidx1 + length) >= len(stokens[sidx1]) or (tidx2 + length) >= len(stokens[sidx2]) or stokens[sidx1][tidx1 + length] != stokens[sidx2][tidx2 + length]:
                            break
                        if stokens[sidx1][tidx1 + length] in STOPWORDS:
                            break
                        max_recurring_length += 1

                    if max_recurring_length == 1:
                        continue

                    if max_recurring_length == max_span_length or not include_sub_clusters:
                        if stokens[sidx1][tidx1 + max_recurring_length - 1].replace("▁", "").lower() in STOPWORDS and max_recurring_length > 1:
                            max_recurring_length -= 1
                        recurring_spans.append(((sidx1, tidx1), (sidx2, tidx2), max_recurring_length))

        spans_to_clusters = {}
        spans_to_representatives = {}
        for idx1, idx2, length in recurring_spans:
            first_span, second_span = (idx1[0], idx1[1], idx1[1] + length - 1), (idx2[0], idx2[1], idx2[1] + length - 1)
            if first_span in spans_to_representatives:
                if second_span not in spans_to_representatives:
                    rep = spans_to_representatives[first_span]
                    cluster = spans_to_clusters[rep]
                    cluster.append(second_span)
                    spans_to_representatives[second_span] = rep
            elif second_span in spans_to_representatives:
                if first_span not in spans_to_representatives:
                    rep = spans_to_representatives[second_span]
                    cluster = spans_to_clusters[rep]
                    cluster.append(first_span)
                    spans_to_representatives[first_span] = rep
            else:
                cluster = [first_span, second_span]
                spans_to_representatives[first_span] = first_span
                spans_to_representatives[second_span] = first_span
                spans_to_clusters[first_span] = cluster

        if validate:
            recurring_spans = [cluster for cluster in spans_to_clusters.values()
                            if self.validate_ngram(stokens, cluster[0][0], cluster[0][1], cluster[0][2] - cluster[0][1] + 1)]
        else:
            recurring_spans = spans_to_clusters.values()
        return recurring_spans


    def validate_ngram(self, stokens, sentidx, start_index, length):
        tokens = stokens[sentidx][start_index: start_index + length]
        # If the vocab at the beginning of the span is a part-of-word (##), we don't want to consider this span.
        # if vocab_word_piece[token_ids[start_index]]:

        if tokens[0][0] in STOPWORDS:
            return False

        # # If the token *after* this considered span is a part-of-word (##), we don't want to consider this span.
        # if (start_index + length) < len(tokens) and tokens[start_index + length].startswith("##"):
        #     return False

        # if any([(not tokens[idx].isalnum()) and (not tokens[idx].startswith("##")) for idx in range(length)]):
        #     return False

        # We filter out n-grams that are all stopwords (e.g. "in the", "with my", ...)
        if any([t.lower() not in STOPWORDS for t in tokens]):
            return True

    def sort_cluster_by_length(self, clusters):
        ret = clusters
        ret.sort(key=lambda x:x[0][2] - x[0][1], reverse=True)
        return ret[:15]

    def mask_important_spans(self, sents):
        stokens = [self.tokenizer.tokenize(s) for s in sents]
        clusters = self.sort_cluster_by_length(self.get_candidate_span_clusters(stokens, 4))
        sid2span = defaultdict(list)
        for cluster in clusters:
            for span in cluster:
                sid2span[span[0]].append(span)
        
        sids = random.sample(list(sid2span.keys()), min(2, len(sid2span)))
        sids.sort()
        sids = set(sids)
        
        ctx_ids = []
        input_ids, labels = [], []
        q_len = 0
        for sid, sent in enumerate(stokens):
            tids = self.tokenizer.convert_tokens_to_ids(sent)

            if sid in sids and q_len < self.query_len and len(sid2span[sid]) > 0:
                sspans = random.sample(sid2span[sid], min(2, len(sid2span[sid])))

                sspans.sort(key=lambda x:x[1] * 1000 + x[2])
                mask_span = [list(sspans[0])]
                for s in sspans[1:]:
                    if s[1] <= mask_span[-1][2]:
                        mask_span[-1][1] = min(s[1], mask_span[-1][1])
                        mask_span[-1][2] = max(s[2], mask_span[-1][2])
                    else:
                        mask_span.append(list(s))
                tids_label = [-100] * len(tids)
                for s in mask_span:
                    for b in range(s[1], s[2] + 1):
                        tids_label[b] = tids[b]
                        tids[b] = self.tokenizer.mask_token_id
                input_ids.append(tids)
                labels.append(tids_label)
                q_len += len(tids)
            else:
                ctx_ids.append(tids)

        return ctx_ids, input_ids, labels

    def process(self, data):
        ctx_input_ids, ctx_attention_mask = [], []
        query_input_ids, query_mask, query_labels = [], [], []

        for doc in data:
            doc = doc.replace("=", "\n")
            sents = [d + "。" for d in doc.split("。")]
            ctx_ids, input_ids, labels = self.mask_important_spans(sents)
            
            cinp = [self.tokenizer.cls_token_id]
            for c in ctx_ids:
                cinp.extend(c)
            cinp.append(self.tokenizer.sep_token_id)
            
            qinp, ql = [self.tokenizer.cls_token_id], [-100]
            for q, l in zip(input_ids, labels):
                qinp.extend(q)
                ql.extend(l)
            qinp.append(self.tokenizer.sep_token_id)
            ql.append(-100)

            cinp, qinp, ql = cinp[:self.ctx_len], qinp[:self.query_len], ql[:self.query_len]

            ctx_attention_mask.append([1] * len(cinp) + [0] * (self.ctx_len - len(cinp)))
            ctx_input_ids.append(cinp + [self.tokenizer.pad_token_id] * (self.ctx_len - len(cinp)))
            
            query_mask.append([1] * len(qinp) + [0] * (self.query_len - len(qinp)))
            query_labels.append(ql + [-100] * (self.query_len - len(qinp))) 
            query_input_ids.append(qinp + [self.tokenizer.pad_token_id] * (self.query_len - len(qinp)))

        out = []
        for q, l in zip(query_input_ids[0], query_labels[0]):
            if l == -100:
                out.append((self.tokenizer.convert_ids_to_tokens(q), -100))
            else:
                out.append((self.tokenizer.convert_ids_to_tokens(q), self.tokenizer.convert_ids_to_tokens(l)))
        # print(self.tokenizer.decode(query_input_ids[0]))
        # print(out)
        return {
            "ctx_input_ids": torch.LongTensor(ctx_input_ids),
            "ctx_attention_mask": torch.LongTensor(ctx_attention_mask),

            "query_input_ids": torch.LongTensor(query_input_ids),
            "query_mask": torch.LongTensor(query_mask),
            "query_labels": torch.LongTensor(query_labels),
        }
