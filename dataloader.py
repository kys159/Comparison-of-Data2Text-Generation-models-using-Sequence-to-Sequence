#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22-01-04 ~ . 
# @Author  : KYS

import torch
from torch.utils.data import Dataset
import time
import numpy as np

from itertools import compress
import copy

class Dataset(Dataset):
    def __init__(self, data_dir, mode, limits):
        self.data_path = [data_dir + '/{}/{}'.format(mode, mode) + '.summary.id', 
                          data_dir + '/{}/{}'.format(mode, mode) + '.box.val.id',
                          data_dir + '/{}/{}'.format(mode, mode) + '.box.val',
                          data_dir + '/{}/{}'.format(mode, mode) + '.box.lab.id', 
                          data_dir + '/{}/{}'.format(mode, mode) + '.box.pos',
                          data_dir + '/{}/{}'.format(mode, mode) + '.box.rpos',
                          'original_data' + '/{}.summary'.format(mode)]
        
        self.limits = limits
        start_time = time.time()
        
        print('Reading datasets ...')
        self.data_set = self.load_data(self.data_path)
        
        self.summaries, self.texts, self.input_texts, self.fields, self.poses, self.rposes, self.org_sums = self.data_set
        self.length = len(self.summaries)
        
        print ('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))
        
    def load_data(self, path):
        summary_path, text_path, input_text, field_path, pos_path, rpos_path, org_sum_path = path
        summaries = open(summary_path, 'r').read().strip().split('\n')
        texts = open(text_path, 'r').read().strip().split('\n')
        input_texts = open(input_text, 'r', encoding = 'utf-8').read().strip().split('\n')
        fields = open(field_path, 'r').read().strip().split('\n')
        poses = open(pos_path, 'r').read().strip().split('\n')
        rposes = open(rpos_path, 'r').read().strip().split('\n')
        org_sums = open(org_sum_path, 'r', encoding = 'utf-8').read().strip().split('\n')
        
        if self.limits > 0:
            summaries = summaries[:self.limits]
            texts = texts[:self.limits]
            input_texts = input_texts[:self.limits]
            fields = fields[:self.limits]
            poses = poses[:self.limits]
            rposes = rposes[:self.limits]
            org_sum = rposes[:self.limits]
        summaries = [list(map(int, summary.strip().split(' '))) for summary in summaries]
        texts = [list(map(int, text.strip().split(' '))) for text in texts]
        input_texts = [list(map(str, text.strip().split(' '))) for text in input_texts]
        fields = [list(map(int, field.strip().split(' '))) for field in fields]
        poses = [list(map(int, pos.strip().split(' '))) for pos in poses]
        rposes = [list(map(int, rpos.strip().split(' '))) for rpos in rposes]
        org_sums = [list(map(str, org_sum.strip().split(' '))) for org_sum in org_sums]
        
        return summaries, texts, input_texts, fields, poses, rposes, org_sums
    
    def __getitem__(self, index):
        return self.summaries[index], self.texts[index], self.input_texts[index], self.fields[index], self.poses[index], self.rposes[index], self.org_sums[index]
    
    def __len__(self):
        return self.length

def gen_transformUNK(origin, index, enc_org, point_dec, enc_batch_indices):
    for t, i, e, p, b in zip(origin, index, enc_org, point_dec, enc_batch_indices):
        yield t, i, e, p, b

def collate_batch(batch):
    batch_data = {'enc_in':[], 'enc_fd':[], 'enc_pos':[], 'enc_rpos':[], 'enc_len':[],
                        'dec_in':[], 'dec_len':[], 'dec_out':[], 'input_text': [], 'org_sums': []} #'enc_batch_extend_vocab': []}

    man_text_len = 100
    max_summary_len = max([len(sample_sum[0]) for sample_sum in batch])
    max_text_len = max([len(sample_txt[1]) for sample_txt in batch])
    
    
    for summary, text, input_texts, field, pos, rpos, org_sum in batch:
        
        summary_len = len(summary)
        text_len = len(text)
        pos_len = len(pos)
        rpos_len = len(rpos)
        assert text_len == len(field)
        assert pos_len == len(field)
        assert rpos_len == pos_len
        gold = summary + [2] + [0] * (max_summary_len - summary_len)
        summary = summary + [0] * (max_summary_len - summary_len)
        text = list(text) + [0] * (max_text_len - text_len)
        input_texts = list(input_texts) + ['PAD'] * (max_text_len - text_len) 
        field = list(field) + [0] * (max_text_len - text_len)
        pos = list(pos) + [0] * (max_text_len - text_len)
        rpos = list(rpos) + [0] * (max_text_len - text_len)
        
        if max_text_len > man_text_len:
            text = text[:man_text_len]
            input_texts = input_texts[:man_text_len]
            field = field[:man_text_len]
            pos = pos[:man_text_len]
            rpos = rpos[:man_text_len]
            text_len = min(text_len, man_text_len)
        
        batch_data['enc_in'].append(text)
        batch_data['enc_len'].append(text_len)
        batch_data['enc_fd'].append(field)
        batch_data['enc_pos'].append(pos)
        batch_data['enc_rpos'].append(rpos)
        batch_data['dec_in'].append(summary)
        batch_data['dec_len'].append(summary_len)
        batch_data['dec_out'].append(gold)
        batch_data['input_text'].append(input_texts)
        batch_data['org_sums'].append(org_sum)
        #batch_data['enc_batch_extend_vocab'].append(np.zeros(max_text_len))
    
    batch_data['dec_point_out'] = copy.deepcopy(batch_data['dec_out'])
    batch_data['enc_point_in'] = copy.deepcopy(batch_data['enc_in'])
    unk_gen = gen_transformUNK(batch_data['org_sums'], batch_data['dec_out'], batch_data['input_text'], batch_data['dec_point_out'], batch_data['enc_point_in'])

    for (org, out, in_text, point_out, enc_batch) in unk_gen:
        for idx, unk in enumerate(list(compress(org, np.array(out[:out.index(2)]) == 3))):
            if unk in in_text:
                batch_input_idx = min([input_idx for input_idx, input in enumerate(in_text) if input == unk])
                # point_out[np.where(np.array(out[:out.index(2)]) == 3)[0][idx]] = 20003 + batch_input_idx
                # enc_batch[batch_input_idx] = 20003 + batch_input_idx
                point_out[np.where(np.array(out[:out.index(2)]) == 3)[0][idx]] = 2150 + batch_input_idx
                enc_batch[batch_input_idx] = 2150 + batch_input_idx
    
    return batch_data