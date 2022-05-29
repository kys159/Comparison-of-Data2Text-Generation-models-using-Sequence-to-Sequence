#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22-01-04 ~ . 
# @Author  : KYS

import torch
from dataloader import Dataset, collate_batch
from torch.utils import data
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from models_one import *

from utils import *
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from pathlib import Path
from tqdm import tqdm
import neptune
import warnings
warnings.filterwarnings(action='ignore')

class Trainer():
    def __init__(self, params):
        self.params = params
        self.experiment_name = params['experiment_name'] # save folder name
        self.hidden_size = params['hidden_size'] # Size of each layer
        self.emb_size = params['emb_size'] # Size of word embedding.
        self.field_size = params['field_size'] # Size of field embedding.
        self.pos_size = params['pos_size'] # Size of position embedding
        self.batch_size = params['batch_size'] # Batch size of train set
        self.epochs = params['epochs'] # Number of training epoch.
        self.source_vocab = params['source_vocab'] # vocabulary size
        self.field_vocab = params['field_vocab'] # field vocabulary size
        self.position_vocab = params['position_vocab'] # vocabulary size
        self.target_vocab = params['target_vocab'] # vocabulary size

        self.report = params['report'] # report valid results after some steps
        self.learning_rate = params['learning_rate'] # learning rate

        self.mode = params['mode'] # train or test
        self.load = params['load'] # chekpoint load directory
        self.dir = params['dir'] # data set directory
        self.limits = params['limits'] # max dataset size

        self.dual_attention = eval(params['dual_attention']) # dual attention layer or normal attention, have to change to bool 
        self.fgate_encoder = eval(params['fgate_encoder']) # add field gate in encoder lstm

        self.field = eval(params['field']) # concat field information to word embedding where input
        self.position = eval(params['position']) # concat position information to word embedding where input
        self.encoder_pos = eval(params['encoder_pos']) # position information in field-gated encoder
        self.decoder_pos = eval(params['decoder_pos']) # position information in dual attention decoder
        self.pointer = eval(params['pointer'])
        self.device = torch.device('cuda')
        self.max_length = 150
        self.curr_min_val = 0.1
        self.iteration_idx = 0
        self.val_idx = 0
        self.copy_penalty = 1.0
        
    def init_models(self):
        '''
        model init
        '''
        
        self.model = SeqUnit(batch_size = self.batch_size, hidden_size = self.hidden_size,
                            emb_size = self.emb_size, field_size = self.field_size,
                            pos_size = self.pos_size, source_vocab = self.source_vocab,
                            field_vocab = self.field_vocab, position_vocab = self.position_vocab,
                            target_vocab = self.target_vocab, field_concat = self.field,
                            position_concat = self.position, fgate_enc = self.fgate_encoder,
                            dual_att = self.dual_attention, encoder_add_pos = self.encoder_pos,
                            decoder_add_pos = self.decoder_pos, learning_rate = self.learning_rate,
                            pointer = self.pointer, start_token=1, stop_token=2, max_length=self.max_length)

    def load_model_checkpoints(self):
        '''
        model checkpoints load
        '''
        try :
            self.model.load_state_dict(torch.load(str(Path(self.experiment_name, 'model_epoch_best.pt'))))
            print('Model checkpoints from best epoch loaded...')
        
        except :
            print('No model loaded, training from scratch...')
        
    def train(self):
        """ Train models
            
        """
        tr_loader_params = {
                'batch_size': self.batch_size,
                'shuffle': True,
                'num_workers': 8,
                'drop_last': True
            }

        val_loader_params = {
                'batch_size': self.batch_size,
                'shuffle': False,
                'num_workers': 8,
                'drop_last': True
            }
        
        # self.gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
        # self.gold_path_valid = 'processed_data/valid/valid_split_for_rouge/gold_summary_'
        self.gold_path_test = 'gold_summary_'
        self.gold_path_valid = 'gold_summary_'
        
        
        train_dataset = Dataset(data_dir = self.dir, mode = self.mode, limits = self.limits)
        valid_dataset = Dataset(data_dir = self.dir, mode = 'valid', limits = self.limits)
        
        self.train_loader = data.DataLoader(train_dataset, **tr_loader_params, collate_fn = collate_batch)
        self.valid_loader = data.DataLoader(valid_dataset, **val_loader_params, collate_fn = collate_batch)
        self.length_train_dataset = len(self.train_loader)
        self.length_valid_dataset = len(self.valid_loader)
        
        # folder for model checkpoints
        model_checkpoints_folder = Path(self.experiment_name)
        if not model_checkpoints_folder.exists():
            model_checkpoints_folder.mkdir()
        
        # model load and send to device
        self.init_models()
        self.load_model_checkpoints()
        
        self.model.to(self.device)
        
        seed_init()
        
        # optimizer

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        for epoch in range(self.epochs + 1):
            self.train_one_epoch(epoch)

    
    def train_one_epoch(self, epoch):
        """ Train one epoch

        """
        
        self.model.train()
        
        train_loss = 0
        self.it_idx = 0
        for _, batch_data in enumerate(self.train_loader):
            self.iteration_idx += 1
            self.it_idx +=1
            
            self.opt.zero_grad()
            progress_bar(self.iteration_idx%self.report, self.report)
            '''
            batch_idx : index
            batch_data : dictionary format with
                        {'enc_in':[], 'enc_fd':[], 'enc_pos':[], 'enc_rpos':[], 'enc_len':[],
                        'dec_in':[], 'dec_len':[], 'dec_out':[]}
            '''
            [i.insert(0,1) for i in batch_data['dec_in']] # <sos> token 추가

            output, pcopy = self.model(torch.tensor(batch_data['enc_in'], device = self.device), 
                                torch.tensor(batch_data['enc_fd'], device = self.device),
                                torch.tensor(batch_data['enc_pos'], device = self.device),
                                torch.tensor(batch_data['enc_rpos'], device = self.device),
                                torch.tensor(batch_data['dec_in'], device = self.device),
                                torch.tensor(batch_data['enc_point_in'], device = self.device))
            
            copy_loss = torch.sum((1-pcopy)[torch.tensor(batch_data['dec_point_out']) >= 20003]) / self.batch_size

            ce_loss = self.criterion(output.contiguous().view(-1, output.size(-1)), torch.tensor(batch_data['dec_point_out'], device = self.device).contiguous().view(-1))

            loss = ce_loss +  self.copy_penalty * copy_loss
       
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            
            train_loss += loss.item()
            #print(loss.item())
            neptune.log_metric('train loss', loss.item())
            
            
            if (self.iteration_idx % self.report == 0):
                self.val_idx += 1
                #val_loss, F_measure1, F_measure2, F_measure3, F_measure4, bleu = self.val(self.valid_loader, self.val_idx)
                _, F_measure1, F_measure2, F_measure3, F_measure4, bleu = self.val(self.valid_loader, self.val_idx)
                    
                # 평가 부분 추가 (rouge, bleu)
                
                #neptune.log_metric('valid loss', val_loss)
                neptune.log_metric('train ROUGE-1', F_measure1)
                neptune.log_metric('train ROUGE-2', F_measure2)
                neptune.log_metric('train ROUGE-3', F_measure3)
                neptune.log_metric('train ROUGE-4', F_measure4)
                
                neptune.log_metric('train BLEU-4', bleu)
        
                train_loss = train_loss / self.it_idx
                print('====> Epoch: {}, Iter: {}  Pairwise loss: {:.8f}'.format(epoch, self.val_idx, train_loss))
                print('\n')
                self.iteration_idx = 0
        
        
    def val(self, dataloader,val_idx, mode = 'valid'):
        """ Validation

        """
        self.model.eval()
        val_loss = 0
        
        if mode == 'valid':
            # texts_path = "original_data/valid.summary"
            #texts_path = "processed_data/valid/valid.box.val"
            texts_path = 'valid/valid.box.val'
            gold_path = self.gold_path_valid
            
        else:
            # texts_path = "original_data/test.summary"
            #texts_path = "processed_data/test/test.box.val"
            texts_path = 'test/test.box.val'
            gold_path = self.gold_path_test

        predict_val_folder = Path(self.experiment_name, str(val_idx))
        if not predict_val_folder.exists():
            predict_val_folder.mkdir()
        
        predict_val_folder = Path(self.experiment_name, str(val_idx), 'pred_summary')
        if not predict_val_folder.exists():
            predict_val_folder.mkdir()
            
        predict_copy_folder = Path(self.experiment_name, str(val_idx), 'res')
        if not predict_copy_folder.exists():
            predict_copy_folder.mkdir()
        
        # for copy words from the infoboxes
        texts = open(texts_path, 'rt', encoding = "UTF8").read().strip().split('\n')
        texts = [list(t.strip().split()) for t in texts]
        v = Vocab()
        
        # with copy
        pred_list, pred_list_copy, gold_list = [], [], []
        pred_unk, pred_mask = [], []
        
        with torch.no_grad():
            k = 0
            for _, batch_data in enumerate(dataloader):
                

                g_tokens, att = self.model.generate(torch.tensor(batch_data['enc_in'], device = self.device), 
                            torch.tensor(batch_data['enc_fd'], device = self.device),
                            torch.tensor(batch_data['enc_pos'], device = self.device),
                            torch.tensor(batch_data['enc_rpos'], device = self.device),
                            torch.tensor(batch_data['enc_point_in'], device = self.device)
                            )
                
                
                idx = 0
                predictions = g_tokens.cpu().numpy()
                att = att.detach().cpu().numpy()
                
                for summary in predictions:
                    
                    with open(self.experiment_name + '/'+ str(val_idx) + '/pred_summary/' + mode + str(k), 'w', -1, "utf-8") as sw:
                        summary = list(summary) 
                        if 2 in summary:
                            summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                        real_sum, unk_sum, mask_sum = [], [], []
                        for tk, tid in enumerate(summary):
                            
                            if tid >= 2150:
                                if batch_data['input_text'][idx][tid - self.source_vocab] == 'PAD':
                                    continue
                                real_sum.append(batch_data['input_text'][idx][tid-self.source_vocab])
                                mask_sum.append("*" + batch_data['input_text'][idx][tid-self.source_vocab] + "*")
                                unk_sum.append('POINTER_TOKEN')
                                continue
                                
                            if tid == 3:
                                sub = texts[k][np.argmax(att[idx,:len(texts[k]),tk])]
                                real_sum.append(sub)
                                mask_sum.append("**" + str(sub) + "**")
                            else:
                                real_sum.append(v.id2word(tid))
                                mask_sum.append(v.id2word(tid))
                            unk_sum.append(v.id2word(tid))
                        sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                        pred_list.append([str(x) for x in real_sum])
                        pred_unk.append([str(x) for x in unk_sum])
                        pred_mask.append([str(x) for x in mask_sum])
                        k += 1
                        idx += 1
            
            write_word(pred_mask, self.experiment_name+ '/'+str(val_idx)+ '/res/', mode + "_summary_copy.txt")
            write_word(pred_unk, self.experiment_name+ '/'+str(val_idx)+ '/res/', mode + "_summary_unk.txt")
                
            for tk in range(k):
                with open(gold_path + str(tk), 'r', -1, "utf-8") as g:
                    gold_list.append([g.read().strip().split()])

            gold_set = [[gold_path + str(i)] for i in range(k)]
            pred_set = [self.experiment_name + '/'+ str(val_idx) + '/pred_summary/' + mode + str(i) for i in range(k)]

            F_measure1_tmp, F_measure2_tmp, F_measure3_tmp, F_measure4_tmp = [],[],[], []
            scorer1 = rouge_scorer.RougeScorer(['rouge1'])
            scorer2 = rouge_scorer.RougeScorer(['rouge2'])
            scorer3 = rouge_scorer.RougeScorer(['rouge3'])
            scorer4 = rouge_scorer.RougeScorer(['rouge4'])

            for i in range(len(pred_set)) :
                pred = open(pred_set[i], "rt", encoding="UTF8")
                pred_lines = pred.readlines()
                gold = open(gold_set[i][0], "rt", encoding="UTF8")
                gold_lines = gold.readlines()
                
                scores1 = scorer1.score(pred_lines[0], gold_lines[0])
                scores2 = scorer2.score(pred_lines[0], gold_lines[0])
                scores3 = scorer3.score(pred_lines[0], gold_lines[0])
                scores4 = scorer4.score(pred_lines[0], gold_lines[0])
                result1 = list(scores1.values())
                result2 = list(scores2.values())
                result3 = list(scores3.values())
                result4 = list(scores4.values())

                F_measure1_tmp.append(result1[0][2])
                F_measure2_tmp.append(result2[0][2])
                F_measure3_tmp.append(result3[0][2])
                F_measure4_tmp.append(result4[0][2])

            F_measure1 = np.mean(F_measure1_tmp)
            F_measure2 = np.mean(F_measure2_tmp)
            F_measure3 = np.mean(F_measure3_tmp)
            F_measure4 = np.mean(F_measure4_tmp)

            bleu = corpus_bleu(gold_list, pred_list)
            copy_result = "with copy F_measure of ROUGE1: %s ROUGE2: %s ROUGE3: %s ROUGE4: %s BLEU: %s\n" % \
            (str(F_measure1), str(F_measure2), str(F_measure3), str(F_measure4), str(bleu))
            
            if mode=='valid':
                #val_loss = val_loss / self.length_valid_dataset * self.batch_size    
                #print('====> Valid loss: {:.4f}'.format(val_loss))
                print('\n')
                print('====> Valid Evaluation Metric result :', copy_result)            
                
                if bleu > self.curr_min_val:
                    self.curr_min_val = bleu
                    torch.save(self.model.state_dict(), str(Path(self.experiment_name, f'model_epoch_best.pt')))
            
            else:
                # test_loss = val_loss / self.length_test_dataset * self.batch_size    
                # print('====> test loss: {:.4f}'.format(test_loss))
                print('\n')
                print('====> test Evaluation Metric result :', copy_result)
                
        #return val_loss, F_measure1, F_measure2, F_measure3, F_measure4, bleu
        return _, F_measure1, F_measure2, F_measure3, F_measure4, bleu

    def test(self):
        """ Test
            
        """
        loader_params = {
                'batch_size': self.batch_size,
                'shuffle': False,
                'num_workers': 8,
                'drop_last': True
            }
        self.gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
        test_dataset = Dataset(data_dir = self.dir, mode = self.mode, limits = self.limits)
        self.test_loader = data.DataLoader(test_dataset, **loader_params, collate_fn = collate_batch)
        self.length_test_dataset = len(self.test_loader)
        
        # model load and send to device
        self.init_models()
        self.load_model_checkpoints()
        self.model.to(self.device)
        
        self.val(dataloader = self.test_loader, val_idx = 'test', mode = self.mode)
        
            