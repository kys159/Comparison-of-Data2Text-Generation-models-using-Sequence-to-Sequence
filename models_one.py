import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import numpy as np

'''
1. sequnit에 padding idx 확인 필요 -> 0으로 수정했고, vocab에서 pad, sos, eos 확인 필요
2. sequnit에 embedding sparse로 해야할지도?
3. attention 부터 진행해야됨. linear layer로 수정하고, dimension 확인. 각각 linear layer 객체 지정해야됨.
'''


def init_wt_normal(wt):
    wt.weight.data.normal_(std=1e-4)

def init_wt_unif(wt):
    wt.data.uniform_(-0.02, 0.02)


class OutputUnit(nn.Module):
    def __init__(self, input_size, output_size):
        super(OutputUnit, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.out = nn.Linear(input_size, input_size)
        self.out2 = nn.Linear(input_size, output_size)
        
        init_wt_normal(self.out2)
    
    
    def forward(self, x):
        
        out = self.out(x)
        out = self.out2(out)
        return out

class Pointer_Generator(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(Pointer_Generator, self).__init__()
        self.emb_size = emb_size,
        self.hidden_size = hidden_size
        
        self.W_c = nn.Parameter(torch.randn(hidden_size, 1))
        self.W_d = nn.Parameter(torch.randn(hidden_size, 1))
        self.W_x = nn.Parameter(torch.randn(emb_size, 1))
        self.bias = nn.Parameter(torch.randn(1))
        
        self.init_weights()
        

                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, context, de_state, x):
    
        p_gen = context @ self.W_c + de_state @ self.W_d + x @ self.W_x + self.bias
        p_gen = torch.sigmoid(p_gen)
        
        return p_gen
        
class Attention(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wp = nn.Linear(input_size, hidden_size, bias=True)
        self.Wq = nn.Linear(input_size, hidden_size, bias=True)
        self.Wt = nn.Linear(2*input_size, hidden_size, bias = True)
        
    def forward(self, de_state, en_outputs):
        g = torch.bmm(F.tanh(self.Wp(en_outputs)), F.tanh(self.Wq(de_state)).permute(0,2,1))
        
        weights = F.softmax(g - torch.max(g, dim = 1, keepdim = True).values, dim = 1)
        context = torch.bmm(weights.permute(0,2,1), en_outputs)
        out = F.tanh(self.Wt(torch.cat((context, de_state), dim = 2)))
        
        return out, weights, context
    
class DualAttention(nn.Module):
    
    def __init__(self, hidden_size, input_size, field_size):
        super(DualAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.Wp = nn.Linear(input_size, hidden_size, bias=True)
        self.Wq = nn.Linear(input_size, hidden_size, bias=True)
        self.Wt = nn.Linear(2*input_size, hidden_size, bias = True)
        
        self.Wx = nn.Linear(field_size, hidden_size, bias = True)
        self.Wy = nn.Linear(input_size, hidden_size, bias=True)
        self.init_linear_wt()

    
    def init_linear_wt(self):
        for weight in self.parameters():
            weight.data.normal_(std=1e-4)
        
    def forward(self, de_state, en_outputs, field_input):
        
        g = torch.bmm(F.tanh(self.Wp(en_outputs)), F.tanh(self.Wq(de_state)).permute(0,2,1))
        g_b = torch.bmm(F.tanh(self.Wx(field_input)), F.tanh(self.Wy(de_state)).permute(0,2,1))
        
        weights_a = F.softmax(g - torch.max(g, dim = 1, keepdim = True).values, dim = 1)
        weights_b = F.softmax(g_b - torch.max(g_b, dim = 1, keepdim = True).values, dim = 1)
        
        weights = torch.div(torch.mul(weights_a, weights_b), (1e-6 + torch.sum(torch.mul(weights_a, weights_b), dim = 1, keepdim=True)))

        context = torch.bmm(weights.permute(0,2,1), en_outputs)
        out = F.tanh(self.Wt(torch.cat((context, de_state), dim = 2)))
        
        return out, weights, context#, coverage#, cov_loss

class LstmUnit(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(LstmUnit, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.randn(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size * 4))
        self.init_weights()
        self.bias = nn.Parameter(torch.zeros(hidden_size * 4))
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                F.sigmoid(gates[:, :HS]), # input
                F.sigmoid(gates[:, HS:HS*2] + 1.0), # forget
                F.tanh(gates[:, HS*2:HS*3]),
                F.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * F.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class fgateLstmUnit(nn.Module):
    def __init__(self, hidden_size, input_size, field_size):
        super(fgateLstmUnit, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.field_size = field_size
        self.W_f = nn.Parameter(torch.randn(input_size, hidden_size * 4))
        self.Wz = nn.Parameter(torch.randn(field_size, hidden_size * 2))
        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size * 4))
        self.init_weights()
        
        self.bias_f = nn.Parameter(torch.zeros(hidden_size * 4))
        self.biasz = nn.Parameter(torch.zeros(hidden_size * 2))
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, word_input, field_input, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = word_input.size()
        
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(word_input.device), 
                        torch.zeros(bs, self.hidden_size).to(word_input.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = word_input[:, t, :]
            z_t = field_input[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W_f + h_t @ self.U_f + self.bias_f
            field_gates = z_t @ self.Wz + self.biasz
            
            i_t, f_t, g_t, o_t = (
                F.sigmoid(gates[:, :HS]), # input
                F.sigmoid(gates[:, HS:HS*2] + 1.0), # forget
                F.tanh(gates[:, HS*2:HS*3]),
                F.sigmoid(gates[:, HS*3:]), # output
            )
            lt, zt = (
                F.sigmoid(field_gates[:,:HS]),
                F.tanh(field_gates[:,HS:])
            )
            
            c_t = f_t * c_t + i_t * g_t + (lt * zt)
            h_t = o_t * F.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)



class SeqUnit(nn.Module):
    def __init__(self, batch_size, hidden_size, emb_size, field_size, pos_size, source_vocab, field_vocab,
                 position_vocab, target_vocab, field_concat, position_concat, fgate_enc, dual_att,
                 encoder_add_pos, decoder_add_pos, learning_rate, pointer, start_token=2, stop_token=2, max_length=150):
        super(SeqUnit, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.field_size = field_size
        self.pos_size = pos_size
        
        self.uni_size = emb_size if not field_concat else emb_size+field_size
        self.uni_size = self.uni_size if not position_concat else self.uni_size+2*pos_size
        self.field_encoder_size = field_size if not encoder_add_pos else field_size+2*pos_size
        self.field_attention_size = field_size if not decoder_add_pos else field_size+2*pos_size
        
        self.source_vocab = source_vocab
        self.field_vocab = field_vocab
        self.position_vocab = position_vocab
        self.target_vocab = target_vocab
        self.field_concat = field_concat # concat field information to input
        self.position_concat = position_concat # concat position informaton to input
        self.fgate_enc = fgate_enc # add field gate in encoder lstm
        self.dual_att = dual_att # dual attention layer or normal attention
        self.encoder_add_pos = encoder_add_pos # position information in field-gated encoder
        self.decoder_add_pos = decoder_add_pos # position information in dual attention encoder
        self.pointer = pointer
        self.learning_rate = learning_rate
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.device = torch.device('cuda')
        
        #self.log_softmax = torch.nn.LogSoftmax(dim=2)

        if self.fgate_enc:
            print("Field-gate is used")
            self.enc_lstm = fgateLstmUnit(self.hidden_size, self.uni_size, self.field_encoder_size)
        else :
            print("normal Lstm is used")
            self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size)
        self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size)
        self.dec_out = OutputUnit(self.hidden_size, self.target_vocab)
        
        self.enc_embedding = nn.Embedding(num_embeddings=self.source_vocab,embedding_dim=self.emb_size,padding_idx=0)
        init_wt_normal(self.enc_embedding)
        #self.dec_embedding = nn.Embedding(num_embeddings=self.source_vocab,embedding_dim=self.emb_size,padding_idx=0)
        
        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            print("Field-concat")
            self.field_embedding = nn.Embedding(num_embeddings=self.field_vocab,embedding_dim=self.field_size,padding_idx=0)
            init_wt_normal(self.field_embedding)
        
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            print("Position concat")
            self.pos_embedding = nn.Embedding(num_embeddings=self.position_vocab,embedding_dim=self.pos_size,padding_idx=0)
            self.rpos_embedding = nn.Embedding(num_embeddings=self.position_vocab,embedding_dim=self.pos_size,padding_idx=0)
            init_wt_normal(self.pos_embedding)
            init_wt_normal(self.rpos_embedding)
            
            
        if self.dual_att:
            print('Dual Attention Mechanism Used')
            self.att_layer = DualAttention(self.hidden_size, self.hidden_size, self.field_attention_size)
        
        else :
            print('Normal Attention used')
            self.att_layer = Attention(self.hidden_size, self.hidden_size)
        
        if self.pointer:
            print("Copy Mechanism is used")
            self.pg = Pointer_Generator(self.emb_size, self.hidden_size)
        
    def forward(self, inputs, fields, pos, rpos, dec_inp, enc_batch_indicies):
        
        inputs_embedding = self.enc_embedding(inputs)
        
        #enc_batch_indicies = enc_batch_indicies.unsqueeze(2).repeat(1,1,dec_inp.size(1))
        
        if self.field_concat:
            inputs_embedding = torch.cat([self.enc_embedding(inputs),
                                            self.field_embedding(fields)], 2)
            field_pos_embedding = self.field_embedding(fields)
        if self.position_concat:
            inputs_embedding = torch.cat([inputs_embedding,
                                               self.pos_embedding(pos),
                                               self.rpos_embedding(rpos)], 2)
            field_pos_embedding = torch.cat([field_pos_embedding,
                                                  self.pos_embedding(pos),
                                                  self.rpos_embedding(rpos)], 2)
            
        elif self.encoder_add_pos or self.decoder_add_pos:
            field_pos_embedding = torch.cat([field_pos_embedding,
                                                  self.pos_embedding(pos),
                                                  self.rpos_embedding(rpos)], 2)
        
        if self.fgate_enc:
            en_outputs, en_state = self.enc_lstm(inputs_embedding, field_pos_embedding)
        else :
            en_outputs, en_state = self.enc_lstm(inputs_embedding)
        
        
        if self.pointer:
            loss_output = torch.zeros([self.batch_size,dec_inp.size(1),self.target_vocab + inputs_embedding.size(1)], device = self.device)
        else:
            loss_output = torch.zeros([self.batch_size,dec_inp.size(1),self.target_vocab], device = self.device)
            
        #att = torch.empty([self.batch_size, len(inputs[0]), 0], device = self.device)
        # cov_att = torch.zeros(self.batch_size, inputs.size(1), 1, device = self.device)
        # covloss = 0.0
        tot_pcopy = torch.zeros(self.batch_size, dec_inp.size(1))
        for i in range(dec_inp.size(1)):
            de_outputs, de_state = self.dec_lstm(self.enc_embedding(dec_inp[:,i]).unsqueeze(1), en_state)
            
            if self.dual_att:
                de_outputs, atts, context = self.att_layer(de_outputs, en_outputs, field_pos_embedding)
            else :
                de_outputs, atts, context = self.att_layer(de_outputs, en_outputs)
            
            if self.pointer:
                pcopy = self.pg(context, de_outputs, self.enc_embedding(dec_inp[:,i]).unsqueeze(1))
                output = self.dec_out(de_outputs)
                
                extend_vocab = torch.zeros([self.batch_size,output.size(1),self.target_vocab + inputs_embedding.size(1)], device = self.device)
                extend_vocab[:,:,:self.target_vocab] = torch.mul(output, (1-pcopy))
                
                enc_batch_att = torch.mul(atts, pcopy.permute(0,2,1))
                for k in range(output.size(1)):
                    extend_vocab[:,k,:].scatter_add_(1, enc_batch_indicies, enc_batch_att[:,:,k])
                
                loss_output[:,i] = extend_vocab.squeeze(1)
                tot_pcopy[:,i] = pcopy.squeeze()
                # loss_output[:,i,:self.target_vocab] = torch.mul(output, pgen)
                # enc_batch_att = torch.mul(atts, (1-pgen).permute(0,2,1))
                # loss_output[:,i,:] = loss_output[:,i,:].scatter_add_(1, enc_batch_indicies, enc_batch_att[:,:,i])
                
                # this_covloss = torch.sum(torch.min(atts, cov_att).squeeze(2), dim =1)
                # covloss += torch.sum(this_covloss)
                
                #cov_att = coverage
            
                # cov_att += atts
                
            else:
                output = self.dec_out(de_outputs)
                #loss_output = torch.cat([loss_output, output], dim = 1)
                loss_output[:,i] = output.squeeze(1)
            
            en_state = de_state

        #return loss_output, covloss
        return loss_output, tot_pcopy
        
    def generate(self, inputs, fields, pos, rpos, enc_batch_indicies):

        if self.field_concat:
            inputs_embedding = torch.cat([self.enc_embedding(inputs),
                                        self.field_embedding(fields)], 2)
            field_pos_embedding = self.field_embedding(fields)
        if self.position_concat:
            inputs_embedding = torch.cat([inputs_embedding,
                                               self.pos_embedding(pos),
                                               self.rpos_embedding(rpos)], 2)
            field_pos_embedding = torch.cat([field_pos_embedding,
                                                  self.pos_embedding(pos),
                                                  self.rpos_embedding(rpos)], 2)
            
        elif self.encoder_add_pos or self.decoder_add_pos:
            field_pos_embedding = torch.cat([field_pos_embedding,
                                                  self.pos_embedding(pos),
                                                  self.rpos_embedding(rpos)], 2)
        
        if self.fgate_enc:
            en_outputs, en_state = self.enc_lstm(inputs_embedding, field_pos_embedding)
        else :
            en_outputs, en_state = self.enc_lstm(inputs_embedding)
        
        outputs = torch.ones([self.batch_size,1], dtype = int, device = self.device)
        g_tokens = torch.empty([self.batch_size,0], dtype = int, device = self.device)
        
        cov_att = torch.zeros(self.batch_size, inputs.size(1), 1, device = self.device)
        att = torch.empty([self.batch_size, len(inputs[0]), 0], device = self.device)
        for i in range(self.max_length):
            de_outputs, de_state = self.dec_lstm(self.enc_embedding(outputs), en_state)
            
            if self.dual_att:
                de_outputs, atts, context, = self.att_layer(de_outputs, en_outputs, field_pos_embedding)
            else :
                de_outputs, atts, context = self.att_layer(de_outputs, en_outputs)
            
            att = torch.cat([att, atts], dim = 2)
            if self.pointer:
                pcopy = self.pg(context, de_outputs, self.enc_embedding(outputs))

                output = self.dec_out(de_outputs)

                extend_vocab = torch.zeros([self.batch_size,output.size(1),self.target_vocab + inputs_embedding.size(1)], device = self.device)
                extend_vocab[:,:,:self.target_vocab] = torch.mul(output, (1-pcopy))
                
                enc_batch_att = torch.mul(atts, pcopy.permute(0,2,1))
                for i in range(output.size(1)):
                    extend_vocab[:,i,:].scatter_add_(1, enc_batch_indicies, enc_batch_att[:,:,i])
                output = torch.argmax(extend_vocab.squeeze(1), dim = 1)
                outputs = torch.where(output >= self.target_vocab, 3, output).view(self.batch_size,-1)
                g_tokens = torch.cat([g_tokens,output.view(self.batch_size,-1)], dim = 1)
                
                #cov_att = coverage
            else:
                output = self.dec_out(de_outputs)
                #loss_output = torch.cat([loss_output, output], dim = 1)
                outputs = torch.argmax(output.squeeze(1), dim = 1).view(self.batch_size, -1)
                g_tokens = torch.cat([g_tokens,outputs.view(self.batch_size,-1)], dim = 1)
            
            en_state = de_state
        
#        return g_tokens, loss_output, att
        return g_tokens, att