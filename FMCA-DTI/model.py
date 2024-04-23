# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class FMCADTI(nn.Module):
    def __init__(self, hp, args):
        super(FMCADTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_kernel = hp.drug_kernel
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = args['input_d_dim']
        self.protein_vocab_size = args['input_p_dim']
        self.drug_input = args['d_channel_size']
        self.protein_input = args['p_channel_size']
        self.attention_dim = hp.conv * 4
        self.drug_attention_head = 3
        self.protein_attention_head = 3
        self.mix_attention_head = 4

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.drug_input[0], out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.BatchNorm1d(self.conv),
            nn.ELU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.BatchNorm1d(self.conv * 2),
            nn.ELU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.BatchNorm1d(self.conv * 4),
            nn.ELU(),
        )
        self.Drug_max_pool = nn.AdaptiveMaxPool1d((1))

        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.protein_input[0], out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.BatchNorm1d(self.conv),
            nn.ELU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.BatchNorm1d(self.conv *2),
            nn.ELU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.BatchNorm1d(self.conv *4),
            nn.ELU(),
        )

        self.Protein_max_pool = nn.AdaptiveMaxPool1d((1))

        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,512)
        self.out = nn.Linear(512, 1)  #1

    def forward(self, drug, protein):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # [B, D_C, F_C] -> [F_C, B, D_C]
        # [B, D_C, T_C] -> [T_C, B, D_C]
        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)

        # cross Attention
        # [F_C, B, D_C] -> [F_C, B, D_C]
        # [T_C, B, D_C] -> [T_C, B, D_C]
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)

        # [F_C, B, D_C] -> [B, D_C, F_C]
        # [T_C, B, D_C] -> [B, D_C, T_C]
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5


        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)


        pair = torch.cat([drugConv, proteinConv], dim=1)
        fully1 = self.relu(self.fc1(pair))
        fully1 = self.dropout1(fully1)
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.dropout2(fully2)
        fully3 = self.fc3(fully2)
        predict = self.out(fully3)

        return predict
