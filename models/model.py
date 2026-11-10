"""Attention block with L0 regularization.
Selected three-stage model from code/architecture/networks_with_l0.py.
Class bodies and parameter names are unchanged.
"""
import torch
import torch.nn as nn
from .layers import (ConvBnAct3d, InputTransition, DownTrasition,
                     Upsample_block, UpTransition_block, OutputTransition,
                     Channel_Attention, Spatial_Attention)
from .ir import SimpleBlock_L0


class Sharing_with_CASA_L0Block(nn.Module):
    """
    CASA_with_SimpleBlock_L0
    Attention with 2 inputs, x_from_branch_1(x_1), x_from_branch_2(x_2),
    Channel + Spatial Attention.  Channel first
    """
    def __init__(self, in_ch, out_ch, dropout = False):
        super(Sharing_with_CASA_L0Block, self).__init__()
        self.F_tr = ConvBnAct3d(in_ch, out_ch)
        self.CA = Channel_Attention(out_ch, out_ch)
        self.SA = Spatial_Attention(out_ch, out_ch)
        self.l0_block_0 = SimpleBlock_L0(out_ch, out_ch)
        self.l0_block_1 = SimpleBlock_L0(out_ch, out_ch)

    def forward(self, x_1, x_2):
        x_cat = torch.cat((x_1, x_2), dim=1)
        x_fusion = self.F_tr(x_cat)
        x_ca = self.CA(x_fusion)
        x = self.SA(x_ca)
        out_1, out_2 = self.l0_block_0(x), self.l0_block_1(x)
        # out_1, out_2 = x_1+x, x_2+x
        return out_1, out_2


class Attention_MT_Networks_with_L0(nn.Module):
    """
    3 stage, CASA+L0
    L0Block
    3 downsample, 2 decoder branches, Mix transition with attention inside
    Attention sharing here : CA SA, just like SENet.
    SENet: Squeeze-and-Excitation Networks, 
    """
    def __init__(self, in_ch, out_ch_1, out_ch_2, dropout=False):
        super(Attention_MT_Networks_with_L0, self).__init__()
        a, b, c = 32, 64, 128
        # a, b, c = 64, 128, 256
        self.in_tr = InputTransition(in_ch, 16)
        self.down_tr32 = DownTrasition( 16, a, depth=1, dropout=dropout)
        self.down_tr64 = DownTrasition( a, b, depth=2, dropout=dropout)
        self.down_tr128 = DownTrasition( b, c, depth=2, dropout=dropout)

        # decoder branch 1
        self.up_sa128_b1 = Upsample_block( c, c, dropout=dropout)  # b1 <-> branch 1
        self.up_tr128_b1 = UpTransition_block( c, depth=2, dropout=dropout)

        self.up_sa64_b1 = Upsample_block( c, b, dropout=dropout)
        self.up_tr64_b1 = UpTransition_block( b, depth=2, dropout=dropout)
        
        self.up_sa32_b1 = Upsample_block( b, a, dropout=dropout)
        self.up_tr32_b1 = UpTransition_block( a, depth=1, dropout=dropout)
        
        self.out_tr_b1 = OutputTransition( a, out_ch_1)

        # decoder branch 2
        self.up_sa128_b2 = Upsample_block( c, c, dropout=dropout)  # b2 <-> branch 2
        self.up_tr128_b2 = UpTransition_block( c, depth=2, dropout=dropout)
        
        self.up_sa64_b2 = Upsample_block( c, b, dropout=dropout)
        self.up_tr64_b2 = UpTransition_block( b, depth=2, dropout=dropout)
        
        self.up_sa32_b2 = Upsample_block( b, a, dropout=dropout)
        self.up_tr32_b2 = UpTransition_block( a, depth=1, dropout=dropout)
        
        self.out_tr_b2 = OutputTransition( a, out_ch_2)

        # mix transition
        self.mix1 = Sharing_with_CASA_L0Block(in_ch= c*2, out_ch= c, dropout=False)
        self.mix2 = Sharing_with_CASA_L0Block(in_ch= b*2, out_ch= b, dropout=False)
        self.mix3 = Sharing_with_CASA_L0Block(in_ch= a*2, out_ch= a, dropout=False)

    def forward(self, x):        
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)

        upcat128_1 = self.up_sa128_b1(out128, out64) # _1 <-> output from branch 1
        upcat128_2 = self.up_sa128_b2(out128, out64) # _2 <-> output from branch 2
        mix1_1, mix1_2 = self.mix1(upcat128_1, upcat128_2)
        uptr64_1 = self.up_tr128_b1(upcat128_1, mix1_1) 
        uptr64_2 = self.up_tr128_b2(upcat128_2, mix1_2)

        upcat64_1 = self.up_sa64_b1(uptr64_1, out32)
        upcat64_2 = self.up_sa64_b2(uptr64_2, out32)
        mix2_1, mix2_2 = self.mix2(upcat64_1, upcat64_2)
        uptr32_1 = self.up_tr64_b1(upcat64_1, mix2_1)
        uptr32_2 = self.up_tr64_b2(upcat64_2, mix2_2)

        upcat32_1 = self.up_sa32_b1(uptr32_1, out16)
        upcat32_2 = self.up_sa32_b2(uptr32_2, out16)
        mix3_1, mix3_2 = self.mix3(upcat32_1, upcat32_2)
        uptr16_1 = self.up_tr32_b1(upcat32_1, mix3_1)
        uptr16_2 = self.up_tr32_b2(upcat32_2, mix3_2)

        out_1 = self.out_tr_b1(uptr16_1)  # 8, 3, 64, 64, 64
        out_2 = self.out_tr_b2(uptr16_2)

        return out_1, out_2
