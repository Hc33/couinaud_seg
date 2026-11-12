"""Three-stage attention-based co-learning network for vessels and Couinaud segments."""
import torch
import torch.nn as nn
from .layers import (ConvBnAct3d, InputTransition, DownTransition,
                     UpsampleBlock, DecoderFusionBlock, OutputTransition,
                     ChannelAttention, SpatialAttention)
from .ir import InformationRoutingBlock


class AttentionSharingBlock(nn.Module):
    """ACB feature aggregation followed by separate vessel and Couinaud IR blocks."""
    def __init__(self, in_ch, out_ch, dropout=False):
        super(AttentionSharingBlock, self).__init__()
        self.F_tr = ConvBnAct3d(in_ch, out_ch)
        self.CA = ChannelAttention(out_ch, out_ch)
        self.SA = SpatialAttention(out_ch, out_ch)
        self.l0_block_0 = InformationRoutingBlock(out_ch, out_ch)
        self.l0_block_1 = InformationRoutingBlock(out_ch, out_ch)

    def forward(self, vessel_features, couinaud_features):
        joint_features = torch.cat((vessel_features, couinaud_features), dim=1)
        fused_features = self.F_tr(joint_features)
        channel_features = self.CA(fused_features)
        attended_features = self.SA(channel_features)
        vessel_routing, couinaud_routing = self.l0_block_0(attended_features), self.l0_block_1(attended_features)
        return vessel_routing, couinaud_routing


class AttentionCoLearningNet(nn.Module):
    """Shared encoder and two task decoders coupled by attention and information routing."""
    def __init__(self, in_ch, vessel_classes, couinaud_classes, dropout=False):
        super(AttentionCoLearningNet, self).__init__()
        channels1, channels2, channels3 = 32, 64, 128
        self.in_tr = InputTransition(in_ch, 16)
        self.down_tr32 = DownTransition(16, channels1, depth=1, dropout=dropout)
        self.down_tr64 = DownTransition(channels1, channels2, depth=2, dropout=dropout)
        self.down_tr128 = DownTransition(channels2, channels3, depth=2, dropout=dropout)

        # Vessel decoder
        self.up_sa128_b1 = UpsampleBlock(channels3, channels3, dropout=dropout)
        self.up_tr128_b1 = DecoderFusionBlock(channels3, depth=2, dropout=dropout)

        self.up_sa64_b1 = UpsampleBlock(channels3, channels2, dropout=dropout)
        self.up_tr64_b1 = DecoderFusionBlock(channels2, depth=2, dropout=dropout)
        
        self.up_sa32_b1 = UpsampleBlock(channels2, channels1, dropout=dropout)
        self.up_tr32_b1 = DecoderFusionBlock(channels1, depth=1, dropout=dropout)
        
        self.out_tr_b1 = OutputTransition(channels1, vessel_classes)

        # Couinaud decoder
        self.up_sa128_b2 = UpsampleBlock(channels3, channels3, dropout=dropout)
        self.up_tr128_b2 = DecoderFusionBlock(channels3, depth=2, dropout=dropout)
        
        self.up_sa64_b2 = UpsampleBlock(channels3, channels2, dropout=dropout)
        self.up_tr64_b2 = DecoderFusionBlock(channels2, depth=2, dropout=dropout)
        
        self.up_sa32_b2 = UpsampleBlock(channels2, channels1, dropout=dropout)
        self.up_tr32_b2 = DecoderFusionBlock(channels1, depth=1, dropout=dropout)
        
        self.out_tr_b2 = OutputTransition(channels1, couinaud_classes)

        # ACB and task-specific IR at each decoder stage
        self.mix1 = AttentionSharingBlock(in_ch=channels3*2, out_ch=channels3, dropout=False)
        self.mix2 = AttentionSharingBlock(in_ch=channels2*2, out_ch=channels2, dropout=False)
        self.mix3 = AttentionSharingBlock(in_ch=channels1*2, out_ch=channels1, dropout=False)

    def forward(self, ct):        
        stem_features = self.in_tr(ct)
        encoder1 = self.down_tr32(stem_features)
        encoder2 = self.down_tr64(encoder1)
        encoder3 = self.down_tr128(encoder2)

        vessel_up3 = self.up_sa128_b1(encoder3, encoder2)
        couinaud_up3 = self.up_sa128_b2(encoder3, encoder2)
        vessel_ir3, couinaud_ir3 = self.mix1(vessel_up3, couinaud_up3)
        vessel_dec3 = self.up_tr128_b1(vessel_up3, vessel_ir3) 
        couinaud_dec3 = self.up_tr128_b2(couinaud_up3, couinaud_ir3)

        vessel_up2 = self.up_sa64_b1(vessel_dec3, encoder1)
        couinaud_up2 = self.up_sa64_b2(couinaud_dec3, encoder1)
        vessel_ir2, couinaud_ir2 = self.mix2(vessel_up2, couinaud_up2)
        vessel_dec2 = self.up_tr64_b1(vessel_up2, vessel_ir2)
        couinaud_dec2 = self.up_tr64_b2(couinaud_up2, couinaud_ir2)

        vessel_up1 = self.up_sa32_b1(vessel_dec2, stem_features)
        couinaud_up1 = self.up_sa32_b2(couinaud_dec2, stem_features)
        vessel_ir1, couinaud_ir1 = self.mix3(vessel_up1, couinaud_up1)
        vessel_dec1 = self.up_tr32_b1(vessel_up1, vessel_ir1)
        couinaud_dec1 = self.up_tr32_b2(couinaud_up1, couinaud_ir1)

        vessel_logits = self.out_tr_b1(vessel_dec1)
        couinaud_logits = self.out_tr_b2(couinaud_dec1)

        return vessel_logits, couinaud_logits
