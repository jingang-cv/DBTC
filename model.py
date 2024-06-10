

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
from models.blocks import *
from cross_backbone import *


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.3):

        super(Conv2d_cd, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.theta = torch.nn.Parameter(torch.FloatTensor([0.3]), requires_grad=True)
       
    def forward(self, x):
        pad_x = self.reflection_pad(x)
        out_normal = self.conv(pad_x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return Conv2d_cd(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class CDCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CDCALayer, self).__init__()
        # global average pooling: feature --> point
        self.channel = channel
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.weight = torch.nn.Parameter(torch.FloatTensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).repeat([channel, 1, 1, 1]), requires_grad=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        pad_x = self.reflection_pad(x)
        y = F.conv2d(pad_x, self.weight, stride=1, padding=0, groups=self.channel)
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Central Difference Channel Attention Block (CDCAB)
class CDCAB(nn.Module):
    def __init__(
        self, conv, n_feat, reduction, out_feat,
        bias=True, bn=False, act=nn.LeakyReLU(inplace=True), res_scale=1):

        super(CDCAB, self).__init__()
        modules_body = []
        self.conv1 = conv(n_feat, out_feat, 3, bias=bias)
        modules_body.append(act)
        modules_body.append(conv(out_feat, out_feat, 3, bias=bias))
        modules_body.append(CDCALayer(out_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.conv1(x)
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops   

class Uformer(nn.Module):
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=32, depths=[1, 2, 4, 8, 4, 8, 4, 2, 1], num_heads=[1, 2, 4, 8, 16, 8, 4, 2, 1],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='ffn', se_layer=False,
                 dowsample=Downsample, upsample=Upsample,depths1=[2, 2],num_heads1=[8, 8],depths2=[1, 1, 1, 1],num_heads2=[8, 8, 8, 8],**kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output

            # Transformer
        self.input0 = PatchEmbed_cross(img_size=128, patch_size=[1,3,5], in_chans=32, embed_dim=32)
        self.input1 = PatchEmbed_cross(img_size=64, patch_size=[1,3,5], in_chans=64, embed_dim=64)
        self.input2 = PatchEmbed_cross(img_size=32, patch_size=[1,3,5], in_chans=128, embed_dim=128)
        self.input3 = PatchEmbed_cross(img_size=16, patch_size=[1,3,5], in_chans=256, embed_dim=256)
        self.input4 = PatchEmbed_cross(img_size=16, patch_size=[1,3,5], in_chans=256, embed_dim=256)
        self.input5 = PatchEmbed_cross(img_size=16, patch_size=[1,3,5], in_chans=512, embed_dim=512)
        self.input6 = PatchEmbed_cross(img_size=32, patch_size=[1,3,5], in_chans=256, embed_dim=256)
        self.input7 = PatchEmbed_cross(img_size=64, patch_size=[1,3,5], in_chans=128, embed_dim=128)
        self.input8 = PatchEmbed_cross(img_size=128,patch_size= [1,3,5], in_chans=64, embed_dim=64)


            #CNN
        self.shallow_conv = Conv2d_cd(3,32,3,1,1)
        self.shallow_act = nn.LeakyReLU(inplace=True)

        # Encoder
            #CNN
        self.conv_encoder0_0 = CDCAB(default_conv,32,16,32)
        self.conv_encoder0_1 = CDCAB(default_conv,32,16,32)

        self.conv_encoder1_0 = CDCAB(default_conv,64,16,64)
        self.conv_encoder1_1 = CDCAB(default_conv,64,16,64)

        self.conv_encoder2_0 = CDCAB(default_conv,128,16,64)
        self.conv_encoder2_1 = CDCAB(default_conv,64,16,64)

        self.conv_encoder3_0 = CDCAB(default_conv,256,16,64)
        self.conv_encoder3_1 = CDCAB(default_conv,64,16,64)
            #Transformer

        self.encoderlayer_0 = Stage(32,128,2,2,8,4)
        self.encoderlayer_1 = Stage(64,64,2,4,8,4)
        self.encoderlayer_2 = Stage(128,32,2,4,8,2)
        self.encoderlayer_3 = Stage(256,16,2,8,8,2)   
            #Fusion
        self.sigmoid = nn.Sigmoid()

        self.patch_unembed_0 = PatchUnEmbed(img_size=128, patch_size=1, in_chans=256, embed_dim=32,norm_layer=None)
        self.modulation_conv0 = Conv2d_cd(32,32,3,1,1)
        self.patch_embed_0 = PatchEmbed(
            img_size=128, patch_size=1, in_chans=64, embed_dim=64,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        self.fusion_conv0 = Conv2d_cd(64,32,1,1,0)

        self.patch_unembed_1 = PatchUnEmbed(img_size=64, patch_size=1, in_chans=64, embed_dim=64,norm_layer=None)
        self.modulation_conv1 = Conv2d_cd(64,64,3,1,1)
        self.patch_embed_1 = PatchEmbed(
            img_size=64, patch_size=1, in_chans=128, embed_dim=128,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        self.fusion_conv1 = Conv2d_cd(128,64,1,1,0)    

        self.patch_unembed_2 = PatchUnEmbed(img_size=32, patch_size=1, in_chans=128, embed_dim=128,norm_layer=None)
        self.modulation_conv2 = Conv2d_cd(64,128,3,1,1)
        self.patch_embed_2 = PatchEmbed(
            img_size=32, patch_size=1, in_chans=128, embed_dim=128,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        self.fusion_conv2 = Conv2d_cd(256,128,1,1,0) 

        self.patch_unembed_3 = PatchUnEmbed(img_size=16, patch_size=1, in_chans=256, embed_dim=256,norm_layer=None)
        self.modulation_conv3 = Conv2d_cd(64,256,3,1,1)
        self.patch_embed_3 = PatchEmbed(
            img_size=16, patch_size=1, in_chans=256, embed_dim=256,
            norm_layer=nn.LayerNorm if self.patch_norm else None) 
        self.fusion_conv3 = Conv2d_cd(512,256,1,1,0)        

        #pool
        self.pool0 = nn.Conv2d(32,64,4,2,1)
        self.pool1 = nn.Conv2d(64,128,4,2,1)
        self.pool2 = nn.Conv2d(128,256,4,2,1)
        #up
        self.up1 =  nn.ConvTranspose2d(512, 128, 2, 2)
        self.up2 =  nn.ConvTranspose2d(256, 64, 2, 2)
        self.up3 =  nn.ConvTranspose2d(128, 32, 2, 2)

        #Bottom
            #CNN
        self.bottom_cnn_1 = CDCAB(default_conv,256,16,64)
        self.bottom_cnn_2 = CDCAB(default_conv,64,16,64)
            #Transformer
        self.bottom_trans = Stage(256,16,2,8,8,2)
            #Fusion
        self.patch_embed_bottom = PatchEmbed(
            img_size=16, patch_size=1, in_chans=256, embed_dim=256,
            norm_layer=nn.LayerNorm if self.patch_norm else None) 
        self.patch_unembed_bottom = PatchUnEmbed(img_size=16, patch_size=1, in_chans=256, embed_dim=256,norm_layer=None)
        self.modulation_conv_bottom = Conv2d_cd(64,256,3,1,1)
        self.fusion_conv_bottom = Conv2d_cd(512,256,1,1,0)   

        # Decoder
            #CNN
        #deocoder0
        self.conv_decoder0_0 = CDCAB(default_conv,512,16,64)
        self.conv_decoder0_1 = CDCAB(default_conv,64,16,64)

        self.conv_decoder1_0 = CDCAB(default_conv,256,16,64)
        self.conv_decoder1_1 = CDCAB(default_conv,64,16,64)
 
        self.conv_decoder2_0 = CDCAB(default_conv,128,16,64)
        self.conv_decoder2_1 = CDCAB(default_conv,64,16,64)

        self.conv_decoder3_0 = CDCAB(default_conv,64,16,64)
        self.conv_decoder3_1 = CDCAB(default_conv,64,16,64)
            #Transformer
        self.decoderlayer_0 = Stage(512,16,2,8,8,2)
        self.decoderlayer_1 = Stage(256,32,2,4,8,2)
        self.decoderlayer_2 = Stage(128,64,2,4,8,4)
        self.decoderlayer_3 = Stage(64,128,2,2,8,4)
            #Fusion
        self._patch_embed_0 = PatchEmbed(
            img_size=16, patch_size=1, in_chans=512, embed_dim=512,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        self._patch_unembed_0 = PatchUnEmbed(img_size=16, patch_size=1, in_chans=512, embed_dim=512,norm_layer=None)
        self._modulation_conv_0 = Conv2d_cd(64,512,3,1,1)
        self._fusion_conv_0 = Conv2d_cd(1024,512,1,1,0)

        self._patch_embed_1 = PatchEmbed(
            img_size=16, patch_size=1, in_chans=256, embed_dim=256,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        self._patch_unembed_1 = PatchUnEmbed(img_size=32, patch_size=1, in_chans=256, embed_dim=256,norm_layer=None)
        self._modulation_conv_1 = Conv2d_cd(64,256,3,1,1)
        self._fusion_conv_1 = Conv2d_cd(512,256,1,1,0)

        self._patch_embed_2 = PatchEmbed(
            img_size=16, patch_size=1, in_chans=128, embed_dim=128,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        self._patch_unembed_2 = PatchUnEmbed(img_size=64, patch_size=1, in_chans=128, embed_dim=128,norm_layer=None)
        self._modulation_conv_2 = Conv2d_cd(64,128,3,1,1)
        self._fusion_conv_2 = Conv2d_cd(256,128,1,1,0)

        self._patch_embed_3 = PatchEmbed(
            img_size=16, patch_size=1, in_chans=64, embed_dim=64,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        self._patch_unembed_3 = PatchUnEmbed(img_size=64, patch_size=1, in_chans=64, embed_dim=64,norm_layer=None)
        self._modulation_conv_3 = Conv2d_cd(64,64,3,1,1)
        self._fusion_conv_3 = Conv2d_cd(128,64,1,1,0)

        self.out = Conv2d_cd(64,3,3,1,1)
        self.out_act = nn.LeakyReLU(inplace=True)


        self.apply(self._init_weights)
        #channel_change
        self.channel_change1 = PatchUnEmbed(
            img_size=16, patch_size=1, in_chans=256, embed_dim=256,
            norm_layer=None)
        self.channel_change1_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.channel_change2 = PatchEmbed(
            img_size=16, patch_size=1, in_chans=256, embed_dim=256,
            norm_layer=None)
        self.channel_change2_1 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"
    
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        # Input Projection
            #CNN
        y = self.shallow_conv(x)
        y = self.shallow_act(y)

            #Transforemr  
        y_trans = self.input0(y)[0]
                                                                    #(1,16384,32)
        y_trans = self.pos_drop(y_trans)

        # print(y.shape)
        #Encoder
        #Encoder 0
            #CNN
        conv0_cnn = self.conv_encoder0_0(y)
        conv0_cnn = self.conv_encoder0_1(conv0_cnn)
            #Transformer
        conv0_trans = self.encoderlayer_0(y_trans,128,128)
        # print(conv0.shape)                                                    #(1,16384,32)
            #Fusion
        conv0_trans_fusion = self.patch_unembed_0(conv0_trans,(128,128))
        conv0_cnn_fusion = self.modulation_conv0(conv0_cnn)
        conv0_cnn_fusion_sigmoid = self.sigmoid(conv0_cnn_fusion)
        conv0_modulation = conv0_cnn_fusion_sigmoid * conv0_trans_fusion 
        encoder0_output = torch.cat([conv0_cnn_fusion,conv0_modulation],1)
        encoder0_output = self.fusion_conv0(encoder0_output)
        # print(encoder0_output.shape)

        #Pool 0
        pool0 = self.pool0(encoder0_output)

        #Encoder 1
            #CNN
        conv1_cnn = self.conv_encoder1_0(pool0)
        conv1_cnn = self.conv_encoder1_1(conv1_cnn)
            #Transformer
        conv1_trans = self.input1(pool0)[0]
        # conv1_trans = self.patch_embed_0(pool0)
        conv1_trans = self.encoderlayer_1(conv1_trans,64,64)
            #Fusion
        conv1_trans_fusion = self.patch_unembed_1(conv1_trans,(64,64))
        conv1_cnn_fusion = self.modulation_conv1(conv1_cnn)
        conv1_cnn_fusion_sigmoid = self.sigmoid(conv1_cnn_fusion)
        conv1_modulation = conv1_cnn_fusion_sigmoid * conv1_trans_fusion 
        encoder1_output = torch.cat([conv1_cnn_fusion,conv1_modulation],1)
        encoder1_output = self.fusion_conv1(encoder1_output)  # 1, 64, 64, 64

        #Pool1
        pool1 = self.pool1(encoder1_output) # 1, 128, 32, 32

        #Encoder 2
            #CNN
        conv2_cnn = self.conv_encoder2_0(pool1)
        conv2_cnn = self.conv_encoder2_1(conv2_cnn)
            #Transformer
        conv2_trans = self.input2(pool1)[0]
        # conv2_trans = self.patch_embed_2(pool1)
        conv2_trans = self.encoderlayer_2(conv2_trans,32,32)
            #Fusion
        conv2_trans_fusion = self.patch_unembed_2(conv2_trans,(32,32))
        conv2_cnn_fusion = self.modulation_conv2(conv2_cnn)
        conv2_cnn_fusion_sigmoid = self.sigmoid(conv2_cnn_fusion)
        conv2_modulation = conv2_cnn_fusion_sigmoid * conv2_trans_fusion 
        encoder2_output = torch.cat([conv2_cnn_fusion,conv2_modulation],1)
        encoder2_output = self.fusion_conv2(encoder2_output)  # 1, 128, 32, 32

        #Pool 2
        pool2 = self.pool2(encoder2_output)

        #Encoder 3
            #CNN
        conv3_cnn = self.conv_encoder3_0(pool2)
        conv3_cnn = self.conv_encoder3_1(conv3_cnn)
            #Transformer
        conv3_trans = self.input3(pool2)[0]
        # conv3_trans = self.patch_embed_3(pool2)
        conv3_trans = self.encoderlayer_3(conv3_trans,16,16)
            #Fusion
        conv3_trans_fusion = self.patch_unembed_3(conv3_trans,(16,16))
        conv3_cnn_fusion = self.modulation_conv3(conv3_cnn)
        conv3_cnn_fusion_sigmoid = self.sigmoid(conv3_cnn_fusion)
        conv3_modulation = conv3_cnn_fusion_sigmoid * conv3_trans_fusion 
        encoder3_output = torch.cat([conv3_cnn_fusion,conv3_modulation],1)
        encoder3_output = self.fusion_conv3(encoder3_output)
        # print(encoder3_output.shape)

        #Bottom
            #CNN
        convb_cnn= self.bottom_cnn_1(encoder3_output)
        convb_cnn= self.bottom_cnn_2(convb_cnn)
            #Transformer
        convb_trans = self.input4(encoder3_output)[0]
        # convb_trans = self.patch_embed_bottom(encoder3_output)
        convb_trans = self.bottom_trans(convb_trans,16,16)
            #Fusion
        convb_trans_fusion = self.patch_unembed_bottom(convb_trans,(16,16))
        convb_cnn_fusion = self.modulation_conv_bottom(convb_cnn)
        convb_cnn_fusion_sigmoid = self.sigmoid(convb_cnn_fusion)
        convb_modulation = convb_cnn_fusion_sigmoid * convb_trans_fusion
        bottom_output =  torch.cat([convb_cnn_fusion,convb_modulation],1)
        bottom_output = self.fusion_conv_bottom(bottom_output)

        
        #skip connection
        decoder_cnn_input0 =  torch.cat([bottom_output,encoder3_output],1)
        decoder_trans_input0 = self.input5(decoder_cnn_input0)[0]
        # decoder_trans_input0 = self._patch_embed_0(decoder_cnn_input0)

        #Decoder 
        #Decoder 0 
            #CNN
        deconv0_cnn = self.conv_decoder0_0(decoder_cnn_input0)
        deconv0_cnn = self.conv_decoder0_1(deconv0_cnn)
            #Transformer
        deconv0_trans = self.decoderlayer_0(decoder_trans_input0,16,16)
            #Fusion
        deconv0_trans_fusion = self._patch_unembed_0(deconv0_trans,(16,16))
        deconv0_cnn_fusion = self._modulation_conv_0(deconv0_cnn)
        deconv0_cnn_fusion_sigmoid = self.sigmoid(deconv0_cnn_fusion)
        deconv0_modulation = deconv0_cnn_fusion_sigmoid * deconv0_trans_fusion
        deconv0_output = torch.cat([deconv0_cnn_fusion,deconv0_modulation],1)
        deconv0_output = self._fusion_conv_0(deconv0_output)

        #Up 0 
        up0 = self.up1(deconv0_output)
        # print(up0.shape)
        #skip connection
        decoder_cnn_input1 = torch.cat([up0,encoder2_output],1)
        decoder_trans_input1 = self.input6(decoder_cnn_input1)[0]
        # decoder_trans_input1 = self._patch_embed_1(decoder_cnn_input1)
        # print(decoder_trans_input1.shape)

        #Decoder 1
            #CNN
        deconv1_cnn = self.conv_decoder1_0(decoder_cnn_input1)
        deconv1_cnn = self.conv_decoder1_1(deconv1_cnn)
        # print(deconv1_cnn.shape)
            #Transformer
        deconv1_trans = self.decoderlayer_1(decoder_trans_input1,32,32)
        # print(deconv1_trans.shape)
            #Fusion
        deconv1_trans_fusion = self._patch_unembed_1(deconv1_trans,(32,32))
        deconv1_cnn_fusion = self._modulation_conv_1(deconv1_cnn)
        deconv1_cnn_fusion_sigmoid = self.sigmoid(deconv1_cnn_fusion)
        deconv1_modulation = deconv1_cnn_fusion_sigmoid * deconv1_trans_fusion
        deconv1_output = torch.cat([deconv1_cnn_fusion,deconv1_modulation],1)
        deconv1_output = self._fusion_conv_1(deconv1_output)
        # print(deconv1_output.shape)
        # Up 1
        up1 = self.up2(deconv1_output)
        # skip connection
        decoder_cnn_input2 = torch.cat([up1,encoder1_output],1)
        # print(decoder_cnn_input2.shape)
        decoder_trans_input2 = self.input7(decoder_cnn_input2)[0]
        # decoder_trans_input2 = self._patch_embed_2(decoder_cnn_input2)
        # print(decoder_trans_input2.shape)

        #Decoder 2
            #CNN
        deconv2_cnn = self.conv_decoder2_0(decoder_cnn_input2)
        deconv2_cnn = self.conv_decoder2_1(deconv2_cnn)
            #Transformer
        deconv2_trans = self.decoderlayer_2(decoder_trans_input2,64,64)
        # print(deconv2_cnn.shape)
        # print(deconv2_trans.shape)
        # Fusion
        deconv2_trans_fusion = self._patch_unembed_2(deconv2_trans,(64,64))
        deconv2_cnn_fusion = self._modulation_conv_2(deconv2_cnn)
        deconv2_cnn_fusion_sigmoid = self.sigmoid(deconv2_cnn_fusion)
        deconv2_modulation = deconv2_cnn_fusion_sigmoid * deconv2_trans_fusion
        deconv2_output = torch.cat([deconv2_cnn_fusion,deconv2_modulation],1)
        deconv2_output = self._fusion_conv_2(deconv2_output)
        # print(deconv2_output.shape)

        #Up 2
        up2 = self.up3(deconv2_output)
        # skip connection
        decoder_cnn_input3 = torch.cat([up2,encoder0_output],1)
        # print(decoder_cnn_input3.shape)
        decoder_trans_input3 = self.input8(decoder_cnn_input3)[0]
        # decoder_trans_input3 = self._patch_embed_3(decoder_cnn_input3)
        # print(decoder_trans_input3.shape)

        #Decoder 3
            #CNN
        deconv3_cnn = self.conv_decoder3_0(decoder_cnn_input3)
        deconv3_cnn = self.conv_decoder3_1(deconv3_cnn)
            #Transformer
        deconv3_trans = self.decoderlayer_3(decoder_trans_input3,128,128)
        #Fusion
        deconv3_trans_fusion = self._patch_unembed_3(deconv3_trans,(128,128))
        deconv3_cnn_fusion = self._modulation_conv_3(deconv3_cnn)
        deconv3_cnn_fusion_sigmoid = self.sigmoid(deconv3_cnn_fusion)
        deconv3_modulation = deconv3_cnn_fusion_sigmoid * deconv3_trans_fusion
        deconv3_output = torch.cat([deconv3_cnn_fusion,deconv3_modulation],1)
        deconv3_output = self._fusion_conv_3(deconv3_output)
        # output_middle.tensor_to_PIL(deconv3_output, 'o')
        
        final_output = self.out(deconv3_output)
        # final_output = self.out_act(final_output)

        
        return x+final_output


