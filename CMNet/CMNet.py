
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from GIAM import GIAM

##########################################################################
class LAM_Module_v2(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out
##########################################################################
class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):
        x = self.conv_first(x)
        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return [out_feature_1, out_feature_2, out_feature_3]

class Encoder2(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Encoder2, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(15, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):
        x = self.conv_first(x)
        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return [out_feature_1, out_feature_2, out_feature_3]
##########################################################################
class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

    def forward(self, outs):
        y_1, y_2, y_3=outs
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1 = self.decoder_1(x_1, feat=False)

        return [out_1, out_2, out_3]      
##########################################################################
class HMCD(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(HMCD, self).__init__()
        self.basic_block = CMDL(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = CMDL(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = CMDL(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.layer_fussion = LAM_Module_v2(in_dim=int(in_channel * 3))
        self.conv_fuss = nn.Conv2d(int(in_channel * 3), int(in_channel), kernel_size=1, bias=True)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        inp_fusion_123 = torch.cat([y_0.unsqueeze(1), y_2.unsqueeze(1), y_4.unsqueeze(1)], dim=1)

        y = self.layer_fussion(inp_fusion_123)
        y = self.conv_fuss(y)

        y = x + y
        return y
##########################################################################
class CMDL(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(CMDL, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t
##########################################################################
class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super(Encoder_Level, self).__init__()
        self.cmd = CMD(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = HMCD(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.cmd(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature
##########################################################################
class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Decoder_Level, self).__init__()
        self.cmd = CMD(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = HMCD(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.cmd(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)
        out = self.conv(x)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out
##########################################################################
class CMD(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(CMD, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x
##########################################################################
class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out
##########################################################################
class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out

##########################################################################
class CMNet(nn.Module):
    def __init__(self,
                 en_feature_num=48,
                 en_inter_num=32,
                 de_feature_num=64,
                 de_inter_num=32,
                 sam_number=1,
                 ):
        super(CMNet, self).__init__()
        self.stage1_encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number)
        self.stage1_decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               sam_number=sam_number)

        self.stage2_encoder = Encoder2(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number)
        self.stage2_decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               sam_number=sam_number)
                              
        self.stage3_encoder = GIAM(upscale=1, img_size=(64, 64),in_chans=15,
                   window_size=8, img_range=1., depths=[2, 2,2,2],
                   embed_dim=60, num_heads=[4, 4,4,4], mlp_ratio=2, upsampler='pixelshuffledirect')
        self.conv  = conv(12, 3, kernel_size=1)
        self.conv2  = conv(15, 3, kernel_size=1)

    def forward(self, x3_img):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)
        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches
        # Two Patches for Stage 2
        x2top_img  = x3_img[:,:,0:int(H/2),:]
        x2bot_img  = x3_img[:,:,int(H/2):H,:]
        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:,:,:,0:int(W/2)]
        x1rtop_img = x2top_img[:,:,:,int(W/2):W]
        x1lbot_img = x2bot_img[:,:,:,0:int(W/2)]
        x1rbot_img = x2bot_img[:,:,:,int(W/2):W]

        ##-------------- Stage 1--------------------
        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop_img)
        feat1_rtop = self.stage1_encoder(x1rtop_img)
        feat1_lbot = self.stage1_encoder(x1lbot_img)
        feat1_rbot = self.stage1_encoder(x1rbot_img)
        ## Concat deep features
        feat1_top = [torch.cat((k,v), 3) for k,v in zip(feat1_ltop,feat1_rtop)]
        feat1_bot = [torch.cat((k,v), 3) for k,v in zip(feat1_lbot,feat1_rbot)]
        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)
        ## Output image at Stage 1
        stage1_img = torch.cat([res1_top[0], res1_bot[0]],2) 
        stage1_img=self.conv(stage1_img)

        ##-------------- Stage 2---------------------
        x2top_cat=torch.cat([x2top_img, res1_top[0]], 1)
        x2bot_cat=torch.cat([x2bot_img, res1_bot[0]], 1)

        feat2_top = self.stage2_encoder(x2top_cat)
        feat2_bot = self.stage2_encoder(x2bot_cat)
        ## Concat deep features
        feat2 = [torch.cat((k,v), 2) for k,v in zip(feat2_top,feat2_bot)]
        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)
        stage2_img=res2[0]
        stage2_img=self.conv(stage2_img)
        ##-------------- Stage 3---------------------
        x3_cat=torch.cat([x3_img, res2[0]],1)
        x3_cat =self.stage3_encoder(x3_cat)
        stage3_img=self.conv2(x3_cat)

        return [stage3_img, stage2_img, stage1_img]
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
