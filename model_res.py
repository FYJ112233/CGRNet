#this work base on non-lcal block
# we aims to use channel non-local to reduce modality discrepancy and two spatial non-local to reduce pose transformation



import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import numpy as np

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x

    


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        #x = self.base.layer1(x)
        #x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class base_resblock(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resblock, self).__init__()

        model_base = resnet50(pretrained=True,last_conv_stride=1, last_conv_dilation=1)

        model_base.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base = model_base

    def forward(self, x, idx):
        if idx not in [1,2,3,4]:
            raise NotImplementedError

        if idx == 1:
            x = self.base.layer1(x)
        elif idx == 2:
            x = self.base.layer2(x)
        elif idx == 3:
            x = self.base.layer3(x)
        else:
            x = self.base.layer4(4)

        return x



class MixtureOfSoftMaxACF(nn.Module):
    """"Mixture of SoftMax"""
    def __init__(self, n_mix, d_k, attn_dropout=0.1):
        super(MixtureOfSoftMaxACF, self).__init__()
        self.temperature = np.power(d_k, 0.5)
        self.n_mix = n_mix
        self.att_drop = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.d_k = d_k
        if n_mix > 1:
            self.weight = nn.Parameter(torch.Tensor(n_mix, d_k))
            std = np.power(n_mix, -0.5)
            self.weight.data.uniform_(-std, std)

    def forward(self, qt, kt, vt):
        B, d_k, N = qt.size()
        m = self.n_mix
        assert d_k == self.d_k
        d = d_k // m
        if m > 1:
            # \bar{v} \in R^{B, d_k, 1}
            bar_qt = torch.mean(qt, 2, True)
            # pi \in R^{B, m, 1}
            pi = self.softmax1(torch.matmul(self.weight, bar_qt)).view(B*m, 1, 1)
        # reshape for n_mix
        q = qt.view(B*m, d, N).transpose(1, 2)
        N2 = kt.size(2)
        kt = kt.view(B*m, d, N2)
        v = vt.transpose(1, 2)
        # {Bm, N, N}
        attn = torch.bmm(q, kt)
        attn = attn / self.temperature
        attn = self.softmax2(attn)
        attn = self.dropout(attn)
        if m > 1:
            # attn \in R^{Bm, N, N2} => R^{B, N, N2}
            attn = (attn * pi).view(B, m, N, N2).sum(1)
        output = torch.bmm(attn, v)
        return output, attn

class _Decoder_block(nn.Module):
    def __init__(self,in_channels,middle_channel,out_channels):
        super(_Decoder_block,self).__init__()
        self.decode = nn.Sequential(nn.Conv2d(in_channels,middle_channel,kernel_size=3,padding=1),nn.BatchNorm2d(middle_channel),nn.ReLU(inplace=True),
                    nn.Conv2d(middle_channel,middle_channel,kernel_size=3,padding=1),nn.BatchNorm2d(middle_channel),nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(middle_channel,out_channels,kernel_size=2,stride=2))

    def forward(self,x):
        return self.decode(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)



class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        # pool_dim = 64

        self.thermal_block = base_resblock(arch=arch)
        self.visible_block = base_resblock(arch=arch)
        #self.discriminator_rgb = Discriminator(n_layer=4, middle_dim=32, num_scales=2)
        #self.discriminator_ir = Discriminator(n_layer=4, middle_dim=32, num_scales=2)


        
        self.cam1 = CAM_Module(64)
        self.cam2 = CAM_Module(64)

        #self.decoder11 = _Decoder_block(256,128,64)
        #self.decoder12 = _Decoder_block(64,32,3)

        #self.decoder21 = _Decoder_block(256,128,64)
        #self.decoder22 = _Decoder_block(64,32,3)

        #self.decoder1 = Decoder(n_upsample=2,n_res=2,dim=256, output_dim=3, res_norm='adain', activ='relu', pad_type='reflect')
        #self.decoder2 = Decoder(n_upsample=2,n_res=2,dim=256, output_dim=3, res_norm='adain', activ='relu', pad_type='reflect')

        ####spatial align###
        #self.cross_pam1 = PAM_Module(512)
        #self.cross_pam2 = PAM_Module(512)

       
        
        self.local2local = PAM_Module(512)
        self.local2global = PAM_Module(512)
        self.global2local = PAM_Module(512)

        self.common_block = base_resblock(arch=arch)

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift = False

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        ####Initialization#####

        self.bottleneck.apply(weights_init_kaiming)
        self.cam1.apply(weights_init_kaiming)
        self.cam2.apply(weights_init_kaiming)
        self.local2global.apply(weights_init_kaiming)
        self.local2local.apply(weights_init_kaiming)
        self.global2local.apply(weights_init_kaiming)


        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x1_rgb, x2_ir, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1_rgb)
        
            x1 = self.visible_block(x1,idx=1)
            x2 = self.thermal_module(x2_ir)
     
            x2 = self.thermal_block(x2,idx=1)

            # x1_cam, atten1 = self.cam1(x1,x1,True)
            # x2_cam, atten2 = self.cam2(x2,x2,True)

            x = torch.cat((x1, x2), 0)
            # x = torch.cat((x1_cam,x2_cam),0)


      
        elif modal == 1:
            x1 = self.visible_module(x1_rgb)
            x1 = self.visible_block(x1,idx=1)
            x = self.cam1(x1,x1)
            #x = self.pam1(x,x)


        elif modal == 2:
            x2 = self.thermal_module(x2_ir)
            x2 = self.thermal_block(x2,idx=1)
            x = self.cam2(x2,x2)
            #x = self.pam2(x,x)

        x = self.common_block(x,idx=2)

        # x_local2local = self.local2local(x,x)
        #
        # x_global = self.avgpool(x)
        # x_global = x_global.expand_as(x)
        #
        # x_global2local = self.global2local(x,x_global)
        #
        # x_local2global = self.local2global(x_global,x)
        #
        # x = x + x_global2local +x_local2local + x_local2global





        x = self.base_resnet(x)


        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        feat = self.bottleneck(x_pool)

        atten1, atten2 = 0, 0
        if self.training:
           return x_pool, self.classifier(feat), atten1, atten2
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

class ReID_net(nn.Module):
    def __init__(self, arch,class_num):
        super(ReID_net,self).__init__()
        self.common_block = base_resblock(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift = False

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self,x,training=False):
        x = self.common_block(x,idx=2)
        x = self.base_resnet(x)
        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if training == True:
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)
            


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x,y, atten=False):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C,height, width = x.size()
        assert x.size() == y.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(y).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize,C, height, width)

        out = self.gamma*out + x

        if atten == False:
            return out
        else:
            return out, attention


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x,y,atten=False):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize,C, height, width = x.size()
        assert x.size() == y.size()

        proj_query = x.view(m_batchsize, C, -1)
        proj_key = y.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize,C, height, width)

        out = self.gamma*out + x
        if atten:
            return out, attention
        else:
            return out


class ACFModule(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, n_mix, d_model, d_k, d_v, norm_layer=torch.nn.BatchNorm2d,
                 kq_transform='conv', value_transform='conv',
                 pooling=True, concat=False, dropout=0.1):
        super(ACFModule, self).__init__()

        self.n_head = n_head
        self.n_mix = n_mix
        self.d_k = d_k
        self.d_v = d_v
        self.pooling = pooling
        self.concat = concat

        if self.pooling:
            self.pool = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

        if kq_transform == 'conv':
            self.conv_qs = nn.Conv2d(d_model, n_head*d_k, 1)
            nn.init.normal_(self.conv_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        elif kq_transform == 'dffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head*d_k, 3, padding=1, bias=False),
                norm_layer(n_head*d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head*d_k, n_head*d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        elif kq_transform == 'dffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head*d_k, 3, padding=4, dilation=4, bias=False),
                norm_layer(n_head*d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head*d_k, n_head*d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        else:
            raise NotImplemented
        #self.conv_ks = nn.Conv2d(d_model, n_head*d_k, 1)
        self.conv_ks = self.conv_qs
        if value_transform == 'conv':
            self.conv_vs = nn.Conv2d(d_model, n_head*d_v, 1)
        else:
            raise NotImplemented

        #nn.init.normal_(self.conv_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.conv_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = MixtureOfSoftMaxACF(n_mix=n_mix, d_k=d_k)

        self.conv = nn.Conv2d(n_head*d_v, d_model, 1, bias=False)
        self.norm_layer = norm_layer(d_model)

    def forward(self, x):
        residual = x

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_, c_, h_, w_ = x.size()

        if self.pooling:
            qt = self.conv_ks(x).view(b_*n_head, d_k, h_*w_)
            kt = self.conv_ks(self.pool(x)).view(b_*n_head, d_k, h_*w_//4)
            vt = self.conv_vs(self.pool(x)).view(b_*n_head, d_v, h_*w_//4)
        else:
            kt = self.conv_ks(x).view(b_*n_head, d_k, h_*w_)
            qt = kt
            vt = self.conv_vs(x).view(b_*n_head, d_v, h_*w_)

        output, attn = self.attention(qt, kt, vt)

        output = output.transpose(1, 2).contiguous().view(b_, n_head*d_v, h_, w_)

        output = self.conv(output)
        if self.concat:
            output = torch.cat((self.norm_layer(output), residual), 1)
        else:
            output = self.norm_layer(output) + residual
        return output



######Distribtor##
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


##################################################################################
# Discriminator
##################################################################################

class Discriminator(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, n_layer, middle_dim, num_scales):
        super(Discriminator, self).__init__()

        self.input_dim = 3
        self.gan_type = 'lsgan'
        self.norm = 'none'
        self.activ = 'lrelu'
        self.pad_type = 'reflect'

        self.n_layer = n_layer # 4
        self.middle_dim = middle_dim # 32
        self.num_scales = num_scales # 3
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.middle_dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class Conv2dBlock(nn.Module):
    
    def __init__(self, input_dim ,output_dim, kernel_size, stride, padding, norm, activation, pad_type):
        super(Conv2dBlock, self).__init__()

        self.use_bias = True

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation, pad_type):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x



