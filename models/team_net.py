"""Model definition."""

import torch
from torch import nn
from models import transforms
import torchvision
import torch.nn.functional as F

import pdb


class TEAM(nn.Module):
    def __init__(self, mv_channels, i_channels, ratio, num_segments):
        super(TEAM, self).__init__()
        self._num_segments = num_segments
        self.fc_squeeze = nn.Linear(mv_channels*2+i_channels, 
                                    (mv_channels*2+i_channels)//ratio,
                                    bias=False)
        self.fc_i = nn.Linear((mv_channels*2+i_channels)//ratio, 
                              i_channels,
                              bias=False)
        self.fc_mv = nn.Linear((mv_channels*2+i_channels)//ratio, 
                              mv_channels,
                              bias=False)
        self.fc_r = nn.Linear((mv_channels*2+i_channels)//ratio, 
                              mv_channels,
                              bias=False)

        self.conv_s_i = nn.Conv2d(3, 1, 
                              kernel_size=3, padding=1, groups=1,
                              bias=False)

        self.conv_s_mv = nn.Conv2d(3, 1, 
                              kernel_size=3, padding=1, groups=1,
                              bias=False)

        self.conv_s_r = nn.Conv2d(3, 1, 
                              kernel_size=3, padding=1, groups=1,
                              bias=False)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, i_x_input, mv_x_input, r_x_input):
        # channel attention
        i_x_c = i_x_input.mean(-1).mean(-1)
        mv_x_c = mv_x_input.mean(-1).mean(-1)
        r_x_c = r_x_input.mean(-1).mean(-1)
        i_mv_r_c = torch.cat([i_x_c, mv_x_c, r_x_c], 1)

        zc_squeeze = self.fc_squeeze(i_mv_r_c)
        zc_squeeze = self.relu(zc_squeeze)
        zc_i = self.fc_i(zc_squeeze)
        zc_mv = self.fc_mv(zc_squeeze)
        zc_r = self.fc_r(zc_squeeze)
        zc_i = self.sigmoid(zc_i)
        zc_mv = self.sigmoid(zc_mv)
        zc_r = self.sigmoid(zc_r)
        zc_i_out = zc_i.unsqueeze(-1).unsqueeze(-1) * i_x_input
        zc_mv_out = zc_mv.unsqueeze(-1).unsqueeze(-1) * mv_x_input
        zc_r_out = zc_r.unsqueeze(-1).unsqueeze(-1) * r_x_input

        # spatial attention
        i_x_s = zc_i_out.mean(1, keepdim=True)
        i_mv_s = zc_mv_out.mean(1, keepdim=True)
        i_r_s = zc_r_out.mean(1, keepdim=True)
        i_mv_r_s = torch.cat([i_x_s, i_mv_s, i_r_s], 1)
        z_i_s = self.conv_s_i(i_mv_r_s)
        z_mv_s = self.conv_s_mv(i_mv_r_s)
        z_r_s = self.conv_s_r(i_mv_r_s)

        zs_i = self.sigmoid(z_i_s)
        zs_mv = self.sigmoid(z_mv_s)
        zs_r = self.sigmoid(z_r_s)


        zs_i_out = zs_i * zc_i_out
        zs_mv_out = zs_mv * zc_mv_out
        zs_r_out = zs_r * zc_r_out
        return zs_i_out, zs_mv_out, zs_r_out


class TEAM_Net(nn.Module):
    def __init__(self, num_class, num_segments, 
                 is_shift=False, shift_div=8, shift_place='blockres',
                 temporal_pool=False,
                 base_model='resnet152',
                 dropout = 0.5):
        super(TEAM_Net, self).__init__()
        self._enable_pbn = False
        self._num_segments = num_segments
        self.I_model = Model(num_class, num_segments, representation='iframe',
                              base_model=base_model, is_shift=is_shift, shift_div=8,
                              dropout = dropout)
        self.MV_model = Model(num_class, num_segments, representation='mv',
                              base_model='resnet18', is_shift=is_shift, shift_div=8,
                              dropout = dropout)
        self.R_model = Model(num_class, num_segments, representation='residual',
                              base_model='resnet18', is_shift=is_shift, shift_div=8,
                              dropout = dropout)
        self._ratio = 4
        if is_shift:
            self.team0 = TEAM(self.MV_model.base_model.conv1.out_channels, 
                                        self.I_model.base_model.conv1.out_channels,
                                        ratio = self._ratio, num_segments = num_segments)  
            self.team1 = TEAM(self.MV_model.base_model.layer2[0].conv1.net.in_channels, 
                                        self.I_model.base_model.layer2[0].conv1.net.in_channels,
                                        ratio = self._ratio, num_segments = num_segments)
            self.team2 = TEAM(self.MV_model.base_model.layer3[0].conv1.net.in_channels, 
                                        self.I_model.base_model.layer3[0].conv1.net.in_channels,
                                        ratio = self._ratio, num_segments = num_segments)
            self.team3 = TEAM(self.MV_model.base_model.layer4[0].conv1.net.in_channels, 
                                        self.I_model.base_model.layer4[0].conv1.net.in_channels,
                                        ratio = self._ratio, num_segments = num_segments)
        else:
            self.team0 = TEAM(self.MV_model.base_model.conv1.out_channels, 
                                        self.I_model.base_model.conv1.out_channels,
                                        ratio = self._ratio, num_segments = num_segments)  
            self.team1 = TEAM(self.MV_model.base_model.layer2[0].conv1.in_channels, 
                                        self.I_model.base_model.layer2[0].conv1.in_channels,
                                        ratio = self._ratio, num_segments = num_segments)
            self.team2 = TEAM(self.MV_model.base_model.layer3[0].conv1.in_channels, 
                                        self.I_model.base_model.layer3[0].conv1.in_channels,
                                        ratio = self._ratio, num_segments = num_segments)
            self.team3 = TEAM(self.MV_model.base_model.layer4[0].conv1.in_channels, 
                                        self.I_model.base_model.layer4[0].conv1.in_channels,
                                        ratio = self._ratio, num_segments = num_segments) 



    def forward(self, i_x, mv_x, r_x):
        i_x = i_x.view((-1, ) + i_x.size()[-3:])

        mv_x = mv_x.view((-1, ) + mv_x.size()[-3:])
        mv_x = self.MV_model.data_bn(mv_x)

        r_x = r_x.view((-1, ) + r_x.size()[-3:])
        r_x = self.R_model.data_bn(r_x)

        # layer0
        i_x = self.I_model.base_model.conv1(i_x)
        i_x = self.I_model.base_model.bn1(i_x)
        i_x = self.I_model.base_model.relu(i_x)
        i_x = self.I_model.base_model.maxpool(i_x)

        mv_x = self.MV_model.base_model.conv1(mv_x)
        mv_x = self.MV_model.base_model.bn1(mv_x)
        mv_x = self.MV_model.base_model.relu(mv_x)
        mv_x = self.MV_model.base_model.maxpool(mv_x)

        r_x = self.R_model.base_model.conv1(r_x)
        r_x = self.R_model.base_model.bn1(r_x)
        r_x = self.R_model.base_model.relu(r_x)
        r_x = self.R_model.base_model.maxpool(r_x)


        # laterial 0
        i_x, mv_x, r_x = self.team0(i_x, mv_x, r_x)

        # layer1
        i_x_res1 = self.I_model.base_model.layer1(i_x)
        mv_x_res1 = self.MV_model.base_model.layer1(mv_x)
        r_x_res1 = self.R_model.base_model.layer1(r_x)

        # laterial 1
        i_x_res1, mv_x_res1, r_x_res1 = self.team1(i_x_res1, mv_x_res1, r_x_res1)

        # layer2
        i_x_res2 = self.I_model.base_model.layer2(i_x_res1)
        mv_x_res2 = self.MV_model.base_model.layer2(mv_x_res1)
        r_x_res2 = self.R_model.base_model.layer2(r_x_res1)

        # laterial 2
        i_x_res2, mv_x_res2, r_x_res2 = self.team2(i_x_res2, mv_x_res2, r_x_res2)


        # layer3
        i_x_res3 = self.I_model.base_model.layer3(i_x_res2)
        mv_x_res3 = self.MV_model.base_model.layer3(mv_x_res2)
        r_x_res3 = self.R_model.base_model.layer3(r_x_res2)     

        # laterial 3
        i_x_res3, mv_x_res3, r_x_res3 = self.team3(i_x_res3, mv_x_res3, r_x_res3)

        # layer4
        i_x_res4 = self.I_model.base_model.layer4(i_x_res3)
        mv_x_res4 = self.MV_model.base_model.layer4(mv_x_res3)
        r_x_res4 = self.R_model.base_model.layer4(r_x_res3)



        i_x_pool = self.I_model.base_model.avgpool(i_x_res4)
        mv_x_pool = self.MV_model.base_model.avgpool(mv_x_res4)
        r_x_pool = self.R_model.base_model.avgpool(r_x_res4)


        i_x_pool = i_x_pool.squeeze()
        mv_x_pool = mv_x_pool.squeeze()
        r_x_pool = r_x_pool.squeeze()

        # pdb.set_trace()
        i_x_pool = self.I_model.base_model.fc(i_x_pool)
        mv_x_pool = self.MV_model.base_model.fc(mv_x_pool)
        r_x_pool = self.R_model.base_model.fc(r_x_pool)

        i_x_pred = self.I_model.new_fc(i_x_pool)
        mv_x_pred = self.MV_model.new_fc(mv_x_pool)
        r_x_pred = self.R_model.new_fc(r_x_pool)


        i_x_pred = i_x_pred.view((-1,self._num_segments)+i_x_pred.size()[1:]).mean(1)
        mv_x_pred = mv_x_pred.view((-1,self._num_segments)+mv_x_pred.size()[1:]).mean(1)
        r_x_pred = r_x_pred.view((-1,self._num_segments)+r_x_pred.size()[1:]).mean(1)
        out = i_x_pred + mv_x_pred + r_x_pred
        return out

    def get_optim_policies(self):
        params_dict = dict(self.named_parameters())
        params = []
        for key, value in params_dict.items():
            decay_mult = 0.0 if 'bias' or 'bn' in key else 1.0
            if 'new_fc.weight' in key:
                lr_mult = 5
            elif 'new_fc.bias' in key:
                lr_mult = 10
            else:
                lr_mult = 1

            params += [{'params': value, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]
        return params


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 is_shift=False, shift_div=8, shift_place='blockres',
                 temporal_pool=False, dropout=0.5,
                 base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self.dropout = dropout

        print(("""
    Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        self.new_fc = nn.Linear(feature_dim, num_class)

        if self._representation == 'mv':
            setattr(self.base_model, 'conv1',
                    nn.Conv2d(2, 64, 
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)


    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            self.base_model.last_layer_name = 'fc'
            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):
        input = input.view((-1, ) + input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        base_out = self.base_model(input)
        return self.new_fc(base_out)

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224


    def get_augmentation(self, data_name):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)

        return torchvision.transforms.Compose(
            [transforms.GroupMultiScaleCrop(self._input_size, scales),
            transforms.GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))
            ])







if __name__ == '__main__':
    model = Model(101, 3, 'iframe',
                  base_model='resnet50', is_shift=False, shift_div=8) 
    pdb.set_trace()
