
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class InnerProd(nn.Module):
    def __init__(self, fc_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)).view(
            B, 1, *sound_size[2:]
        )
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, C)
        z = (feat_img * self.scale).view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI * WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img * self.scale, feat_sound).view(
            B, HI, WI, HS, WS
        )
        z = z + self.bias
        return z

def init_weights(net):
    classname = net.__class__.__name__
    if classname.find("Conv") != -1:
        net.weight.data.normal_(0.0, 0.001)
    elif classname.find("BatchNorm") != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        net.weight.data.normal_(0.0, 0.0001)

# Hook function to store the outputs
outputs = {}

def hook_fn(module, input, output):
    outputs['downconv_output'] = output


class CondUNetBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_input_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        inner_output_nc=None,
        noskip=False,
        cond_nc=None,
    ):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.noskip = noskip
        self.cond_nc = cond_nc
        self.submodule = submodule

        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            assert cond_nc > 0
            inner_output_nc = inner_input_nc + cond_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        self.downnorm = nn.BatchNorm2d(inner_input_nc)
        self.uprelu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        if outermost:
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1
            )

        elif innermost:
            self.downrelu = nn.LeakyReLU(0.2, True)
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            # Register the hook on the innermost downconv layer
            self.downconv.register_forward_hook(hook_fn)

            self.upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            self.upnorm = nn.BatchNorm2d(outer_nc)

        else:
            self.downrelu = nn.LeakyReLU(0.2, True)
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            self.upnorm = nn.BatchNorm2d(outer_nc)

    def forward(self, x, cond):
        if self.outermost:
            x_ = self.downconv(x)
            x_ = self.submodule(x_, cond) # 我觉得最外层不用submodule呀 我试试看这样直接跑呢？，这一行是我后来注释掉的。
            x_ = self.upconv(self.upsample(self.uprelu(x_)))

        elif self.innermost:
            x_ = self.downconv(self.downrelu(x))

            B, _, H, W = x_.size()
            #cond_ = cond.unsqueeze(-1).unsqueeze(-1) * torch.ones(
                #(B, self.cond_nc, H, W), device=x_.device
            #) #
            cond_ = cond.unsqueeze(-1).unsqueeze(-1) * torch.ones(
                (B, 512, H, W), device=x_.device
            ) #
            #print('=======cond shape check========',x_.shape,cond_.shape)
            x_ = torch.concat((x_, cond_), 1)
            x_ = self.upnorm(self.upconv(self.upsample(self.uprelu(x_))))

        else:
            x_ = self.downnorm(self.downconv(self.downrelu(x)))
            x_ = self.submodule(x_, cond)  #这里也不用呀，就最内层需要呀，这一行是我后来注释掉的。
            x_ = self.upnorm(self.upconv(self.upsample(self.uprelu(x_))))

        if self.outermost or self.noskip:
            return x_
        else:
            return torch.cat([x, x_], 1)



class CondUNet(nn.Module):
    """A UNet model."""

    def __init__(
        self,
        in_dim=1,
        out_dim=64,
        cond_dim=32,
        num_downs=5,
        ngf=64,
        use_dropout=False,
    ):
        super().__init__()

        # Construct the U-Net structure
        unet_block = CondUNetBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            innermost=True,
            cond_nc=cond_dim,
        )
        for _ in range(num_downs - 5):
            unet_block = CondUNetBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block
            )
        unet_block = CondUNetBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block
        )
        unet_block = CondUNetBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block
        )
        unet_block = CondUNetBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block
        )
        unet_block = CondUNetBlock(
            out_dim,
            ngf,
            input_nc=in_dim,
            submodule=unet_block,
            outermost=True,
        )

        self.bn0 = nn.BatchNorm2d(in_dim)
        self.unet_block = unet_block

    def forward(self, x, cond):
        x = self.bn0(x)
        #print('==   cond shape check==:',x.shape, cond.shape)
        x = self.unet_block(x, cond) # torch.Size([32, 1, 256, 256]) torch.Size([32, 32])
        return x


#def distillation_loss(student_output, teacher_output):
    #return F.mse_loss(student_output, teacher_output)

def distillation_loss(student_output, teacher_output):
    # 归一化每个样本的特征
    student_norm = F.normalize(student_output, p=2, dim=-1)  # [batch_size, feature_dim]
    teacher_norm = F.normalize(teacher_output, p=2, dim=-1)  # [batch_size, feature_dim]

    # 计算每个样本的余弦相似度
    cosine_similarity = torch.sum(student_norm * teacher_norm, dim=-1)  # [batch_size]

    # 计算 batch 平均的余弦损失
    loss = 1 - cosine_similarity.mean()  

    return loss



class CLIPSep_ML(torch.nn.Module):
    """Separation model based on the CLIP model."""

    def __init__(
        self,
        n_mix,
        layers=7,
        channels=32,
        use_log_freq=True,
        use_weighted_loss=True,
        use_binary_mask=True,
    ):
        super().__init__()
        self.n_mix = n_mix
        self.use_log_freq = use_log_freq
        self.use_weighted_loss = use_weighted_loss
        self.use_binary_mask = use_binary_mask

        # Create the neural net
        self.sound_net = CondUNet(
            in_dim=1, out_dim=32, cond_dim=512, num_downs=layers
        ) # here I change output dim to 32 to match my
        #self.frame_net_mid = nn.Linear(512, channels)
        self.frame_net_mid = nn.Linear(512, 512) # here change
        self.frame_net_late = nn.Linear(512, channels)
        self.synth_net = InnerProd(fc_dim=channels)
        for module in self.sound_net.modules():
            if isinstance(module, CondUNetBlock) and module.innermost:
                module.downconv.register_forward_hook(hook_fn)
        # Initialize the weights
        self.sound_net.apply(init_weights)
        self.frame_net_mid.apply(init_weights)
        self.frame_net_late.apply(init_weights)
        self.synth_net.apply(init_weights)


    def forward(self, batch, audio_embed_ref=None, drop_closest=None):
        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]
        img_emb = batch['frames'] #
        #audio_mix = batch["audio_mix"]
        audio_mix_rep = batch["audio_mix_rep"]


    

        # Pass through the frame net with the precomputed frame features (img_emb) -> Bx1xC
        feat_frames_mid = [
            torch.sigmoid(self.frame_net_mid(img_emb[n])) for n in range(N)
        ]
        feat_frames_late = [
            torch.sigmoid(self.frame_net_late(img_emb[n])) for n in range(N)
        ]


        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Fuse visual features and predict masks
        pred_masks = []
        for n in range(N):
            # Pass through the sound net -> BxCxHxW
            #print('==================:',feat_frames_mid[n].shape) # 32*32
            feat_sound = self.sound_net(log_mag_mix, feat_frames_mid[n])
            
            pred_masks = [
            self.synth_net(feat_frames_late[n], feat_sound) for n in range(N)
        ] # 这里是后处理
            #pred_masks.append(feat_sound)
        student_clap_emb = outputs['downconv_output']# 提取特征！！！！！！
        student_clap_emb = student_clap_emb.mean(dim=(2, 3))
        audio_mix_rep = audio_mix_rep.squeeze(1)
        loss_dist = distillation_loss(student_clap_emb, audio_mix_rep)
        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(mask) for mask in pred_masks] # 
        
        # Compute the binary cross-entropy loss
        loss_cons = torch.mean(
            torch.stack(
                [
                    F.binary_cross_entropy(pred_masks[n], gt_masks[n], weight)
                    for n in range(N)
                ]
            )
        )
        loss = loss_dist+loss_cons
        return (
            loss,
            {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "mag_mix": mag_mix,
                "mags": mags,
                "weight": weight,
                "loss_distill": loss_dist,
                "loss_cons": loss_cons
            },
        )