import _clip.clip as clip
import numpy as np
import torch
import torch.nn as nn
from _clip.model import ModifiedResNet, VisualTransformer
from src.models import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False)

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)

    def reset_parameters(self):
        nn.init.normal_(self.model.token_embedding.weight, std=0.02)
        nn.init.normal_(self.model.positional_embedding, std=0.01)
        self.model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.model.visual, ModifiedResNet):
            raise NotImplementedError('Resetting parameters of the ModifiedResNet is not supported yet')
            if self.model.visual.attnpool is not None:
                std = self.model.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.model.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.model.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.model.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.model.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.model.visual.layer1, self.model.visual.layer2, self.model.visual.layer3, self.model.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
        elif isinstance(self.model.visual, VisualTransformer):
            width = self.model.visual.width
            scale = width ** -0.5

            self.model.visual.conv1.reset_parameters()
            self.model.visual.class_embedding.data.normal_(std=scale)
            self.model.visual.positional_embedding.data.normal_(std=scale * width)
            self.model.visual.ln_pre.reset_parameters()
            self.model.visual.ln_post.reset_parameters()
            self.model.visual.proj.data.normal_(std=scale)

            proj_std = (self.model.visual.width ** -0.5) * ((2 * self.model.visual.layers) ** -0.5)
            attn_std = self.model.visual.width ** -0.5
            fc_std = (2 * self.model.visual.width) ** -0.5
            for block in self.model.visual.transformer.resblocks:
                block.attn._reset_parameters()
                block.mlp.c_fc.reset_parameters()
                block.mlp.c_proj.reset_parameters()
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
                block.ln_1.reset_parameters()
                block.ln_2.reset_parameters()

        if hasattr(self.model, 'transformer'):
            raise NotImplementedError('Resetting parameters of the transformer is not supported yet')
            proj_std = (self.model.transformer.width ** -0.5) * ((2 * self.model.transformer.layers) ** -0.5)
            attn_std = self.model.transformer.width ** -0.5
            fc_std = (2 * self.model.transformer.width) ** -0.5
            for block in self.model.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            if self.model.text_projection is not None:
                nn.init.normal_(self.model.text_projection, std=self.model.transformer.width ** -0.5)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            self.enc_out = self.image_encoder(inputs)
            outputs = self.classification_head(self.enc_out)
            return outputs
        else:
            return self.classification_head(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        return utils.torch_load(filename)

    def reset_parameters(self):
        if self.image_encoder is not None:
            self.image_encoder.reset_parameters()
        self.classification_head.reset_parameters()

    def vc_reg(self, alpha: float, beta: float) -> torch.Tensor:
        cov_mat = torch.cov(self.enc_out.T)
        d = cov_mat.shape[0]

        # v_loss
        diag = cov_mat.diag() + 1e-7
        v_loss = torch.maximum(torch.zeros_like(diag), 1 - torch.sqrt(diag)).mean()

        # c_loss
        off_diag_mat = 1 - torch.eye(d, device=cov_mat.device, dtype=cov_mat.dtype)
        c_loss = ((cov_mat * off_diag_mat) ** 2).sum() / (d * (d - 1))
        return alpha * v_loss + beta * c_loss
