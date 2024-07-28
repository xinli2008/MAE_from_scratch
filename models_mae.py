from functools import partial
from utils.position_embed import get_2d_sincos_pos_embed
import torch
import torch.nn as nn
from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer backbone
    """
    def __init__(self, img_size = 224, patch_size = 16, in_channels = 3, embedding_dim = 1024, 
                 depth = 24, num_heads = 16, decoder_embedding_dim = 512, decoder_depth = 8,
                 decoder_num_head = 16, mlp_ratio = 4, norm_layer = nn.LayerNorm, norm_pix_loss = False):
        super().__init__()

        # --------------------------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embedding = PatchEmbed(img_size = img_size, patch_size = patch_size, in_chans = in_channels, embed_dim = embedding_dim)
        self.in_channels = in_channels
        self.num_patches = self.patch_embedding.num_patches

        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_dim), requires_grad = False)

        self.blocks = nn.ModuleList([
            Block(embedding_dim, num_heads, mlp_ratio, qkv_bias = True, norm_layer = norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embedding_dim)
        # --------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embedding_dim, decoder_embedding_dim, bias = True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embedding_dim))
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embedding_dim), requires_grad = False)

        self.decoder_block = nn.ModuleList([
            Block(decoder_embedding_dim, decoder_num_head, mlp_ratio, qkv_bias = True, norm_layer = norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embedding_dim)
        self.decoder_pred = nn.Linear(decoder_embedding_dim, patch_size * patch_size * in_channels, bias = True)
        # --------------------------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        """
        initialize some parameters and freeze position embedding by sin-cos embedding
        """
        pos_embed = get_2d_sincos_pos_embed(self.position_embed.shape[-1], int(self.patch_embedding.num_patches ** 0.5), cls_token = True)
        self.position_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_emb = get_2d_sincos_pos_embed(self.decoder_pos_emb.shape[-1], int(self.patch_embedding.num_patches ** 0.5), cls_token = True)
        self.decoder_pos_emb.data.copy_(torch.from_numpy(decoder_pos_emb).float().unsqueeze(0))

        # initialize patch embedding like nn.linear (instead of nn.Conv2d)
        w = self.patch_embedding.proj.weight.data
        torch.nn.init.xavier_normal_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_tokens, std = 0.02)
        torch.nn.init.normal_(self.mask_token, std = 0.02)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        else:
            pass
    
    def patchify(self, imgs):
        """
        imgs:  torch.Tensor, [b, c, h, w]
        x:     torch.Tensor, [b, L, patch_size**2**c] where L = patch_count * patch_count
        """
        patch_size = self.patch_embedding.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size ==0 ,\
            "width should be equal to height and should be divisible by patch_size"
        
        patch_count = imgs.shape[2] // patch_size
        x = torch.reshape(imgs, shape = (imgs.shape[0], self.in_channels, patch_count, patch_size, patch_count, patch_size))
        x = torch.permute(x, dims = (0, 2, 4, 1, 3, 5)).contiguous()
        x = torch.reshape(x, shape = (x.shape[0], patch_count * patch_count, -1))

        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffing
        Args:
            x:              torch.Tensor, [b, L, c]
            mask_ratio:     float, e.g. 0.75
        Return:
            x_masked:       torch.Tensor,  [b, L*mask_ratio, c]
            mask:           torch.Tensor,  [1, L]
            ids_restore:    torch.Tensor,  [1, L]  
        """
        b, L, c = x.shape
        keep_length = int(L * (1 - mask_ratio))

        noise = torch.rand(b, L, device = x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim = -1, descending = False)
        ids_restore = torch.argsort(ids_shuffle, dim = -1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :keep_length]
        x_masked = torch.gather(x, dim = 1, index = ids_keep.unsqueeze(-1).repeat(1, 1, c))

        # generate the binary mask: 0 is keep while 1 is remove
        mask = torch.ones([b, L], device = x.device)
        mask[:, :keep_length] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim = 1, index = ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Perform masked autoencoder's encoder
        Args:
            x:          torch.Tensor, [b, c, h, w]
            mask_ratio: float, (0-1), e.g. 0.5
        Return:
            x:          torch.Tensor, [b, ]
            mask:
            ids_restore:
        """
        # patch embedding
        x = self.patch_embedding(x)    # [b, L, c]

        # add position embedding
        x = x + self.position_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_tokens + self.position_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim = 1)

        # apply transformer encoder block
        for blk in self.blocks:
            x = blk(x)

        # norm layer
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Perform masked autoencoder's decoder
        Args:
            x:
            ids_restore:
        Return:
            x:
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim = 1)    # no cls token
        x_ = torch.gather(x_, dim = 1, index = ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim = 1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_emb

        # apply Transformer blocks
        for blk in self.decoder_block:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)   # [b, L, c]

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        Perform MAE forward loss
        Args:
            imgs: torch.Tensor, [b, c, h, w]
            pred: torch.Tensor, [b, L, patch_count * patch_count * 3]
            mask: torch.Tensor, [b, L]
        """
        target = self.patchify(imgs)  # [b, L, c] -> [b, L, patch_count * patch_count * 3]
        if self.norm_pix_loss:
            mean = target.mean(dim = -1, keepdim = True)
            var  = target.var(dim = -1, keepdim = True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        # --------------------------------------------------------------------------------------------
        # Introduction to MSE-loss and MAE-loss
        # MAE-loss和MSE-loss是两种常见的回归损失函数。它们用于衡量预测值和真实值之间的差异, 但计算方法和对误差的敏感程度不同。
        # 1. MSE-loss的计算公式为: loss = (pred - target) ** 2
        #    MSE-loss的特点是: MSE对于较大的误差进行更高的惩罚, 因此在某些情况下, 这有助于减少大的预测错误。
        # 2. MAE-loss的计算公式为: loss = torch.abs(pred - target)
        #    MAE-loss的特点是: 对离散值不敏感, 因为它对误差的惩罚是线性的。
        # --------------------------------------------------------------------------------------------
        loss = (pred - target) ** 2
        loss = torch.mean(loss, dim = -1, keepdim = True)

        # compute the loss only on the masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss
 
    def forward(self, imgs, mask_ratio = 0.75):
        # encoder
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # decoder
        pred = self.forward_decoder(latent, ids_restore)
        # loss
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size = 16, embedding_dim = 768, depth = 12, num_heads = 12,
        decoder_embedding_dim = 512, decoder_depth = 8, decoder_num_head = 16,
        mlp_ratio = 4, norm_layer = partial(nn.LayerNorm, eps = 1e-6), **kwargs
    )
    return model

def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size = 16, embedding_dim = 1024, depth = 24, num_heads = 16,
        decoder_embedding_dim = 512, decoder_depth = 8, decoder_num_head = 16,
        mlp_ratio = 4, norm_layer = partial(nn.LayerNorm, eps = 1e-6), **kwargs
    )
    return model

def mae_vit_huge_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size = 14, embedding_dim = 1280, depth = 32, num_heads = 16,
        decoder_embedding_dim = 512, decoder_depth = 8, decoder_num_head = 16,
        mlp_ratio = 4, norm_layer = partial(nn.LayerNorm, eps = 1e-6), **kwargs
    )
    return model

if __name__ == "__main__":
    input_tensor = torch.randn([1, 3, 224, 224])
    mask_ratio = 0.5

    model = MaskedAutoencoderViT()    
    output = model(input_tensor, mask_ratio)
    