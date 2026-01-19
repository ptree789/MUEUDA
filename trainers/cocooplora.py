import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import copy
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits

# 定义lora线性层，替换原始的nn.liner
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=12, alpha=6, dropout_p: float = 0.0, bias=True,):  # 添加 dtype 参数
        super(LoRALinear, self).__init__()
        self.base_layer = copy.deepcopy(base_layer)

        # 打印 base_layer 的类型和属性
        #print(f"base_layer 类型: {type(self.base_layer)}")
        #print(f"base_layer 属性: {dir(self.base_layer)}")  # 列出所有属性和方法
        #print(f"base_layer 是否有 weight 属性: {hasattr(self.base_layer, 'weight')}")

        #if hasattr(self.base_layer, 'weight'):
            #print(f"base_layer.weight: {self.base_layer.weight}")  # 如果有 weight 属性，打印它

        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)
        #self.dtype = nn.Linear.weight.dtype  # 保存 dtype 参数
        #base_layer = nn.Linear
        #dropout_p: float = 0.0     # lora dropout
        self.lora_A = nn.Parameter(torch.empty((r, base_layer.in_features), dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.empty((base_layer.out_features, r), dtype=base_layer.weight.dtype))

        #初始化
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)
        # 冻结原来的层的参数
        for param in self.base_layer.parameters():
            param.requires_grad = False
        #self.linear = nn.Linear(in_features, out_features, bias=bias).to(self.dtype)  # 将 linear 层转换为指定 dtype

        #self.dropout = nn.Dropout(dropout_p)

        # LoRA 的 A 和 B 矩阵，使用指定 dtype
        #self.A = nn.Parameter(torch.randn(out_features, r, dtype=dtype))
        #self.B = nn.Parameter(torch.randn(r, in_features, dtype=dtype))

        #self.A = nn.Parameter(torch.empty((r, in_features), dtype=base_layer.weight.dtype))
        #self.B = nn.Parameter(torch.empty((out_features, r), dtype=base_layer.weight.dtype))

        #self.scaling = self.alpha / self.r

        #nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        #nn.init.zeros_(self.B)


    @property
    def weight(self):
    # 返回等效的 weight,确保 dtype 一致
        #print("self.lora_A.shape:", self.lora_A.shape)
        #print("self.lora_B.shape:", self.lora_B.shape)
        return self.base_layer.weight + (self.lora_B @ self.lora_A) * (self.alpha / self.r)
    


    @property
    def bias(self):
        # 返回 bias,直接使用 linear 层的 bias
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling = float(self.alpha) / float(self.r)     # lora 缩放系数
        lora_adjustment = F.linear(self.dropout(x), self.lora_A)
        lora_adjustment = F.linear(lora_adjustment, self.lora_B)
        #print(f"lora_adjustment 类型: {type(lora_adjustment)}")
        #print(f"lora_adjustment 属性: {dir(lora_adjustment)}")  # 列出所有属性和方法
        #print(f"lora_adjustment 是否有 weight 属性: {hasattr(lora_adjustment, 'weight')}")

        return self.base_layer(x) + lora_adjustment * scaling

    '''def forward(self, x):
        # 确保所有张量在相同的 dtype 下执行
        x = x.to(self.linear.weight.dtype)
        print(f"x.shape: {x.shape}")
        print(f"self.A.shape: {self.A.shape}")
        print(f"self.B.shape: {self.B.shape}")

        return self.linear(x) + self.scaling * (self.A @ (self.B @ x.T)).T'''
    
    '''def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.scaling = self.alpha / self.r     # lora 缩放系数
        #x = x.to(self.dtype)
        lora_adjustment = F.linear(self.dropout(x), self.A)
        lora_adjustment = F.linear(lora_adjustment, self.B)
        return self.linear(x) + lora_adjustment * self.scaling'''
    

'''def replace_linear_with_lora(model, r=8, alpha=1.0, dtype=torch.float16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module.in_features, module.out_features, r=r, alpha=alpha))
        else:
            replace_linear_with_lora(module, r=r, alpha=alpha)'''

def replace_linear_with_lora(
    module: nn.Module,
    r: int = 12,
    alpha: int = 6,
    dropout_p: float = 0.0,
    embed_requires_grad: bool = False,      # embedding 层是否训练
    norm_requires_grad: bool = False,       # norm 层是否训练
    head_requires_grad: bool = False,       # lm_head 层是否训练（Causal LM才有）
    #test_mode: bool = False,                # 测试模式，用于控制 lora_B 是否为全零
):
    """
    找到 module 中所有线性层并递归替换
    """
    for name, child in module.named_children():
        # 先处理额外的层，lm_head 也是 linear，所以先处理
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            requires_grad = embed_requires_grad if 'embed' in name \
                            else norm_requires_grad if 'norm' in name \
                            else head_requires_grad
            for param in child.parameters():
                param.requires_grad = requires_grad
        # 替换所有线性层，QLoRA 做法
        elif isinstance(child, nn.Linear):
            lora_linear = LoRALinear(child, r=r, alpha=alpha, dropout_p=dropout_p)
            setattr(module, name, lora_linear)
        # 递归向下替换
        else:
            replace_linear_with_lora(
                child, r, alpha, dropout_p,
                embed_requires_grad, norm_requires_grad, head_requires_grad,
                #test_mode=test_mode
            )


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16set
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Applying LoRA to the image encoder")
        replace_linear_with_lora(self.model.image_encoder, r=12, alpha=6)

        print("Network structure after applying LoRA:")
        print(self.model)

        print("Turning off gradients in text encoder")
        #name_to_update = ["prompt_learner", "image_encoder"]
        #name_to_update = "prompt_learner"

        
        #for name, param in self.model.named_parameters():
            #if name_to_update not in name:
                #param.requires_grad_(False)


        # 先冻结 text_encoder 的所有参数
        for name, param in self.model.text_encoder.named_parameters():
            param.requires_grad = False

        # 更新视觉编码器和prompt_learner
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad = True
            elif "lora_A" in name:
                param.requires_grad = True
            elif "lora_B" in name:
                param.requires_grad = True
            #elif "image_encoder" in name:
                #param.requires_grad = True
            else:
                param.requires_grad = False


        # 保留文本编码器冻结，只更新视觉编码器和prompt_learner
        #for name, param in self.model.named_parameters():
            #if "text_encoder" not in name:  # 仅冻结文本编码器
                #param.requires_grad_(True)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        #self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)

        # NOTE: 对视觉编码器和prompt_learner使用优化器
        self.optim = build_optimizer(self.model, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
