import os.path as osp
from collections import OrderedDict
import math
import os
from PIL import Image


import torch
import torch.nn as nn
import copy
import pickle
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from trainers.teacher import EMATeacher

from torchvision import transforms
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

    device = torch.device(f'cuda:{cfg.TRAINER.DEVICES[0]}')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  

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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class TeacherTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text, tokenized_prompts):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        #print(type(x))
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

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(f'cuda:{cfg.TRAINER.DEVICES[0]}')
        print('shujksuhjkskugh', self.device)

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(self.device)
            print(f"prompt device: {prompt.device}")
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

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # (n_cls, n_tkn)
        print(f"tokenized_prompts device: {tokenized_prompts.device}")
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
        ).to(self.device)

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
            pts_i = self.construct_prompts(ctx_i, prefix, suffix).to(self.device)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts).to(self.device)
        
        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.device = torch.device(f'cuda:{cfg.TRAINER.DEVICES[0]}')
        print('shujksuhjkskugh', self.device)
        clip_model = clip_model.to(self.device)

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


    def forward(self, image, label=None, dis = None):

        if dis is not None:
            final_results = LCMUDA.uncertainty_measure(self, image)
            return  final_results

        tokenized_prompts = self.tokenized_prompts.to(self.device)
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        #image_features = self.image_encoder(image.to(dtype=self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if label is not None and self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        
        return logits
    
def print_model_device_info(model):
    for name, param in model.named_parameters():
        print(f"{name} is on device: {param.device}")

class TeacherCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.device = torch.device(f'cuda:{cfg.TRAINER.DEVICES[0]}')
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        n_ctx = cfg.TRAINER.LCMUDA.N_CTX
        ctx_init = cfg.TRAINER.LCMUDA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype).to(self.device) 
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(self.device) 
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # Save the other parts of the model
        self.image_encoder = clip_model.visual.to(self.device)
        self.text_encoder = clip_model.encode_text
        self.logit_scale = clip_model.logit_scale.to(self.device)
        self.dtype = clip_model.dtype

        # Generate tokenized_prompts
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.prompts = prompts
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)

        print(type(self.tokenized_prompts))
        # Printing equipment information
        print(f"self.tokenized_prompts device: {self.tokenized_prompts.device}")

    def forward(self, image, label=None):
        # Obtain the text and image features of the CLIP model
        #tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        image = image.to(self.device)
        #print(f"image device: {image.device}")
        #print(f"self.tokenized_prompts device: {self.tokenized_prompts.device}")
        #print_model_device_info(self.image_encoder)
        #print_model_device_info(self.text_encoder)

        # Calculate image features
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        tokenized_prompts = self.tokenized_prompts.to(self.device)
        # Obtain text features
        text_features = self.text_encoder(tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate the matching score of the image and text features
        logits = logit_scale * image_features @ text_features.t()

        # Apply softmax to convert logits to probabilities
        self.probs = F.softmax(logits, dim=-1)
        #print('targetprobs:', self.probs)
        
        if label is not None and self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        # If there is no label, a pseudo-label will be returned
        pesudo_labels = torch.argmax(logits, dim=1)  
        pesudo_probs = self.probs.gather(1, pesudo_labels.unsqueeze(1)).squeeze(1)  
        return pesudo_labels, self.probs, pesudo_probs
        #return logits
    

# Define the lora linear layer to replace the original nn.liner
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=8, alpha=4, dropout_p: float = 0.0, bias=True,):  # 添加 dtype 参数
        super(LoRALinear, self).__init__()
        self.base_layer = copy.deepcopy(base_layer)
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)
        #self.dtype = nn.Linear.weight.dtype 
        #base_layer = nn.Linear
        #dropout_p: float = 0.0     # lora dropout
        self.lora_A = nn.Parameter(torch.empty((r, base_layer.in_features), dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.empty((base_layer.out_features, r), dtype=base_layer.weight.dtype))

        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)
        # Freeze the parameters of the original layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self):
    # Return the equivalent weight and ensure that the dtype is consistent
        return self.base_layer.weight + (self.lora_B @ self.lora_A) * (self.alpha / self.r)

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling = float(self.alpha) / float(self.r)     # lora scaling factor
        lora_adjustment = F.linear(self.dropout(x), self.lora_A)
        lora_adjustment = F.linear(lora_adjustment, self.lora_B)

        return self.base_layer(x) + lora_adjustment * scaling

def replace_linear_with_lora(
    module: nn.Module,
    r: int = 8,
    alpha: int = 4,
    dropout_p: float = 0.0,
    embed_requires_grad: bool = False,      # Whether the embedding layer is trained
    norm_requires_grad: bool = False,       # Whether the norm layer is trained
    head_requires_grad: bool = False,       # Whether the lm_head layer is trained
    #test_mode: bool = False,                # Test mode, used to control whether lora_B is all zeros
):
    """
    Find all the linear layers in the module and replace them recursively
    """
    for name, child in module.named_children():
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            requires_grad = embed_requires_grad if 'embed' in name \
                            else norm_requires_grad if 'norm' in name \
                            else head_requires_grad
            for param in child.parameters():
                param.requires_grad = requires_grad
        # Replace all linear layers, QLoRA approach
        elif isinstance(child, nn.Linear):
            lora_linear = LoRALinear(child, r=r, alpha=alpha, dropout_p=dropout_p)
            setattr(module, name, lora_linear)
        else:
            replace_linear_with_lora(
                child, r, alpha, dropout_p,
                embed_requires_grad, norm_requires_grad, head_requires_grad,
                #test_mode=test_mode
            )

@TRAINER_REGISTRY.register()
class LCMUDA(TrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.LCMUDA.PREC in ["fp16", "fp32", "amp"]

    def __init__(self, cfg):
        super().__init__(cfg)
        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        if n_domain <= 0:
            n_domain = self.num_source_domains
        self.split_batch = batch_size // n_domain
        self.n_domain = n_domain
        self.dtype = torch.float16

        # Initialize the feature prototype dictionary to store the feature prototypes of each category
        self.feature_prototypes = {}
        self.features_by_class = {}
        
        # Specified GPU device
        if hasattr(cfg.TRAINER, 'DEVICES'):
            # If it is a single card, directly specify the device
            if isinstance(cfg.TRAINER.DEVICES, list) and len(cfg.TRAINER.DEVICES) == 1 :

                device_id = cfg.TRAINER.DEVICES[0]
                torch.cuda.set_device(device_id)
                self.device = torch.device(f'cuda:{device_id}')
            elif isinstance(cfg.TRAINER.DEVICES, list) and len(cfg.TRAINER.DEVICES) > 1 :
                devices = cfg.TRAINER.DEVICES
                self.device = torch.device(f'cuda:{devices[0]}')  
            else:
                raise ValueError("Unsupported format for TRAINER.DEVICES, expected int or list.")
        else:
            self.device = torch.device('cuda:0')

    def build_model(self):
        cfg = self.cfg

        classnames = self.dm.dataset.classnames


        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        #print_model_device_info(clip_model)
        
        if cfg.TRAINER.LCMUDA.PREC == "fp32" or cfg.TRAINER.LCMUDA.PREC == "amp":
            # CLIP's default precision is fp16set
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.teacher_model = TeacherCLIP(cfg, classnames, clip_model)

        self.device = torch.device("cuda:2")
        self.model.to(self.device)
        self.teacher_model.to(self.device)
#############################################################################
        # Specify the source domain and data path
        source_domain_names = cfg.DATASET.SOURCE_DOMAINS 
        DATASET_NAME = cfg.DATASET.NAME
        print('DATASET_NAME:' , DATASET_NAME)
        data_file_name = ''
        
        # Change the first letter of each domain name in source_domains to uppercase only when DATASET_NAME is OfficeHome
        if DATASET_NAME == 'OfficeHome':
            source_domain_names = [domain.title() for domain in source_domain_names]
            data_file_name = 'officehome'
            for key, value in self.dm.lab2cname.items():
                if value.lower() == 'toothbrush':
                    self.dm.lab2cname[key] = 'ToothBrush'
                elif value.lower() == 'tv':
                    self.dm.lab2cname[key] = 'TV'
                else:
                    self.dm.lab2cname[key] = value.title()  
            self.data_lab2cname = self.dm.lab2cname
        if DATASET_NAME == 'OF' :
            data_file_name = DATASET_NAME
            self.data_lab2cname = {0: 'Alarm_Clock', 1: 'Backpack', 2: 'Batteries', 3: 'Bed', 4: 'Bike', 5: 'Bottle', 6: 'Bucket',\
                                7: 'Calculator', 8: 'Calendar', 9: 'Candles', 10: 'Chair', 11: 'Clipboards', 12: 'Computer',\
                                13: 'Couch', 14: 'Curtains'}
        elif DATASET_NAME == 'OF31':
            data_file_name = DATASET_NAME
            #self.data_lab2cname = {0: 'back_pack', 1: 'bike', 2: 'bike_helmet',  3: 'bookcase', 4: 'bottle', 5: 'calculator',  6: 'desk_chair', \
                                    #7: 'desk_lamp', 8: 'desktop_computer', 9: 'file_cabinet', 10: 'headphones', 11: 'keyboard', 12: 'laptop_computer',  13: 'letter_tray', \
                                    #14: 'mobile_phone', 15: 'monitor', 16: 'mouse', 17: 'mug',  18: 'paper_notebook', 19: 'pen'}
            
            self.data_lab2cname = {0: 'back_pack', 1: 'bike', 2: 'bike_helmet',  3: 'bookcase', 4: 'bottle', 5: 'calculator',  6: 'desk_chair', \
                                    7: 'desk_lamp', 8: 'desktop_computer', 9: 'file_cabinet', 10: 'headphones', 11: 'keyboard', 12: 'laptop_computer',  13: 'letter_tray', \
                                    14: 'mobile_phone', 15: 'monitor', 16: 'mouse', 17: 'mug',  18: 'paper_notebook', 19: 'pen', 20: 'projector'}            
            
        elif DATASET_NAME == 'DOMAINNET':
            data_file_name = DATASET_NAME
            self.data_lab2cname = self.dm.lab2cname
            '''self.data_lab2cname = {0: 'aircraft_carrier', 1: 'airplane', 2: 'alarm_clock', 3: 'ambulance', 4: 'angel', 5: 'animal_migration',\
                    6: 'ant', 7: 'anvil', 8: 'apple', 9: 'arm', 10: 'asparagus', 11: 'axe', 12: 'backpack', 13: 'banana', 14: 'bandage',\
                    15: 'barn', 16: 'baseball', 17: 'baseball_bat', 18: 'basket', 19: 'basketball', 20: 'bat', 21: 'bathtub', 22: 'beach',\
                    23: 'bear', 24: 'beard', 25: 'bed', 26: 'bee', 27: 'belt', 28: 'bench', 29: 'bicycle', 30: 'binoculars', 31: 'bird', 32: 'birthday_cake',\
                    33: 'blackberry', 34: 'blueberry', 35: 'book', 36: 'boomerang', 37: 'bottlecap', 38: 'bowtie', 39: 'bracelet', 40: 'brain', 41: 'bread',\
                    42: 'bridge', 43: 'broccoli', 44: 'broom', 45: 'bucket', 46: 'bulldozer', 47: 'bus', 48: 'bush', 49: 'butterfly', 50: 'cactus', 51: 'cake',\
                    52: 'calculator', 53: 'calendar', 54: 'camel', 55: 'camera', 56: 'camouflage', 57: 'campfire', 58: 'candle', 59: 'cannon', 60: 'canoe',\
                    61: 'car', 62: 'carrot', 63: 'castle', 64: 'cat', 65: 'ceiling_fan', 66: 'cell_phone', 67: 'cello', 68: 'chair', 69: 'chandelier', 70: 'church',\
                    71: 'circle', 72: 'clarinet', 73: 'clock', 74: 'cloud', 75: 'coffee_cup', 76: 'compass', 77: 'computer', 78: 'cookie', 79: 'cooler',\
                    80: 'couch', 81: 'cow', 82: 'crab', 83: 'crayon', 84: 'crocodile', 85: 'crown', 86: 'cruise_ship', 87: 'cup', 88: 'diamond', 89: 'dishwasher',\
                    90: 'diving_board', 91: 'dog', 92: 'dolphin', 93: 'donut', 94: 'door', 95: 'dragon', 96: 'dresser', 97: 'drill', 98: 'drums', 99: 'duck',\
                    100: 'roller_coaster', 101: 'rollerskates', 102: 'sailboat', 103: 'sandwich', 104: 'saw', 105: 'saxophone', 106: 'school_bus',\
                    107: 'scissors', 108: 'scorpion', 109: 'screwdriver', 110: 'sea_turtle', 111: 'see_saw', 112: 'shark', 113: 'sheep', 114: 'shoe', 115: 'shorts',\
                    116: 'shovel', 117: 'sink', 118: 'skateboard', 119: 'skull', 120: 'skyscraper', 121: 'sleeping_bag', 122: 'smiley_face', 123: 'snail',\
                    124: 'snake', 125: 'snorkel', 126: 'snowflake', 127: 'snowman', 128: 'soccer_ball', 129: 'sock', 130: 'speedboat', 131: 'spider',\
                    132: 'spoon', 133: 'spreadsheet', 134: 'square', 135: 'squiggle', 136: 'squirrel', 137: 'stairs', 138: 'star', 139: 'steak', 140: 'stereo',\
                    141: 'stethoscope', 142: 'stitches', 143: 'stop_sign', 144: 'stove', 145: 'strawberry', 146: 'streetlight', 147: 'string_bean', 148: 'submarine',\
                    149: 'suitcase',150: 'sun', 151: 'swan', 152: 'sweater', 153: 'swing_set', 154: 'sword', 155: 'syringe', 156: 't-shirt', 157: 'table',\
                    158: 'teapot', 159: 'teddy-bear', 160: 'telephone', 161: 'television', 162: 'tennis_racquet', 163: 'tent', 164: 'The_Eiffel_Tower', 165: 'The_Great_Wall_of_China',
                    166: 'The_Mona_Lisa', 167: 'tiger', 168: 'toaster',  169: 'toe', 170: 'toilet', 171: 'tooth', 172: 'toothbrush', 173: 'toothpaste', 174: 'tornado', 175: 'tractor',\
                    176: 'traffic_light', 177: 'train', 178: 'tree', 179: 'triangle', 180: 'trombone', 181: 'truck', 182: 'trumpet', 183: 'umbrella', 184: 'underwear',\
                    185: 'van', 186: 'vase', 187: 'violin', 188: 'washing_machine', 189: 'watermelon', 190: 'waterslide', 191: 'whale', 192: 'wheel', 193: 'windmill',\
                    194: 'wine_bottle', 195: 'wine_glass', 196: 'wristwatch', 197: 'yoga', 198: 'zebra', 199: 'zigzag'}'''
            
        else: 
            data_file_name = DATASET_NAME

        data_labels = list(self.data_lab2cname.keys())
        date_class_names = list(self.dm.lab2cname.values())

        data_file_name = data_file_name

        #Replace it with your path
        data_path = f'/xxx/xxx/UniMDA/DATA/{data_file_name}'

        # Call the function for calculating the feature prototype
        self.init_batch_size = 16
        cache_path = f'/xxx/xxx/UniMDA/feature_prototypes/{DATASET_NAME}/{DATASET_NAME}_feature_prototypes_cache.pkl'  
        self.init_class_prototypes = self.init_feature_prototypes(source_domain_names, data_path, list(self.data_lab2cname.keys()), self.init_batch_size, cache_path)

#############################################################################
        print("Applying LoRA to the image encoder")
        replace_linear_with_lora(self.model.image_encoder, r=8, alpha=4)

        # Create a visual teacher model
        print("Creating visual teacher model with EMA")
        self.model.to(self.device)
        self.visual_teacher = copy.deepcopy(self.teacher_model.image_encoder)
        alpha = 0.999
        pseudo_label_weight='none'
        self.ema_teacher = EMATeacher(cfg, self.visual_teacher, alpha=alpha, pseudo_label_weight=pseudo_label_weight)
        #self.ema_teacher = EMATeacher(cfg, self.model.image_encoder, alpha=alpha, pseudo_label_weight=pseudo_label_weight)
        
        print("Network structure after applying LoRA:")
        print(self.model)
        print('------------------------------------------------------------------------------')
        print(self.ema_teacher)

        print("Turning off gradients in text encoder")

        # Freeze all parameters of text_encoder first
        for name, param in self.model.text_encoder.named_parameters():
            param.requires_grad = False

        # Update the visual encoder and prompt_learner
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad = True
            elif "lora_A" in name:
                param.requires_grad = True
            elif "lora_B" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.ema_teacher.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        #self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)

        # NOTE: Use optimizers for the visual encoder and prompt_learner
        #self.optim = build_optimizer(self.model, cfg.OPTIM)
        #self.t = nn.Parameter(torch.tensor([2.07], dtype=torch.float, requires_grad=True))  
        #params = list(self.model.parameters()) + [self.t]
        
        self.model.register_parameter("t", nn.Parameter(torch.tensor([2.1], dtype=torch.float, requires_grad=True)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)

        #self.optim = build_optimizer(params, cfg.OPTIM)
        # Make sure to pass the correct optim_cfg


        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # The original registration of prompt_learning, since cocoop only learns from the prompt, does not save the model weights of the clip. 
        # This work has improved the clip, so it needs to be changed. The part marked with # is the original approach of cocoop
        #self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("self_model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        
        # Choose between single-card training and multi-card training. If the user specifies only one GPU in cfg.TRAINER.DEVICES, directly use that GPU for training
        if isinstance(cfg.TRAINER.DEVICES, list) and len(cfg.TRAINER.DEVICES) == 1:
            single_gpu_id = cfg.TRAINER.DEVICES[0]
            print(f"Using single GPU: {single_gpu_id}")
            torch.cuda.set_device(single_gpu_id)  
            self.model.to(f"cuda:{single_gpu_id}")  
            self.model = nn.DataParallel(self.model, device_ids=[single_gpu_id])

        # If there are multiple Gpus and the user specifies multiple Gpus for training
        elif device_count > 1 and len(cfg.TRAINER.DEVICES) > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), using GPUs: {cfg.TRAINER.DEVICES}")
            self.model = nn.DataParallel(self.model, device_ids=cfg.TRAINER.DEVICES)

        # If there are multiple Gpus and the user does not specify a specific list of Gpus, all available Gpus will be used
        elif device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), using all available GPUs!")
            self.model = nn.DataParallel(self.model)

        # If only one GPU is detected or no device is specified
        else:
            print(f"Using single available GPU.")
            self.model.to(self.device)  

        # print(f'self.device: {self.device}, index: {self.device.index if self.device.index is not None else "default cuda:0"}')

        # Before updating the weights, save a copy of the weights of the teacher model
        self.teacher_weights_before = {name: param.clone() for name, param in self.teacher_model.image_encoder.named_parameters()}


    def forward_backward(self, batch_x, batch_u):

        source_image, source_label, target_image = self.parse_batch_train(batch_x, batch_u)
        self.model.to(self.device)
        model = self.model
        teacher_model = self.teacher_model
        optim = self.optim
        scaler = self.scaler
        self.epoch = self.epoch
        prec = self.cfg.TRAINER.LCMUDA.PREC
        if prec == "amp":
            with autocast():
                loss_source = model(source_image, source_label)
                logits_target = self.model(target_image)
                # predicted_labels = torch.argmax(logits_target, dim=1)
                # pesudo_labels
                pesudo_labels = teacher_model(target_image)

                # Calculate unsupervised losses (the loss function can be adjusted as needed)
                loss_target = F.cross_entropy(logits_target, pesudo_labels)
                loss = loss_source + loss_target
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_source = model(source_image, source_label)
            #print('loss_source:', loss_source)
            #print('source_label:', source_label)
            logits_target = self.model(target_image)
            #loss_source_tensor = torch.tensor(loss_source, device=source_image.device)
            average_loss_source = loss_source.mean()
            
            # pesudo_labels
            pesudo_labels, self.probs, pesudo_probs = teacher_model(target_image)

            self.features_by_class, self.feature_prototypes = self.compute_feature_prototypes(source_image, source_label, self.init_class_prototypes, self.features_by_class, self.epoch)


            # Calculate unsupervised losses
            loss_target = F.cross_entropy(logits_target, pesudo_labels)
            t_loss = self.compute_t_loss(source_image, source_label, self.model.module.t)
            loss = average_loss_source + loss_target + t_loss
            source_label_similarity_pairs = []
            source_all_label_similarities = []
            source_pesudo_labels, source_probs, pesudo_probs  = teacher_model(source_image)

            for source_img in source_image :

                source_img = source_img.type(self.model.module.dtype)
                source_feature = self.model.module.image_encoder(source_img.unsqueeze(0))
                #print('self.model.module.image_encoder:', self.model.module.device)
                source_best_label, source_similarity, source_l_s = self.classify_with_prototypes(source_feature)
                #pesudo_label = teacher_model(tar_img)
                # Output the source domain prediction label, similarity, and the true label of the source domain

                # Add (target_best_label, similarity) to the array
                source_label_similarity_pairs.append((source_best_label, source_similarity))
                #l_s.append((l_s))

                source_all_label_similarities.append(source_l_s)
            
            # After the loop is completed, the label_similarity_pairs array contains all combinations of (target_best_label, similarity)
            #print("source_All target best labels and similarities:", source_label_similarity_pairs)
            # all_label_similarities contains the list of l_s obtained in each loop
            #print("source_All label similarities for each target image:", source_all_label_similarities)
            
            ###The difference between the probability and the pseudo-label is taken as a power function, with e as the base
            results = []
            # Traverse each corresponding array in all_label_similarities and targetprobs
            for label, label_similarities, source_probabilities in zip(source_pesudo_labels, source_all_label_similarities, source_probs):
                label_int = int(label)
                similarity = dict(label_similarities)[label_int]
                #print("sssssssssss:", similarity)
                miresult = torch.exp(source_probabilities[label] - similarity)
                # Add the results to the Results list
                results.append(miresult)
                # First, convert the tensor in miresults to CPU, and then convert it to int
                results_float = [float(tensor.cpu().item()) for tensor in results]
                
            #print("Final results:", results_float)
#--------------------------------------------------------------------------
            optim.zero_grad()
            loss.backward()
            optim.step()
        loss_summary = {"loss": loss.item()}
        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u

    def load_model(self, directory, epoch = None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        #print('names:', names)

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
            #print('state_dict:', state_dict)
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def compute_feature_prototypes(self, source_image, source_label, init_class_prototypes, features_by_class, epoch):
        """
        Calculate the feature prototypes of each category. When epoch == 0, take the average of all features of that category. When epoch > 0, update using EMA
        """
        alpha = 0.999 
        self.feature_prototypes = init_class_prototypes
        with torch.no_grad():
            for img, label in zip(source_image, source_label):
                img = img.to(self.device)
                if isinstance(self.model, torch.nn.DataParallel):
                    img = img.to(torch.float16)  
                else:
                    img = img.type(self.model.dtype) 

                #img = img.type(self.model.module.dtype)
                #img = img.type(self.model.dtype)
                #print('tttttttttttttype:', self.model.module.dtype)

                feature = self.model.module.image_encoder(img.unsqueeze(0)) 
                feature = feature.to(self.device)
                
                label = label.item()

                if epoch == 0:
                    # Method One: Feature prototypes obtained from clip before training
                    self.feature_prototypes = {k: v.to(self.device) for k, v in init_class_prototypes.items()}  
                    # This is the initialization method during model training. At the beginning of the training, the feature prototypes are incomplete and the features are not precise
                    #if label not in features_by_class:
                        #features_by_class[label] = []
                    #features_by_class[label].append(feature)
                else:
                    if label in self.feature_prototypes:
                        self.feature_prototypes[label] = self.feature_prototypes[label].to(self.device) 
                        self.feature_prototypes[label] = alpha * self.feature_prototypes[label] + (1 - alpha) * feature
                        self.feature_prototypes = {k: v.to(self.device) for k, v in self.feature_prototypes.items()}
                    else:
                        # If there was no feature prototype for this category before (which is extremely rare)
                        self.feature_prototypes[label] = feature
        return features_by_class, self.feature_prototypes
    
    def classify_with_prototypes(self, features, threshold=0.1):
        """
        Classification is based on feature prototypes
        """
        # Used during verification
        # self.feature_prototypes = self.init_class_prototypes
        
        distances = []
        for label, prototype in self.feature_prototypes.items():
            # Calculate the cosine similarity between the feature and each prototype

            similarity = F.cosine_similarity(features, prototype)
            distances.append((label, similarity.item()))
        label_similarity = distances.copy()
        # Find the prototype with the highest similarity
        distances.sort(key=lambda x: x[1], reverse=True)
        best_label, best_similarity = distances[0]

        # Determine whether it is classified as unknown based on the threshold
        if best_similarity < threshold:
            return "unknown", best_similarity
        return best_label, best_similarity, label_similarity
    

    
    def init_feature_prototypes(self, source_domains, data_path, labels, batch_size, cache_path):

        # Check and create the folder where the feature prototypes are saved
        cache_dir = os.path.dirname(cache_path)  # Get the directory part of the file path
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)  # Create a non-existent folder
            print(f"Created directory: {cache_dir}")

        if os.path.exists(cache_path):
            print(f"Loading feature prototypes from {cache_path}...")
            with open(cache_path, 'rb') as f:
                feature_prototypes = pickle.load(f)
            return feature_prototypes

        # Obtain the path of the source domain image
        image_paths_by_class = {label: [] for label in labels}

        # Traverse the source domain and collect image paths of all categories
        for domain in source_domains:
            domain_path = os.path.join(data_path, domain)
            for label in labels:
                class_path = os.path.join(domain_path, self.data_lab2cname.get(label, "Key not found"))
                print('class_path:', class_path)
                if os.path.exists(class_path):
                    image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
                    image_paths_by_class[label].extend(image_files)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        feature_prototypes = {}

        print("Extracting features and computing prototypes...please wait...")

        with torch.no_grad():
            for classname, image_paths in image_paths_by_class.items():
                features = []

                for i in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[i:i+batch_size]
                    images = [preprocess(Image.open(img_path).convert("RGB")) for img_path in batch_paths]
                    image_tensors = torch.stack(images).to(self.device).half()

                    batch_features = self.model.image_encoder(image_tensors)
                    features.append(batch_features)

                if features:
                    features = torch.cat(features, dim=0)
                    feature_prototype = features.mean(dim=0)
                    feature_prototypes[classname] = feature_prototype
        if len(labels) == len(feature_prototypes):
            print("Feature prototypes have been computed for all classes.")
        else:
            raise ValueError(f"Mismatch between number of labels ({len(labels)}) and number of feature prototypes ({len(feature_prototypes)}).")

        # Save the feature prototype to the local file
        with open(cache_path, 'wb') as f:
            pickle.dump(feature_prototypes, f)
            print(f"Feature prototypes saved to {cache_path}.")

        return feature_prototypes


    def uncertainty_measure(self, test_image, dis = True):

        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        clip_model = load_clip_to_cpu(cfg)

        self.model.to(self.device)
        t_model = self.teacher_model
        source_label_similarity_pairs = []
        source_all_label_similarities = []
        test_image_similarity = []

        source_pesudo_labels, source_probs, pesudo_probs  = t_model(test_image)

        for source_img in test_image :

            source_img = source_img.type(self.model.module.dtype)
            source_feature = self.model.module.image_encoder(source_img.unsqueeze(0))
            source_best_label, source_similarity, source_l_s = self.classify_with_prototypes(source_feature)
            # Output the source domain prediction label, similarity, and the true label of the source domain
            source_label_similarity_pairs.append((source_best_label, source_similarity))
            test_image_similarity.append((source_similarity))
            source_all_label_similarities.append(source_l_s)
        
        # After the loop is completed, the label_similarity_pairs array contains all combinations of (target_best_label, similarity)
        # all_label_similarities contains the list of l_s obtained in each loop
        #print("source_All label similarities for each target image:", source_all_label_similarities)
        
        results = []
        for label, label_similarities, source_probabilities in zip(source_pesudo_labels, source_all_label_similarities, source_probs):
            label_int = int(label)
            similarity = dict(label_similarities)[label_int]
            miresult = torch.exp(source_probabilities[label]  - similarity)
            results.append(miresult)
            results_float = [float(tensor.cpu().item()) for tensor in results]
        return results_float, test_image_similarity
    

    def compute_mu_sigma_d_after_train(self):
        """
        After the training is completed, recalculate the features of all images and calculate μ_d and σ_d
        """
        print("Start calculating μ_d and σ_d...")
        
        all_similarities = [] 
        
        # Replace it with your path
        data_path = f'/xxx/xxx/UniMDA/DATA/{self.cfg.DATASET.NAME}'
        source_domain_names = self.cfg.DATASET.SOURCE_DOMAINS
        labels = list(self.data_lab2cname.keys())

        image_paths_by_class = {label: [] for label in labels}
        for domain in source_domain_names:
            domain_path = os.path.join(data_path, domain)
            for label in labels:
                class_path = os.path.join(domain_path, self.data_lab2cname[label])
                if os.path.exists(class_path):
                    image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
                    image_paths_by_class[label].extend(image_files)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
        ])

        with torch.no_grad():
            for label, image_paths in image_paths_by_class.items():
                if label not in self.feature_prototypes:
                    continue  
                prototype = self.feature_prototypes[label].to(self.device)

                for img_path in image_paths:
                    
                    image = Image.open(img_path).convert('RGB')
                    image = preprocess(image).unsqueeze(0).to(self.device)
                    image = image.type(self.model.module.dtype)

                    #image = image.type(self.model.module.dtype)
                    feature = self.model.module.image_encoder(image)
                    feature = feature / feature.norm(dim=-1, keepdim=True)  # 归一化

                    # Calculate the cosine similarity
                    similarity = F.cosine_similarity(feature, prototype.unsqueeze(0), dim=-1)
                    all_similarities.append(similarity.cpu().item())

        # Calculate the mean and standard deviation
        all_similarities = torch.tensor(all_similarities)
        mu_d = all_similarities.mean().item()
        sigma_d = all_similarities.std().item()
        return mu_d, sigma_d
    
    def compute_mu_sigma_t_after_train(self, tau=1.0):
        """
        After the training is completed, recalculate the classification probability p of all images and calculate μ_t and σ_t
        """
        print("Start calculating μ_t and σ_t...")
        
        all_t_values = []  

        ### Replace it with your path
        data_path = f'/xxx/xxx/UniMDA/DATA/{self.cfg.DATASET.NAME}'
        source_domain_names = self.cfg.DATASET.SOURCE_DOMAINS
        labels = list(self.data_lab2cname.keys())

        image_paths_by_class = {label: [] for label in labels}
        for domain in source_domain_names:
            domain_path = os.path.join(data_path, domain)
            for label in labels:
                class_path = os.path.join(domain_path, self.data_lab2cname[label])
                if os.path.exists(class_path):
                    image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
                    image_paths_by_class[label].extend(image_files)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
        ])

        with torch.no_grad():
            for label, image_paths in image_paths_by_class.items():
                if label not in self.feature_prototypes:
                    continue  
                prototype = self.feature_prototypes[label].to(self.device)

                for img_path in image_paths:

                    image = Image.open(img_path).convert('RGB')
                    image = preprocess(image).unsqueeze(0).to(self.device) 
                    image = image.type(self.model.module.dtype)  

                    feature = self.model.module.image_encoder(image)
                    feature = feature / feature.norm(dim=-1, keepdim=True)  

                    d = F.cosine_similarity(feature, prototype.unsqueeze(0), dim=-1)

                    logits = self.model.module(image)  
                    probs = F.softmax(logits, dim=-1)  
                    p = probs[:, label]  # Obtain the probability value of the current category

                    # σ((p - d)/τ) + d
                    sigma = torch.sigmoid((p - d) / tau)
                    t_value = sigma + d
                    all_t_values.append(t_value.cpu().item())

        all_t_values = torch.tensor(all_t_values)
        mu_t = all_t_values.mean().item()
        sigma_t = all_t_values.std().item()
        return mu_t, sigma_t
    
    def compute_t_loss(self, source_image, source_label, t):
        """
        Calculate t_loss
        """
        source_image_uncertainty_measure, source_image_similarity = self.uncertainty_measure(source_image)
        t_loss = torch.tensor(0.0, device=self.device)

        min_error_samples = len(source_label) / 4
        t_loss_list = []  
        for i in range(len(source_label)):
            if source_image_uncertainty_measure[i]  + source_image_similarity[i] * 1.1 < t:
                pred_logit = t - (source_image_uncertainty_measure[i]  + source_image_similarity[i] * 1.1)

                true_label = torch.tensor([0.0], device=self.device)
                t_loss_sample = F.binary_cross_entropy_with_logits(pred_logit, true_label)
                t_loss_list.append(t_loss_sample)

        # print('t_loss_list:', t_loss_list)
        # If there are misclassified samples in the batch, calculate the average loss. Otherwise, return 0
        if len(t_loss_list) >= min_error_samples:
            t_loss = torch.stack(t_loss_list).mean()
        else:
            t_loss = torch.tensor(0.0, device=self.device)

        return t_loss


