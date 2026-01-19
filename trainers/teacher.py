from copy import deepcopy

import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class EMATeacher(nn.Module):

    def __init__(self, cfg, model, alpha, pseudo_label_weight):
        super(EMATeacher, self).__init__()
        self.ema_model = deepcopy(model)
        device = torch.device(f'cuda:{cfg.TRAINER.DEVICES[0]}')
        self.ema_model = self.ema_model.to(device)
        self.alpha = alpha
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

        print(f'EMA model is on device: {next(self.ema_model.parameters()).device}')
        
        
    def _init_ema_weights(self, model):
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()


    def _update_ema(self, model, teacher_model, iter):
        if iter == 0:
            alpha_teacher = 0
            #alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        else:
            alpha_teacher = self.alpha
        self.ema_model = teacher_model
        #print(f'model is on device: {next(model.parameters()).device}')
        
        #print(f'EMA model is on device: {next(self.ema_model.parameters()).device}')
        


        model_state_dict = model.state_dict()
        ema_state_dict = self.ema_model.state_dict()

        for name, param in model_state_dict.items():
            if "lora" in name:  
                ema_param = ema_state_dict[name]  
                print("name:", name)
                print("ema_state_dict[name]", ema_state_dict[name])
                print("param:", param)
                #if iter == 0:
                    #ema_param.copy_(param)  
                #else:
                ema_param.copy_(alpha_teacher * ema_param + (1 - alpha_teacher) * param)


        '''for (ema_name, ema_param), (name, param) in zip(self.ema_model.named_parameters(), model.named_parameters()):
            print("ema_name:", ema_name)
            print("name:", name)
            if iter == 0:
                ema_param.data = ema_param.data
            else:
                if "lora" in name:  
                    if not param.data.shape:  # scalar tensor
                        print("ema_name:", ema_name)
                        print("name:", name)
                        ema_param.data = \
                            alpha_teacher * ema_param.data + \
                            (1 - alpha_teacher) * param.data
                    #else:
                        #ema_param.data[:] = \
                            #alpha_teacher * ema_param[:].data[:] + \
                            #(1 - alpha_teacher) * param[:].data[:]'''


    '''def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        
        #alpha_teacher = 1 
        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]


    def update_weights(self, model, teacher_model, iter):
        # Init/update ema model
        if iter == 0:
            return teacher_model
            #self._init_ema_weights(model)
        if iter > 0:
            self._update_ema(model, teacher_model, iter)
        return self.ema_model

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        logits, _ = self.ema_model(target_img)

        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight
