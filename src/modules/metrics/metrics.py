from collections import OrderedDict

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.functional.classification import binary_average_precision
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import average_precision_score
import numpy as np

class MeanAveragePrecision(Metric):

    def __init__(self,
                 num_classes,
                 class_names,
                 background_index,
                 ignore_index,
                 merge_ignore_index_into_background_index,
                 return_dict,                
    ):
    
        super().__init__()

        self.num_classes = num_classes
        self.class_names = class_names
        self.background_index = background_index
        self.ignore_index = ignore_index
        self.merge_ignore_index_into_background_index = merge_ignore_index_into_background_index
        self.return_dict = return_dict

        # if merge_ignore_index_into_background_index:
        #     self.ignore_index = [self.background_index]
        # else:
        #     self.ignore_index = [self.background_index, self.ignore_index]
        self.ignore_index = [self.ignore_index]
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, oad_logits, oad_labels):

        preds, targets = self.prepare_preds_and_targets(oad_logits, oad_labels)

        # self.preds.append(preds.detach())
        # self.targets.append(targets.detach())    

        
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())        

    def compute(self):    

        self.result = {}

        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)

        preds = preds.cpu()
        targets = targets.cpu()

        valid_index = np.where(targets[:, 21]  != 1)[0]
        targets = targets[valid_index]
        #targets = F.one_hot(targets.to(dtype=torch.long), num_classes=22)
        preds = preds[valid_index]

        
        self.result["per_class_AP"] = {}
        for class_idx, class_name in enumerate(self.class_names):
            if class_idx not in self.ignore_index:
                if torch.any(targets[:,class_idx]):
                    self.result["per_class_AP"]["PerClassAP_"+class_name] = average_precision_score(np.array(targets[:,class_idx], dtype=int), np.array(preds[:, class_idx], dtype=np.float32))
        
        mean_ap_with_back= np.mean(list(self.result['per_class_AP'].values()))
        per_class_AP_without_background = self.result['per_class_AP'].copy()
        del per_class_AP_without_background['PerClassAP_Background']
        self.result["mAP"] = torch.mean(torch.tensor(list(per_class_AP_without_background.values()), dtype=torch.float32))
        
        if self.return_dict:
            return self.result        
        return self.result["mAP"]     

        # self.result["per_class_AP"] = {}
        # for class_idx, class_name in enumerate(self.class_names):
        #     if class_idx not in self.ignore_index:
        #         if torch.any(targets==class_idx):
        #             temp = targets.clone()
        #             temp[temp!=class_idx] = -1
        #             temp[temp==class_idx] = 1
        #             temp[temp==-1] = 0
        #             self.result["per_class_AP"]["PerClassAP_"+class_name] = binary_average_precision(preds[:, class_idx], temp)
        
        # self.result["mAP"] = torch.mean(torch.tensor(list(self.result["per_class_AP"].values())))
        
        # if self.return_dict:
        #     return self.result        
        # return self.result["mAP"]
    

    def prepare_preds_and_targets(self, logits, labels):

        if len(logits.shape) > 2:
            logits =  torch.flatten(logits, start_dim=0, end_dim=1)

        if len(labels.shape) > 2:
            labels = torch.flatten(labels, start_dim=0, end_dim=1)
            #targets = torch.argmax(labels, dim=1)

        preds = F.softmax(logits, dim=-1)

        return preds, labels

class MulticlassAccuracyWrapper(MulticlassAccuracy):

    
    def __init__(self,
                    num_classes,
                    top_k,
                    average,
                    ignore_index,
    ):
    
        super().__init__(num_classes, top_k, average, ignore_index=ignore_index)#

    def prepare_preds_and_targets(self, oad_logits, oad_labels):

        if len(oad_logits.shape) > 2:
            oad_logits =  torch.flatten(oad_logits, start_dim=0, end_dim=1)

        if len(oad_labels.shape) > 2:
            oad_labels = torch.flatten(oad_labels, start_dim=0, end_dim=1)
            targets = torch.argmax(oad_labels, dim=1)

        preds = F.softmax(oad_logits, dim=-1)

        return preds, targets


class WandBConfusionMatrix(Metric):

       
    
    def __init__(self,
                 num_classes,
                 class_names,
                 ignore_index,           
    ):
        
    
        super().__init__()

        self.num_classes = num_classes
        self.class_names = class_names
        self.ignore_index = ignore_index

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

     
    def update(self, oad_logits, oad_labels):       

        preds, targets = self.prepare_preds_and_targets(oad_logits, oad_labels)

        self.preds.append(preds.detach())
        self.targets.append(targets.detach())        

    
    def compute(self, phase): 

        preds = dim_zero_cat(self.preds).detach().cpu().numpy()
        targets = dim_zero_cat(self.targets).detach().cpu().numpy()

        #confusion_matrix = self.multiclass_confusion_matrix(preds, targets)
        #confusion_matrix_plot = self.multiclass_confusion_matrix.plot(confusion_matrix, labels=self.class_names)
        
        confusion_matrix_plot = wandb.plot.confusion_matrix(probs=preds, y_true=targets, class_names=self.class_names)
        wandb.log({f"{phase}/ConfusionMatrix": confusion_matrix_plot})

    def prepare_preds_and_targets(self, logits, labels):

        if len(logits.shape) > 2:
            logits =  torch.flatten(logits, start_dim=0, end_dim=1)

        if len(labels.shape) > 2:
            labels = torch.flatten(labels, start_dim=0, end_dim=1)
            targets = torch.argmax(labels, dim=1)

        preds = F.softmax(logits, dim=-1)

        return preds, targets



