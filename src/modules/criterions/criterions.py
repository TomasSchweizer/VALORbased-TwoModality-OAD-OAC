import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionBackgroundSphereLoss(nn.Module):
    """ This criterion attempts to maintain the magnitude of known samples higher than an specified threshold as well as push the magnitude of unknown samples towards 0 (zero).
    """

    def __init__(self, min_action_sphere_radius, max_action_sphere_radius, reduction):

        super().__init__()

        self.min_action_sphere_radius = min_action_sphere_radius
        self.max_action_sphere_radius = max_action_sphere_radius
        self.reduction = reduction

    def forward(self, actionbackground_features_allframes, actionbackground_idxs_allframes):

         # > 1 for action 0 for background
        neg_idxs = (actionbackground_idxs_allframes < 0.0)
        pos_idxs = (actionbackground_idxs_allframes >= 0.0)

        # compute feature magnitude
        mag = actionbackground_features_allframes.norm(p=2, dim=1)
        # for knowns we want a certain minimum magnitude
        mag_diff_from_ring = torch.clamp(self.min_action_sphere_radius - mag, min=0.0)   


        # create container to store loss per sample
        loss = torch.zeros(actionbackground_features_allframes.shape[0], device=actionbackground_features_allframes.device)
        # knowns: punish if magnitude is inside of ring
        loss[pos_idxs] = mag_diff_from_ring[pos_idxs]
        # unknowns: punish any magnitude
        loss[neg_idxs] = mag[neg_idxs]
        # exponentiate each element and remove negative values

        # #TODO delete later added max l2 norm loss
        if self.max_action_sphere_radius != None:
            pos_idxs_with_zero = torch.where(loss==0.0)[0]
            loss[pos_idxs_with_zero] += torch.clamp(mag[pos_idxs_with_zero] - self.max_action_sphere_radius, min=0.0) 

        loss = torch.pow(loss, 2)

        if torch.any(torch.isnan(loss)):
            print("Sphere loss is nan")

        if self.reduction == "mean":
            loss = torch.mean(loss)
            return loss
        elif self.reduction == "sum":
            loss = torch.sum(loss)
            return loss
        elif self.reduction == "none":
            return loss


class ActionBackgroundExamplarContrastiveLoss(nn.Module):
    def __init__(self, apply_action_wise_mean, temp, base_temp):
        
        super().__init__()
        self.apply_action_wise_mean = apply_action_wise_mean
        self.temp = temp
        self.base_temp = base_temp

    def forward(self, actionbackground_features_actionframes, action_idxs_actionframes, examplars):

      
        # TODO check if to normalize examplars
        examplars = F.normalize(examplars, dim=1)
        
        num_examplars = examplars.shape[0]
        num_features = actionbackground_features_actionframes.shape[0]

        # TODO check if mean makes any sense here
        if self.apply_action_wise_mean:
            mean_mask_indices = torch.arange(num_examplars, device=action_idxs_actionframes.device).unsqueeze(1).repeat(1, num_features)
            mean_mask = torch.eq(mean_mask_indices, action_idxs_actionframes.unsqueeze(0))

            mean_masked_actionbackground_features_actionframes = actionbackground_features_actionframes.unsqueeze(0).repeat(num_examplars, 1, 1)
            mean_masked_actionbackground_features_actionframes[~mean_mask] = 0.0
            mean_actionbackground_features_actionframes = mean_masked_actionbackground_features_actionframes.sum(dim=1) / mean_mask.sum(dim=1, dtype=torch.float).unsqueeze(1)
            mean_action_idxs_actionframes = torch.nonzero(mean_mask.sum(dim=1))

            # remove values for actions which didnt occur in the batch
            mean_actionbackground_features_actionframes = mean_actionbackground_features_actionframes[~torch.any(mean_actionbackground_features_actionframes.isnan(),dim=1)]

            # overwrite to perform contrastive loss 
            action_idxs_actionframes = mean_action_idxs_actionframes
            actionbackground_features_actionframes = mean_actionbackground_features_actionframes

        idx_vec = torch.arange(0,num_examplars, device=action_idxs_actionframes.device)
        mask = torch.eq(action_idxs_actionframes[:,None], idx_vec[None,:]) # Mask to select only values for which the in the nominator the labels match

        rep_dot_examplar = (actionbackground_features_actionframes @ examplars.T) / self.temp
       
        # Try with and without for numerical stability
        logits_max, _ = torch.max(rep_dot_examplar, dim=1, keepdim=True)
        logits = rep_dot_examplar - logits_max.detach()

        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True)) # Log trick Log(A/B) = Log(A) - Log(B)
        mean_log_prob = (mask * log_prob).sum(dim=1).mean() 
        contrastive_loss = - (self.temp / self.base_temp) * mean_log_prob # returns negative log prob of the compactness loss
        
        if torch.isnan(contrastive_loss):
            print("Contrastive loss is nan")
        
        return contrastive_loss

class ActionBackgroundContrastiveLoss(nn.Module):
    
    def __init__(self, temp, background_sphere_radius):
        
        super().__init__()
        self.temp = temp
        self.background_sphere_radius = background_sphere_radius
    
    def forward(self, actionbackground_features_allframes, actionbackground_binaryidxs_allframes):


        # TODO uncomment if normalization is not done in forward pass
        # action_idxs_mask = (actionbackground_binaryidxs_allframes==1.0)
        # background_idxs_mask = (actionbackground_binaryidxs_allframes==0.0)
        # actionbackground_features_allframes_normalized = torch.zeros_like(actionbackground_features_allframes, device=actionbackground_features_allframes.device)
        # actionbackground_features_allframes_normalized[action_idxs_mask] = actionbackground_features_allframes[action_idxs_mask] / torch.norm(actionbackground_features_allframes[action_idxs_mask], dim=1, keepdim=True)
        # actionbackground_features_allframes_normalized[background_idxs_mask] = self.background_sphere_radius * (actionbackground_features_allframes[background_idxs_mask] / torch.norm(actionbackground_features_allframes[background_idxs_mask], dim=1, keepdim=True))

        maxpool_featsim = torch.exp(actionbackground_features_allframes @ actionbackground_features_allframes.T / self.temp)
        same_idx = (actionbackground_binaryidxs_allframes[:,None]==actionbackground_binaryidxs_allframes[None,:]).to(dtype=torch.float32)
        I = torch.eye(len(same_idx), device=same_idx.device,dtype=torch.float32)
        same_idx = same_idx - I
        not_same_idx = 1.0 - same_idx - I
        countpos = torch.sum(same_idx)
        
        if countpos == 0:
            #print("Batch has no same pairs")
            contrastive_loss = 0.0
        elif torch.all(actionbackground_binaryidxs_allframes==0) or torch.all(actionbackground_binaryidxs_allframes==1):
            #print("Batch has only same pairs")
            contrastive_loss = 0.0
        else:
            maxpool_featsim_pos = same_idx * maxpool_featsim
            maxpool_featsim_negsum = torch.sum(not_same_idx * maxpool_featsim, dim=1)
            simprob = maxpool_featsim_pos/(maxpool_featsim_negsum + maxpool_featsim_pos) + not_same_idx
            contrastive_loss = torch.sum(-torch.log(simprob + I)) / countpos
        
            if torch.isnan(contrastive_loss):
                print("Stop")

        return contrastive_loss

class ActionContrastiveLoss(nn.Module):
    
    def __init__(self, temp):
        
        super().__init__()
        self.temp = temp
    
    def forward(self, action_features_actionframes, action_idxs_actionframes):

        # TODO just for testing
        norms = torch.norm(action_features_actionframes, dim=1)
        if torch.any(norms==0.0):
            print("norm is zero")
        action_features_actionframes_normalized = F.normalize(action_features_actionframes, p=2, dim=1)

        maxpool_featsim = torch.exp(action_features_actionframes_normalized @ action_features_actionframes_normalized.T / self.temp)
        same_idx = (action_idxs_actionframes[:,None]==action_idxs_actionframes[None,:]).to(dtype=torch.float32)
        I = torch.eye(len(same_idx), device=same_idx.device,dtype=torch.float32)
        same_idx = same_idx - I
        not_same_idx = 1.0 - same_idx - I
        countpos = torch.sum(same_idx)
        
        if countpos == 0:
            #print("Batch has no same pairs")
            contrastive_loss = 0.0
        elif torch.all(torch.diff(action_idxs_actionframes)==0):
            #print("Batch has only same pairs")
            contrastive_loss = 0.0
        else:
            maxpool_featsim_pos = same_idx * maxpool_featsim
            maxpool_featsim_negsum = torch.sum(not_same_idx * maxpool_featsim, dim=1)
            simprob = maxpool_featsim_pos/(maxpool_featsim_negsum + maxpool_featsim_pos) + not_same_idx
            contrastive_loss = torch.sum(-torch.log(simprob + I)) / countpos
        
            if torch.isnan(contrastive_loss):
                print("Action contrastive loss is nan")

        return contrastive_loss

class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temp):
        
        super().__init__()
        self.temperature = temp

    def forward(self, feats, labels):
        # feats shape: [B, D]
        # labels shape: [B]
        
        # TODO uncomment if normalization is not done in forward pass
        #feats = F.normalize(feats, dim=-1, p=2) 
        
        # compute ground-truth distribution p
        mask = torch.eq(labels.view(-1, 1), labels.contiguous().view(1, -1)).to(feats.device, dtype=torch.float32)
        mask.fill_diagonal_(0.0) # self masking
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        # compute contrastive distribution q       
        logits = torch.matmul(feats, feats.T) / self.temperature
        logits.fill_diagonal_(-1e9) # self masking
        # optional: minus the largest logit to stablize logits
        logits = self._stablize_logits(logits)        
        loss = self._compute_cross_entropy(p, logits)

        if torch.isnan(loss):
            print("Supervised contrastive loss is nan")

        return loss    
    
    def _stablize_logits(self, logits):
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()
        return logits
    
    def _compute_cross_entropy(self, p, q):
        q = F.log_softmax(q, dim=-1)
        loss = torch.sum(p * q, dim=-1)
        return - loss.mean()


class ActionSimilarToExamplarsLoss(nn.Module):
    
    def __init__(self):
        
        super().__init__()

    def forward(self, action_features_actionframes, action_idxs_actionframes, examplars, examplars_variances):

        # examplars_matrix_ordered = examplars[action_idxs_actionframes]
        # variances_matrix_ordered = examplars_variances[action_idxs_actionframes]

        # action_similar_to_examplars_loss = torch.mean(torch.mean(torch.abs(action_features_actionframes - examplars_matrix_ordered) / variances_matrix_ordered, dim=1))

        # return action_similar_to_examplars_loss

        #! More similar to paper 
        examplars_matrix_ordered = examplars[action_idxs_actionframes]
        variances_matrix_ordered = examplars_variances[action_idxs_actionframes]

        action_similar_to_examplars_loss = torch.mean(torch.sum(torch.abs(action_features_actionframes - examplars_matrix_ordered) / variances_matrix_ordered, dim=1))

        return action_similar_to_examplars_loss

class MaximalEntropyLoss(nn.Module):
    # TODO check if changing negative index to 0 works
    """ This criterion produces more rigorous decision boundaries for known classes (target >= 0) and also increases the entropy for negative training samples (target < 0).
    """

    def __init__(self, margin, num_classes, reduction, weight):

        super().__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.reduction = reduction
        self.weight = weight

        self.register_buffer("eye", torch.eye(self.num_classes))
        self.register_buffer("ones", torch.ones(self.num_classes))
        self.unknowns_multiplier = 1.0 / self.num_classes
    
    def forward(self, action_logits_allframes, action_targets_idxs_allframes):

        #TODO delete later just for testing the basic Mamba with objectosphere loss
        if action_logits_allframes.shape[1] == 21:
            action_targets_idxs_allframes = action_targets_idxs_allframes - 1

        # initialize variables with zeros
        categorical_targets = torch.zeros_like(action_logits_allframes)
        margin_logits = torch.zeros_like(action_logits_allframes)

        # get boolean tensor (true/false) indicating elements satisfying criteria
        neg_indexes = (action_targets_idxs_allframes < 0)
        pos_indexes = (action_targets_idxs_allframes >= 0)
        # convert known targets to categorical: "target 0 to [1 0 0]", "target 1 to [0 1 0]", "target 2 to [0 0 1]" (ternary example)
        categorical_targets[pos_indexes, :] = self.eye[action_targets_idxs_allframes[pos_indexes]]
        # expand self.ones matrix considering unknown_indexes to obtain maximum entropy: "[0.5  0.5] for two classes" and "[0.33  0.33 0.33] for three classes"
        categorical_targets[neg_indexes, :] = (self.ones.expand(neg_indexes.count_nonzero().item(), self.num_classes) * self.unknowns_multiplier)
        
        target_logits = action_logits_allframes - self.margin
        margin_logits[pos_indexes] = action_logits_allframes[pos_indexes] * (1 - self.eye[action_targets_idxs_allframes[pos_indexes]]) + target_logits[pos_indexes] * self.eye[action_targets_idxs_allframes[pos_indexes]]
        margin_logits[neg_indexes] = action_logits_allframes[neg_indexes]

        # obtain negative log softmax in range [0, +inf)
        negative_log_values = (-1) * torch.nn.functional.log_softmax(margin_logits, dim=1)

         # obtain ground-truth loss for knowns and distributed loss for unknown classes (element wise)
        loss = negative_log_values * categorical_targets
        # get loss for each sample in batch
        loss = torch.sum(loss, dim=1)
        # compute weighted loss
        if self.weight is not None:
            loss = loss * self.weight
        # return batch loss
        if   self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum':  return loss.sum()
        else: return loss

class EntropicOpenSetLoss(torch.nn.Module):
    # TODO check if changing negative index to 0 works
    """ This criterion increases the entropy for negative training samples (target < 0).
    """
    def __init__(self, num_classes, reduction, weight):
        
        super().__init__()

        self.num_classes = num_classes
        self.reduction = reduction
        self.weight = weight

        self.register_buffer("eye", torch.eye(self.num_classes))
        self.register_buffer("ones", torch.ones(self.num_classes))
        self.unknowns_multiplier = 1.0 / self.num_classes

    def forward(self, action_logits_allframes, action_targets_idxs_allframes):

        # get boolean tensor (true/false) indicating elements satisfying criteria
        categorical_targets = torch.zeros_like(action_logits_allframes)
        neg_indexes = (action_targets_idxs_allframes < 0)
        pos_indexes = (action_targets_idxs_allframes >= 0)
        # convert known targets to categorical: "target 0 to [1 0 0]", "target 1 to [0 1 0]", "target 2 to [0 0 1]" (ternary example)
        categorical_targets[pos_indexes, :] = self.eye[action_targets_idxs_allframes[pos_indexes]]
        # expand self.ones matrix considering unknown_indexes to obtain maximum entropy: "[0.5  0.5] for two classes" and "[0.33  0.33 0.33] for three classes"
        categorical_targets[neg_indexes, :] = (self.ones.expand(neg_indexes.count_nonzero().item(), self.num_classes) * self.unknowns_multiplier)
        # obtain negative log softmax in range [0, +inf)
        negative_log_values = (-1) * torch.nn.functional.log_softmax(action_logits_allframes, dim=1)
        # obtain ground-truth loss for knowns and distributed loss for unknown classes (element wise)
        loss = negative_log_values * categorical_targets
        # get loss for each sample in batch
        # loss = torch.sum(loss, dim=1) # batch is in current setup 1
        # compute weighted loss
        if self.weight is not None:
            loss = loss * self.weight
        # return batch loss
        if   self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum':  return loss.sum()
        else: return loss

class NoneUniformCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, reduction, n_last, use_weights, dataset_name):
        
        super().__init__()
        self.num_classes = num_classes
        self.n_last = n_last
        self.reduction = reduction
        self.use_weights = use_weights
        self.dataset_name = dataset_name

        if self.dataset_name == "thumos":
            self.weights = torch.tensor(([0.00399121, 1.        , 0.17611259, 0.38874895, 0.18483034,
                                            0.25923852, 0.37368846, 0.38680033, 0.07342214, 0.37795918,
                                            0.34864458, 0.07959429, 0.09210265, 0.09197457, 0.09747368,
                                            0.11823289, 0.31909028, 0.52914286, 0.70903522, 0.20893502,
                                            0.36399371, 0.2100726 ]), dtype=torch.float32)
        else:
            self.weights = None

        self.cel = nn.CrossEntropyLoss(weight= self.weights, reduction=self.reduction)

    def forward(self, logits, targets): 
        """
        logits: b,l,num_used_classes
        targets: b,l,
        """
        logits = logits[:,-self.n_last:,:]
        targets = targets[:,-self.n_last:]
        if logits.ndim > 2:
            logits = torch.flatten(logits, start_dim=0, end_dim=1)
        if targets.ndim > 1:
            targets = torch.flatten(targets, start_dim=0, end_dim=1)

        loss = self.cel(logits, targets)

        return loss
    

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, reduction, use_weights, dataset_name):
        
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.use_weights = use_weights
        self.dataset_name = dataset_name

        if self.dataset_name == "thumos" and self.num_classes == 21:

            #! Effective number class weights
            # self.weights = torch.tensor([0.14495639, 3.20366488, 0.62699289, 1.29094439, 0.65413561,
            #                                 0.88621339, 1.24385354, 1.28485124, 0.30990282, 1.25720669,
            #                                 1.16555789, 0.32864552, 0.36686407, 0.36647147, 0.38334813,
            #                                 0.44733941, 1.07318283, 1.73008395, 2.29299191, 0.72925025,
            #                                 1.21354272], dtype=torch.float32)
            
            #! Inverse number of samples per class scaled between 0 and 1
            self.weights = torch.tensor( [0.00391679, 1.        , 0.17611259, 0.38874895, 0.18483034,
                                            0.25923852, 0.37368846, 0.38680033, 0.07342214, 0.37795918,
                                            0.34864458, 0.07959429, 0.09210265, 0.09197457, 0.09747368,
                                            0.11823289, 0.31909028, 0.52914286, 0.70903522, 0.20893502,
                                            0.36399371], dtype=torch.float32)
        
        elif self.dataset_name == "thumos" and self.num_classes == 22:
    
            self.weights = torch.tensor(([0.00399121, 1.        , 0.17611259, 0.38874895, 0.18483034,
                                0.25923852, 0.37368846, 0.38680033, 0.07342214, 0.37795918,
                                0.34864458, 0.07959429, 0.09210265, 0.09197457, 0.09747368,
                                0.11823289, 0.31909028, 0.52914286, 0.70903522, 0.20893502,
                                0.36399371, 0.2100726 ]), dtype=torch.float32)
            
        elif  self.dataset_name == "thumos" and self.num_classes == 42:

            self.weights = torch.tensor([0.00633966, 1.        , 0.37260903, 0.1850304 , 0.1025695 ,
                0.40890008, 0.11617366, 0.18737976, 0.26295896, 0.23212583,
                0.36506747, 0.38437253, 0.26582969, 0.40685046, 0.89357798,
                0.10437205, 0.17858453, 0.35703812, 0.59317905, 0.31278099,
                0.17734887, 0.08372013, 0.24435524, 0.09063838, 0.29822413,
                0.09637839, 0.14472511, 0.09712804, 0.23756098, 0.11889648,
                0.280046  , 0.32860999, 0.28496197, 0.53991131, 0.50835073,
                0.51808511, 0.31258023, 0.21976534, 0.54051054, 0.36479401,
                0.41376381, 1.0], dtype=torch.float32) #TODO check if weight for non existing ambiguous class in training important
        
        elif self.dataset_name == "thumos" and self.num_classes == 20:
            self.weights = torch.tensor([3.07231664, 0.60128658, 1.23801649, 0.62731647, 0.84987921,
                1.19285633, 1.23217315, 0.297197  , 1.20566201, 1.11777076,
                0.31517126, 0.35182288, 0.35144637, 0.3676311 , 0.42899878,
                1.02918301, 1.6591516 , 2.19898069, 0.69935145, 1.16378824], dtype=torch.float32)
        else:
            self.weights = None

        self.cel_train = nn.CrossEntropyLoss(weight=self.weights, reduction=self.reduction)
        self.cel_valid = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, logits, targets): 
        """
        logits: b*l,num_used_classes
        targets: b*l,
        """
        if logits.shape[1] == self.weights.shape[0]: 
            loss = self.cel_train(logits, targets)
        else: 
            loss = self.cel_valid(logits, targets)

        if torch.isnan(loss):
            print("Stop wcel is nan")

        return loss

class WeightedNLLLoss(nn.Module):
    def __init__(self, num_classes, reduction, use_weights, dataset_name):
        
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.use_weights = use_weights
        self.dataset_name = dataset_name

        if self.dataset_name == "thumos" and self.num_classes == 21:

            #! Effective number class weights
            # self.weights = torch.tensor([0.14495639, 3.20366488, 0.62699289, 1.29094439, 0.65413561,
            #                                 0.88621339, 1.24385354, 1.28485124, 0.30990282, 1.25720669,
            #                                 1.16555789, 0.32864552, 0.36686407, 0.36647147, 0.38334813,
            #                                 0.44733941, 1.07318283, 1.73008395, 2.29299191, 0.72925025,
            #                                 1.21354272], dtype=torch.float32)
            
            #! Inverse number of samples per class scaled between 0 and 1
            self.weights = torch.tensor( [0.00391679, 1.        , 0.17611259, 0.38874895, 0.18483034,
                                            0.25923852, 0.37368846, 0.38680033, 0.07342214, 0.37795918,
                                            0.34864458, 0.07959429, 0.09210265, 0.09197457, 0.09747368,
                                            0.11823289, 0.31909028, 0.52914286, 0.70903522, 0.20893502,
                                            0.36399371], dtype=torch.float32)
        
        elif self.dataset_name == "thumos" and self.num_classes == 20:
            self.weights = torch.tensor([3.07231664, 0.60128658, 1.23801649, 0.62731647, 0.84987921,
                1.19285633, 1.23217315, 0.297197  , 1.20566201, 1.11777076,
                0.31517126, 0.35182288, 0.35144637, 0.3676311 , 0.42899878,
                1.02918301, 1.6591516 , 2.19898069, 0.69935145, 1.16378824], dtype=torch.float32)
        else:
            self.weights = None

        self.nlll= nn.NLLLoss(weight=self.weights, reduction=self.reduction)

    def forward(self, logits, targets): 
        """
        logits: b*l,num_used_classes
        targets: b*l,
        """
        # softplus_logits = F.softplus(logits)
        # negative_softplus_logits = - softplus_logits
        loss = self.nlll(logits, targets)

        return loss

class WeightedFocalLoss(nn.Module):
    
    def __init__(self, num_classes, gamma, reduction, use_weights, dataset_name):
        
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.use_weights = use_weights
        self.dataset_name = dataset_name

        if self.dataset_name == "thumos" and self.num_classes == 21:
            
            #! Effective number class weights
            # self.weights = torch.tensor([0.14495639, 3.20366488, 0.62699289, 1.29094439, 0.65413561,
            #                                 0.88621339, 1.24385354, 1.28485124, 0.30990282, 1.25720669,
            #                                 1.16555789, 0.32864552, 0.36686407, 0.36647147, 0.38334813,
            #                                 0.44733941, 1.07318283, 1.73008395, 2.29299191, 0.72925025,
            #                                 1.21354272], dtype=torch.float32)
            
            #! Inverse number of samples per class scaled between 0 and 1
            self.weights = torch.tensor( [0.00391679, 1.        , 0.17611259, 0.38874895, 0.18483034,
                                            0.25923852, 0.37368846, 0.38680033, 0.07342214, 0.37795918,
                                            0.34864458, 0.07959429, 0.09210265, 0.09197457, 0.09747368,
                                            0.11823289, 0.31909028, 0.52914286, 0.70903522, 0.20893502,
                                            0.36399371], dtype=torch.float32)

        
        elif self.dataset_name == "thumos" and self.num_classes == 20:
            self.weights = torch.tensor([3.07231664, 0.60128658, 1.23801649, 0.62731647, 0.84987921,
                1.19285633, 1.23217315, 0.297197  , 1.20566201, 1.11777076,
                0.31517126, 0.35182288, 0.35144637, 0.3676311 , 0.42899878,
                1.02918301, 1.6591516 , 2.19898069, 0.69935145, 1.16378824], dtype=torch.float32)
        else:
            self.weights = None


    def forward(
        self,
        logits,
        labels,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes).to(torch.float32)

        weights = self.weights.unsqueeze(0).to(device=labels_one_hot.device)
        weights = weights.repeat(batch_size, 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, num_classes)

        return self._focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.gamma)

    def _focal_loss(self, logits, labels, alpha=None, gamma=2):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
        logits: A float tensor of size [batch, num_classes].
        labels: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
        focal_loss: A float32 scalar representing normalized total loss.
        """
        bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * bc_loss

        if alpha is not None:
            weighted_loss = alpha * loss
            focal_loss = torch.sum(weighted_loss)
        else:
            focal_loss = torch.sum(loss)

        focal_loss /= torch.sum(labels)
        return focal_loss

class OddsRatioLoss(nn.Module):

    def __init__(self, reduction):
        
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, labels):
        """
            logits: b*l, num_used_classes
            labels: b*l, 2
        """
        winning_labels = labels[:,0]
        losing_labels = labels[:,1]
        assert torch.all(winning_labels != losing_labels)
        logits_idxs = torch.arange(logits.shape[0], device=logits.device)

        #! Slower not numerical as good, but easier to interpret
        probs = F.softmax(logits, dim=1)
        winning_probs = probs[logits_idxs, winning_labels]
        losing_probs = probs[logits_idxs, losing_labels] 

        winning_odds = winning_probs / (1.0 - winning_probs)
        losing_odds = losing_probs / (1.0 - losing_probs)

        odds_ratio = winning_odds / losing_odds
        log_odds_ratio = torch.log(odds_ratio)
        sigmoid_log_odds_ratio = F.sigmoid(log_odds_ratio)
        ratio = -torch.log(sigmoid_log_odds_ratio)

        if self.reduction == "mean":
            loss = torch.mean(ratio)
        elif self.reduction == "sum":
            loss = torch.sum(ratio)
        else: 
            raise ValueError("Reduction is not implemented!")


        #! Fast computation, check if correct
        # log_probs = F.log_softmax(logits, dim=1)

        # winning_log_probs = log_probs[logits_idxs, winning_labels]
        # losing_log_probs = log_probs[logits_idxs, losing_labels] 
        # assert torch.all(winning_log_probs != losing_log_probs)

        # log_odds = (winning_log_probs - losing_log_probs) - (torch.log(1 - torch.exp(winning_log_probs)) - torch.log(1 - torch.exp(losing_log_probs)))
        # sig_log_odds_ratio = torch.nn.functional.sigmoid(log_odds)
        # ratio = - torch.log(sig_log_odds_ratio) # TODO check if the loss is correctly implemented with this minus here

        # if self.reduction == "mean":
        #     loss = torch.mean(ratio)
        # elif self.reduction == "sum":
        #     loss = torch.sum(ratio)
        # else: 
        #     raise ValueError("Reduction is not implemented!")

        return loss


class ContrastiveMultimodalAlignmentLoss(nn.Module):

    def __init__(
            self,
            starting_temp,

    ):
        super().__init__()

        self.temp = nn.Parameter(torch.tensor(starting_temp))


    def forward(self,
                text_tokens, 
                text_features_contra,
                text_features_contra_weighted,
                video_features_pooled_contra,
                video_features_pooled_contra_weighted,
                video_features_mask,
                oad_labels = None
        ):
    
        mask_txt_tokens = (text_tokens != 0).to(dtype=torch.long, device=text_tokens.device)
        #mask_video_features  = torch.ones(*video_features_pooled_contra.shape[:2]).to(dtype=torch.long, device=video_features_pooled_contra.device)
        mask_video_features = video_features_mask

        weights_txt_tokens = text_features_contra_weighted.squeeze(2)
        weights_video_features  = video_features_pooled_contra_weighted.squeeze(2)

        score_matrix_tv = self.compute_fine_matrix(text_features_contra,
                                                video_features_pooled_contra,
                                                mask_txt_tokens, 
                                                mask_video_features, 
                                                weights_txt_tokens, 
                                                weights_video_features)
        
        contra_loss_tv = self.contrastive_loss(score_matrix_tv).mean()

        return contra_loss_tv

    def compute_fine_matrix(self, featA, featB, maskA, maskB, weightA, weightB):

        weightA.masked_fill_((1 - maskA.clone().detach()).to(torch.bool), float("-inf"))
        weightA = torch.softmax(weightA, dim=-1)  # B x N_t
        #print(weightA.shape)

        weightB.masked_fill_((1 - maskB.clone().detach()).to(torch.bool), float("-inf"))
        weightB = torch.softmax(weightB, dim=-1)  # B x N_v
        #print(weightB.shape)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [featA, featB])
        #print(retrieve_logits.shape)
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, maskA])
        #print(retrieve_logits.shape)
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, maskB])
        #print(retrieve_logits.shape)

        
        A2B_logits = retrieve_logits.max(dim=-1)[0]  # abtv -> abt
        B2A_logits = retrieve_logits.max(dim=-2)[0]  # abtv -> abv
        #print(A2B_logits.shape)
        #print(B2A_logits.shape)


        A2B_logits = torch.einsum('abt,at->ab', [A2B_logits, weightA])
        B2A_logits = torch.einsum('abv,bv->ab', [B2A_logits, weightB])
        score_matrix = (A2B_logits + B2A_logits) / 2.0
        #print(A2B_logits.shape)
        #print(B2A_logits.shape)


        return score_matrix
    
    def contrastive_loss(self, score_matrix):  ### labels for unicl 
    
        score_matrix = score_matrix / self.temp
        matrix1 = -F.log_softmax(score_matrix, dim=1)
        matrix2 = -F.log_softmax(score_matrix, dim=0)      
        loss1 = matrix1.diag()
        loss2 = matrix2.diag()
        contra_loss = torch.mean(torch.cat((loss1,loss2), dim=0))

        return contra_loss


class SupervisedContrastiveMultimodalAlignmentLoss(nn.Module):

    def __init__(
            self,
            starting_temp,

    ):
        super().__init__()

        #self.temp = nn.Parameter(torch.tensor(starting_temp, dtype=torch.float32))
        self.temp = torch.tensor(starting_temp, dtype=torch.float32)

    def forward(self,
                text_tokens, 
                text_features_contra,
                text_features_contra_weighted,
                video_features_pooled_contra,
                video_features_pooled_contra_weighted,
                video_features_mask,
                oad_labels = None
        ):

        mask_txt_tokens = (text_tokens != 0).to(dtype=torch.long, device=text_tokens.device)
        mask_video_features = video_features_mask

        weights_txt_tokens = text_features_contra_weighted.squeeze(2)
        weights_video_features  = video_features_pooled_contra_weighted.squeeze(2)

        score_matrix_tv = self.compute_fine_matrix(text_features_contra,
                                                video_features_pooled_contra,
                                                mask_txt_tokens, 
                                                mask_video_features, 
                                                weights_txt_tokens, 
                                                weights_video_features)
        
        if len(oad_labels.shape) > 2:
            oad_labels = torch.flatten(oad_labels, start_dim=0, end_dim=1)
            oad_labels = torch.argmax(oad_labels, dim=1)
        
        labels_sections, laebls_section_counts  = torch.unique_consecutive(oad_labels, dim=0, return_counts=True)

        mask = torch.eq(labels_sections.view(-1, 1), labels_sections.contiguous().view(1, -1)).to(score_matrix_tv.device, dtype=torch.float32)
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        contra_loss_tv = self.contrastive_loss(score_matrix_tv,p).mean()

        return contra_loss_tv
    
    def compute_fine_matrix(self, featA, featB, maskA, maskB, weightA, weightB):

        weightA.masked_fill_((1 - maskA.clone().detach()).to(torch.bool), float("-inf"))
        weightA = torch.softmax(weightA, dim=-1)  # B x N_t
        #print(weightA.shape)

        weightB.masked_fill_((1 - maskB.clone().detach()).to(torch.bool), float("-inf"))
        weightB = torch.softmax(weightB, dim=-1)  # B x N_v
        #print(weightB.shape)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [featA, featB])
        #print(retrieve_logits.shape)
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, maskA])
        #print(retrieve_logits.shape)
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, maskB])
        #print(retrieve_logits.shape)

        
        A2B_logits = retrieve_logits.max(dim=-1)[0]  # abtv -> abt
        B2A_logits = retrieve_logits.max(dim=-2)[0]  # abtv -> abv
        #print(A2B_logits.shape)
        #print(B2A_logits.shape)


        A2B_logits = torch.einsum('abt,at->ab', [A2B_logits, weightA])
        B2A_logits = torch.einsum('abv,bv->ab', [B2A_logits, weightB])
        score_matrix = (A2B_logits + B2A_logits) / 2.0
        #print(A2B_logits.shape)
        #print(B2A_logits.shape)


        return score_matrix
    
    def contrastive_loss(self, score_matrix, p):  ### labels for unicl 
    
        score_matrix = score_matrix / self.temp
        matrix1 = -F.log_softmax(score_matrix, dim=1)
        matrix2 = -F.log_softmax(score_matrix, dim=0)      
        loss1 = torch.sum(matrix1 * p, dim=1)
        loss2 = torch.sum(matrix2 * p, dim=0)
        contra_loss = torch.mean(torch.cat((loss1,loss2), dim=0))

        return contra_loss


class MultimodalCaptioningLoss(nn.Module):

    def __init__(
            self,

    ):
        super().__init__()

    def forward(self, logits_over_vocab, labels_txt_tokens):

        logits_over_vocab = logits_over_vocab[:, :labels_txt_tokens.shape[1], :]
        logits_over_vocab = logits_over_vocab[labels_txt_tokens != -1]
        multimodal_captioning_loss = F.cross_entropy(logits_over_vocab, labels_txt_tokens[labels_txt_tokens != -1])

        return multimodal_captioning_loss
    

class OADLoss(nn.Module):
    def __init__(self, num_classes, reduction, use_weights, dataset_name):
        
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.use_weights = use_weights
        self.dataset_name = dataset_name
        
        if self.dataset_name == "thumos" and self.num_classes == 22 and self.use_weights:
    
            self.weights = torch.tensor(([0.00399121, 1.        , 0.17611259, 0.38874895, 0.18483034,
                                0.25923852, 0.37368846, 0.38680033, 0.07342214, 0.37795918,
                                0.34864458, 0.07959429, 0.09210265, 0.09197457, 0.09747368,
                                0.11823289, 0.31909028, 0.52914286, 0.70903522, 0.20893502,
                                0.36399371, 0.2100726 ]), dtype=torch.float32)      
            
            # self.weights = torch.tensor(([0.1, 1.        , 1, 1, 1,
            #                     1, 1, 1, 1, 1,
            #                     1, 1, 1, 1, 1,
            #                     1, 1, 1, 1, 1,
            #                     1, 1 ]), dtype=torch.float32)         
        else:
            self.weights = None

        self.cel_train = nn.CrossEntropyLoss(weight=self.weights, reduction=self.reduction)
        self.cel_valid = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, oad_logits, oad_labels): 
        """
        logits: b*l,num_used_classes
        targets: b*l,
        """

        logits, labels = self.prepare_logits_and_labels(oad_logits, oad_labels)

        if self.use_weights:
            if logits.shape[1] == self.weights.shape[0]: 
                loss = self.cel_train(logits, labels)
            else: 
                loss = self.cel_valid(logits, labels)
        else:
            loss = self.cel_valid(logits, labels)


        if torch.isnan(loss):
            print("Stop wcel is nan")

        return loss
    
    def prepare_logits_and_labels(self, logits, labels):

        if len(logits.shape) > 2:
            logits =  torch.flatten(logits, start_dim=0, end_dim=1)

        if len(labels.shape) > 2:
            labels = torch.flatten(labels, start_dim=0, end_dim=1)
            labels = torch.argmax(labels, dim=1)

        return logits, labels
    


class MultipLableCrossEntropyLoss(nn.Module):

    def __init__(self, num_classes, reduction, use_weights, dataset_name):
            
            super().__init__()
            self.num_classes = num_classes
            self.reduction = reduction
            self.use_weights = use_weights
            self.dataset_name = dataset_name
            self.ignore_index = 21

            self.weights = torch.tensor(([0.00399121, 1.        , 0.17611259, 0.38874895, 0.18483034,
                    0.25923852, 0.37368846, 0.38680033, 0.07342214, 0.37795918,
                    0.34864458, 0.07959429, 0.09210265, 0.09197457, 0.09747368,
                    0.11823289, 0.31909028, 0.52914286, 0.70903522, 0.20893502,
                    0.36399371, 0.2100726 ]), dtype=torch.float32)


    def forward(self, oad_logits, oad_labels):

        oad_logits, oad_labels = self.prepare_logits_and_labels(oad_logits, oad_labels)


        logsoftmax = nn.LogSoftmax(dim=1).to(oad_logits.device)
        if self.ignore_index >= 0:
            if (oad_labels[:, self.ignore_index] == 1).all():
                return torch.tensor(0.0, dtype=torch.float32).to(oad_logits.device)
            notice_index = [i for i in range(oad_labels.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-oad_labels[:, notice_index] * logsoftmax(oad_logits[:, notice_index]), dim=1)
            #output = torch.sum(-(target[:, notice_index]*self.weights[None,notice_index].to(logits.device)) * logsoftmax(logits[:, notice_index]), dim=1) # B
            if self.reduction == 'mean':      
                return torch.mean(output[oad_labels[:, self.ignore_index] != 1])
            elif self.reduction == 'sum':
                return torch.sum(output[oad_labels[:, self.ignore_index] != 1])
            else:
                return output[oad_labels[:, self.ignore_index] != 1]
        else:
            output = torch.sum(-oad_labels * logsoftmax(oad_logits), dim=1)
            #output = torch.sum(-(target[:, :] * self.class_weights[None,:].to(logits.device)) * logsoftmax(logits), dim=1)
            if self.reduction == 'mean':
                return torch.mean(output)
            elif self.reduction == 'sum':
                return torch.sum(output)
            else:
                return output
            
    def prepare_logits_and_labels(self, logits, labels):

        if len(logits.shape) > 2:
            logits =  torch.flatten(logits, start_dim=0, end_dim=1)

        if len(labels.shape) > 2:
            labels = torch.flatten(labels, start_dim=0, end_dim=1)

        return logits, labels