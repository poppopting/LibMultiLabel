import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=1, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class KLDivergenceLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss for multilabel tasks with label smoothing.

    This loss function computes the KL divergence between the predicted probabilities
    and the smoothed target distribution. Label smoothing is applied to prevent the
    model from becoming overconfident.

    Args:
        smoothing (float): The label smoothing factor. Typically a small value (e.g., 0.1).

    Attributes:
        smoothing (float): The smoothing factor for label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(KLDivergenceLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        """
        Forward pass for the KL divergence loss with label smoothing.

        Args:
            logits (torch.Tensor): The raw output from the model (logits).
            target (torch.Tensor): The target distribution (binary labels for each class).

        Returns:
            torch.Tensor: The computed KL divergence loss.
        """
        # Apply label smoothing: positive labels (1) to 1 - smoothing, negative labels (0) to smoothing
        target = target * (1 - self.smoothing) + (1 - target) * self.smoothing
        
        # Convert logits to probabilities using sigmoid
        p = torch.sigmoid(logits)
        q = target  # Smoothed target distribution
        
        # Compute KL divergence
        # kl_div = F.kl_div(p.log(), q, reduction='batchmean')
        # return kl_div

        kl_div = F.kl_div(p.log(), q, reduction='batchmean') + F.kl_div((1-p).log(), 1-q, reduction='batchmean') 
        return kl_div / p.size(1)


class JSDivergenceLoss(nn.Module):
    """
    Jensen-Shannon Divergence Loss for multilabel tasks with label smoothing.

    This loss function computes the JS divergence between the predicted probabilities
    and the smoothed target distribution. Label smoothing is applied to prevent the
    model from becoming overconfident.

    Args:
        smoothing (float): The label smoothing factor. Typically a small value (e.g., 0.1).

    Attributes:
        smoothing (float): The smoothing factor for label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(JSDivergenceLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        """
        Forward pass for the JS divergence loss with label smoothing.

        Args:
            logits (torch.Tensor): The raw output from the model (logits).
            target (torch.Tensor): The target distribution (binary labels for each class).

        Returns:
            torch.Tensor: The computed JS divergence loss.
        """
        # Apply label smoothing: positive labels (1) to 1 - smoothing, negative labels (0) to smoothing
        target = target * (1 - self.smoothing) + (1 - target) * self.smoothing
        
        # Convert logits to probabilities using sigmoid
        p = torch.sigmoid(logits)
        q = target  # Smoothed target distribution
        
        # Compute the average distribution
        m = 0.5 * (p + q)
        
        # # Compute JS divergence
        # js_div = 0.5 * F.kl_div(p.log(), m, reduction='batchmean') + \
        #          0.5 * F.kl_div(q.log(), m, reduction='batchmean')
        # return js_div

        # Compute JS divergence
        js_div = 0.5 * F.kl_div(p.log(), m, reduction='batchmean') + \
                 0.5 * F.kl_div(q.log(), m, reduction='batchmean') + \
                 0.5 * F.kl_div((1 - p).log(), 1-m, reduction='batchmean') + \
                 0.5 * F.kl_div((1 - q).log(), 1-m, reduction='batchmean') 
        return js_div / p.size(1)

class MultiLabelHingeLoss(nn.Module):
    """
    Multi-Label Hinge Loss for multi-label classification tasks.

    This loss function computes the hinge loss for each label independently
    and averages the loss across all labels for each sample.

    Args:
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    
    Attributes:
        reduction (str): Reduction method for the computed loss.
    """
    def __init__(self, reduction='mean', use_L2=True):
        super(MultiLabelHingeLoss, self).__init__()
        self.reduction = reduction
        self.use_L2 = use_L2

    def forward(self, logits, target):
        """
        Forward pass for the multi-label hinge loss.

        Args:
            logits (torch.Tensor): The raw output from the model (logits).
            target (torch.Tensor): The target binary labels for each class (0 or 1).

        Returns:
            torch.Tensor: The computed multi-label hinge loss.
        """
        # Convert target labels from {0, 1} to {-1, 1}
        target = 2 * target - 1
        
        # Compute hinge loss for each label: max(0, 1 - y * logits)
        hinge_loss = torch.clamp(1 - logits * target, min=0)
        if self.use_L2:
            hinge_loss = hinge_loss ** 2

        # Apply reduction: mean, sum or none
        if self.reduction == 'mean':
            return hinge_loss.mean()
        elif self.reduction == 'sum':
            return hinge_loss.sum()
        else:
            return hinge_loss

