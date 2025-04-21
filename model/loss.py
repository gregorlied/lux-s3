import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, n_classes, weights=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

        self.weights = weights
        if self.weights is None:
            self.weights = [1] * self.n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, mask):
        if mask is not None:
            valid_indices = mask > 0
            score = score[valid_indices]
            target = target[valid_indices]

        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], mask)
            loss += dice * self.weights[i]
        return loss / self.n_classes
