import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

# binary cross entropy loss for classification result, we are comparing class by class and accumulate the loss to be our final loss
def cross_entropy(output, target):
    loss = 0
    for class_index in range(output.size()[1]):
        loss += F.binary_cross_entropy_with_logits(output[:, class_index], target[:, class_index].float())
    return loss