import torch
from sklearn.metrics import roc_auc_score, recall_score, precision_score

def No_Finding_accuracy(output, target, index=0):
    return accuracy_calc(output, target, index)

def Enlarged_Cardiomediastinum_acc(output, target, index=1):
    return accuracy_calc(output, target, index)

def Cardiomegaly_acc(output, target, index=2):
    return accuracy_calc(output, target, index)

def Lung_Opacity_acc(output, target, index=3):
    return accuracy_calc(output, target, index)

def Lung_Lesion_acc(output, target, index=4):
    return accuracy_calc(output, target, index)

def Edema_acc(output, target, index=5):
    return accuracy_calc(output, target, index)

def Consolidation_acc(output, target, index=6):
    return accuracy_calc(output, target, index)

def Pneumonia_acc(output, target, index=7):
    return accuracy_calc(output, target, index)

def Atelectasis_acc(output, target, index=8):
    return accuracy_calc(output, target, index)

def Pneumothorax_acc(output, target, index=9):
    return accuracy_calc(output, target, index)

def Pleural_Effusion_acc(output, target, index=10):
    return accuracy_calc(output, target, index)

def Pleural_Other_acc(output, target, index=11):
    return accuracy_calc(output, target, index)

def Fracture_acc(output, target, index=12):
    return accuracy_calc(output, target, index)

def Support_Devices_acc(output, target, index=13):
    return accuracy_calc(output, target, index)

# accuracy is calulated as the percetage of correctly classified label of a certain class
def accuracy_calc(output, target, class_index):
    with torch.no_grad():
        num_class = output.size()[1]
        num_sample = output.size()[0]
        validated_output = torch.sigmoid(output[:, class_index]).ge(0.5).float()
        acc = (target[:, class_index].float() == validated_output.float()).float().sum() / num_sample
    return acc


# this is mean of accuracy accross all classes
def multi_target_accuracy(output, target):
    with torch.no_grad():
        num_class = output.size()[1]
        num_sample = output.size()[0]
        accuracy = []
        for class_index in range(num_class):
            validated_output = torch.sigmoid(output[:, class_index]).ge(0.5).float()

            acc = (target[:, class_index].float() == validated_output.float()).sum() / num_sample
            # print ("==================== Index: {} ======================".format(class_index))
            # print (torch.sigmoid(output[:, class_index]))
            # print (output[:, class_index])
            # print (validated_output)
            # print (target[:, class_index])
            # print (acc)
            # print ("==================== Index: {} ======================".format(class_index))

            accuracy.append(acc)
    return sum(accuracy) / len(accuracy)

def false_negative_rate(output, target):
    pass

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def avg_auc_metric(output, target):
    output = output.detach().cpu()
    target = target.detach().cpu().byte()
    auc_scores = []

    for i in range(target.shape[1]):
        try:
            auc_scores.append(roc_auc_score(target[:, i], output[:, i], labels=[0, 1]))
        except ValueError:
            pass
    auc_scores = torch.tensor(auc_scores)
    return auc_scores.mean()

# Recall gives us an idea about when it’s actually yes, how often does it predict yes.
def avg_recall_score(output, target):
    output = output.detach().cpu()
    target = target.detach().cpu().byte()

    recall_scores = [recall_score(target[:, i], torch.sigmoid(output[:, i]).ge(0.5).float(), zero_division=1) for i in range(target.shape[1])]
    recall_scores = torch.tensor(recall_scores)
    return recall_scores.mean()

# Precsion tells us about when it predicts yes, how often is it correct.
def avg_precision_score(output, target):
    output = output.detach().cpu()
    target = target.detach().cpu().byte()
    precision_scores = [precision_score(target[:, i], torch.sigmoid(output[:, i]).ge(0.5).float(), zero_division=1) for i in range(target.shape[1])]
    precision_scores = torch.tensor(precision_scores)
    return precision_scores.mean()

# def confusion_matrix(output, target):
#     output = output.detach().cpu()
#     target = target.detach().cpu().byte()
