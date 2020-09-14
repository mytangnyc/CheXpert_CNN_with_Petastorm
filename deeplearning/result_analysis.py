import re
import collections
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import argparse
import collections
import torch
import numpy as np
import deeplearning.data_loader.data_loaders as module_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import deeplearning.model.loss as module_loss
import deeplearning.model.metric as module_metric
import deeplearning.model.model as module_arch
from deeplearning.parse_config import ConfigParser
from deeplearning.trainer import Trainer

'''
result analysis script that reads in the log and model to generate presentable results, loss curve, confusion matrix, accuracy curve
'''

# regex that will parse the log file generated
regex = "((?:[A-Za-z]+_?)+)\s*:\s*(\d+\.?\d*)"
def log_extractor(log_path):
    log_dict = collections.defaultdict(list)
    with open(log_path, 'r') as f:
        for line in f.readlines():
            match = re.search(regex, line)
            if match:
                log_dict[match.group(1)].append(round(float(match.group(2)), 2))
    return log_dict

def plot_loss_curve(param_dict):
    plt.plot(param_dict["epoch"], param_dict["loss"], label="train loss")
    plt.plot(param_dict["epoch"], param_dict["val_loss"], label="val loss")
    plt.legend()
    plt.title("loss curve")
    plt.show()

    plt.savefig("saved/loss.png")

def no_findings_accuracy(param_dict):
    plt.plot(param_dict["epoch"], param_dict["loss"], label="train no findings accuracy")
    plt.plot(param_dict["epoch"], param_dict["val_loss"], label="val no findings accuracy")
    plt.legend()
    plt.title("loss curve")
    plt.show()

    plt.savefig("saved/loss.png")


def plot_confusion_matrix_helper(y_true, y_pred, class_names, normalized):
	cm = confusion_matrix(y_true, y_pred)
	display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
	display.plot()

def main(config):
    params = log_extractor("/home/kaiyuewang/Downloads/info.log")
    plot_loss_curve(params)
    no_findings_accuracy(params)
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_set_config'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    num_class = 14
    # build model architecture
    model = config.init_obj('arch', module_arch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(config.resume, map_location=torch.device(device))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
    new_state_dict = {"module." + key: value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)


    # prepare model for testing
    model = model.to(device)
    model.eval()

    predicted_label = torch.empty((0, 14)).float()
    true_labels = torch.empty((0, 14)).float()
    print(predicted_label.shape)
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted_label = torch.cat((predicted_label, output.to("cpu").float()))
            true_labels = torch.cat((true_labels, target.to("cpu").float()))
            # np.vstack(true_labels, target.to("cpu").numpy())
            print (predicted_label.size())

    labels = ['No Finding',
                   'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                   'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                   'Support Devices']
    for class_index in range(num_class):
        class_output = torch.sigmoid(predicted_label[:, class_index]).ge(0.5).float()
        class_label = true_labels[:, class_index]
        plot_confusion_matrix_helper(class_label.to('cpu'), class_output.to('cpu'), ["Negative", "Positive"], normalized='true')
        plt.title("confusion_matrix_{}".format(labels[class_index]))
        plt.savefig("saved/updated-confusion_matrix_{}".format(labels[class_index]))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="/home/kaiyuewang/bdh-spring-2020-project-CheXpert/deeplearning/saved/models/ChexpertNet/0426_120934/model_best.pth", type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-l', '--log-path', default="", type=str,
                      help='path to generated log')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.

    config = ConfigParser.from_args(args)
    main(config)


