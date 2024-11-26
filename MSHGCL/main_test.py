import os
import torch
import torch.nn as nn
from data_preprocessing import CustomData
from dataset import load_test_dataset
from metrics import *
import argparse

import warnings
warnings.filterwarnings("ignore")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_average(self):
        return self.avg
######################################################################################################
class BCEWithLogitsLoss0(nn.BCEWithLogitsLoss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', threshold=None):
        super(BCEWithLogitsLoss0, self).__init__(weight, size_average, reduce, reduction, threshold)
    def forward(self, input, target):
        return super(BCEWithLogitsLoss0, self).forward(input, target)


def val(model, criterion, dataloader, device, weight_interl, weight_intral):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]

        with torch.no_grad():
            pred = model((head_pairs, tail_pairs, rel))
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap, aupr = do_compute_metrics(pred_probs, label)

    running_loss.reset()

    model.train()

    return acc, auroc, f1_score, precision, recall, ap, aupr

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')

    args = parser.parse_args()

    params = dict(
        model='MSHGCL-DDI',
        data_root='data/preprocessed/',
        # data_root='data/small_preprocessed/',
        save_dir='save',
        dataset='drugbank',
        batch_size=args.batch_size,
    )


    data_path = os.path.join(params.get('data_root'), params.get('dataset'))
    test_loader = load_test_dataset(root=data_path, batch_size=params.get('batch_size'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load('data/preprocessed/model_big.pth')

    criterion = nn.BCEWithLogitsLoss()

    test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap, test_aupr = val(model, criterion, test_loader, device, params.get('weight_interl'), params.get('weight_intral'))

    print(f"Test Acc: {test_acc:.4f}, Test AUC-ROC: {test_auroc:.4f}, Test F1 Score: {test_f1_score:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test AP: {test_ap:.4f}, Test AUPR: {test_aupr:.4f}")

if __name__ == "__main__":
    main()