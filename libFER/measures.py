""" 
   * Source: libFER.measures.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI

"""

import torch

from sklearn.metrics import confusion_matrix


def get_confusion_matrix_sklearn(y_true, y_pred, num_classes):

    """get_confusion_matrix_sklearn function

    Note: function for getting a confusion matrix using sklearn

    Arguments: 
        y_true: label
        y_pred: predicted output
        num_classes: number of classes 

    Returns:
        a calculated confusion matrix

    """

    return confusion_matrix(y_true, y_pred, labels=[i for i in range(num_classes)])


def get_confusion_matrix_1(y_true, y_pred, num_classes):

    """get_confusion_matrix_1 function

    Note: function for getting a confusion matrix

    Arguments: 
        y_true: label
        y_pred: predicted output
        num_classes: number of classes 

    Returns:
        cfm (tensor): a calculated confusion matrix

    """

    cfm = torch.zeros((args.num_classes, num_classes), dtype=int)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cfm[t, p] +=1
    return cfm


# rewrite sklearn method to torch
def get_confusion_matrix_2(y_true, y_pred, num_classes):

    """get_confusion_matrix_2 function

    Note: function for getting a confusion matrix

    Arguments: 
        y_true: label
        y_pred: predicted output
        num_classes: number of classes 

    Returns:
        (tensor): a calculated confusion matrix
    """

    N = num_classes
    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)
    return torch.sparse.LongTensor(
        torch.stack([y_true, y_pred]), 
        torch.ones_like(y_true, dtype=torch.long),
        torch.Size([N, N])).to_dense()


# weird trick with bincount
def get_confusion_matrix_3(y_true, y_pred, num_classes):

    """get_confusion_matrix_3 function

    Note: function for getting a confusion matrix

    Arguments: 
        y_true: label
        y_pred: predicted output
        num_classes: number of classes 

    Returns:
        y (tensor): a calculated confusion matrix
    """

    N = num_classes
    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)
    y = N * y_true + y_pred
    #print(y)
    y = torch.bincount(y)
    #print(y)
    if len(y) < N * N:
        y = torch.cat((y, torch.zeros(N * N - len(y), dtype=torch.long)))
    #    print(y)
    y = y.reshape(N, N)
    return y


def accuracy(output, label, topk=(1,)):

    """accuracy function

    Note: function for getting accuracy value

    Arguments: 
        output: predicted ouput
        label: label
        topk: top1 or top5

    Returns:
        res (list): list of accuracies for multiple top k
    """

    # Computes the precision@k for the sprcified values of k
    maxk = max(topk)
    batch_size = label.size(0)

    # topk: returns the k largest elements of the given input tensor along a given dimension.
    _, pred = output.topk(maxk, 1, True, True)
    # transpose
    pred = pred.t()
    # expand_as: input의 size와 같게 변경
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res