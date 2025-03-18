from sklearn.metrics import f1_score


def classification(y_pred, y_true):

    print('Micro F1: %.4f' % (f1_score(y_true, y_pred, average='micro')))
    print('Macro F1: %.4f' % (f1_score(y_true, y_pred, average='macro')))