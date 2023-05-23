import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)


def plot_auc(y_true, y_score, epoch):
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.title('ROC Curve (Epoch=%d)' % epoch, fontweight='bold', fontsize=16)
    plt.plot(fpr, tpr, 'dodgerblue', label='Test AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'lightgrey')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.xticks()
    plt.show()
