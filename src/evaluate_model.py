from sklearn.metrics import(
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    hamming_loss,
    ConfusionMatrixDisplay,
    multilabel_confusion_matrix,
    average_precision_score,
    auc,
    precision_recall_curve
)

import numpy as np

import matplotlib.pyplot as plt

def evaluate_model(y_test, y_pred, y_prob):
    print(classification_report(y_test, y_pred))

    # === confusion matrix ===

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    plt.imshow(cm)

    plt.title("Confusion Matrix SVM")

    plt.xlabel("Predicted Label - Nhãn dự đoán")
    plt.ylabel("True Label - Nhãn thật")

    plt.colorbar()

    plt.xticks([0,1], ["Non-Toxic", "Toxic"])
    plt.yticks([0,1], ["Non-Toxic", "Toxic"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i,j],
                ha = "center",
                va="center"
            )
    
    plt.savefig("results/confusion_matrix_linear_SVM.png")

    # plt.show()

    # === ROC Curve ===

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8,6))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC Curve (AUC = {roc_auc:.3f})"
    )

    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve - SVM")

    plt.legend()

    plt.savefig("results/roc_curve_Linear_SVM.png")

    # plt.show()

    print("ROC-AUC:", roc_auc)



def evaluate_multi_labels_model(y_test, y_pred, y_prob_2d, cols):

    if hasattr(y_test, "values"):
        y_test = y_test.values

    print(classification_report(y_test, y_pred, target_names=cols, zero_division=0))

    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    print()

    # metric Hamming loss
    print("Hamming loss:", hamming_loss(y_test, y_pred))

    print()

    # metric ROC-AUC/PR-AUC


    print("ROC-AUC (macro):", roc_auc_score(y_test, y_prob_2d, average='macro'))
    print("PR-AUC (macro):", average_precision_score(y_test, y_prob_2d, average='macro'))

    print()

    fig, axes = plt.subplots(1, 2, figsize=(16,6))

    # ROC-AUC từng nhãn
    for i, label in enumerate(label_cols):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_prob_2d[:, i])
        roc_auc = auc(fpr,tpr)
        axes[0].plot(fpr,tpr, label=f"{label} (AUC={roc_auc:.3f})")

    axes[0].plot([0, 1],[0, 1], 'k--')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve - Multi-label")
    axes[0].legend(loc="lower right", fontsize=8)

    # PR-AUC từng nhãn
    for i, label in enumerate(label_cols):
        precision, recall, _ =  precision_recall_curve(y_test[:, i], y_prob_2d[:, i])
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, label=f"{label} (AUC={pr_auc:.3f})")

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Curve - Multi-label")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    # plt.savefig("results/multilabel_roc_pr_curves.png", dpi=150)
    # plt.show()



    # metric Confusion matrix
    mcm = multilabel_confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    for i, (ax, label) in enumerate(zip(axes.flat, label_cols)):
        ConfusionMatrixDisplay(mcm[i], display_labels=['Not', label]).plot(ax=ax)
        ax.set_title(f"Confusion Matrix - {label}")

    plt.tight_layout()
    plt.savefig("results/multilabel_confusion_matrices_threshold.png", dpi=150)
    plt.show()
