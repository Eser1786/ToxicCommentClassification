from sklearn.metrics import(
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt

def evaluate_model(y_test, y_pred, y_prod):
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

    fpr, tpr, thresholds = roc_curve(y_test, y_prod)

    roc_auc = roc_auc_score(y_test, y_prod)

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