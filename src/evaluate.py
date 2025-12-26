# src/evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


def _plot_confusion_matrix(cm, labels, title, save_path):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def evaluate(trained_models, plots_dir="plots"):
    """
    trained_models: dict name -> (model, X_test, y_test, statsbomb_xg_test)
    """
    os.makedirs(plots_dir, exist_ok=True)

    # ROC (tutti insieme)
    fig_roc = plt.figure()
    plt.title("ROC Curve (Test Set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # PR (tutti insieme)
    fig_pr = plt.figure()
    plt.title("Precision-Recall Curve (Test Set)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    # Per baseline PR usiamo la prevalenza della classe positiva nel test
    any_model = next(iter(trained_models.values()))
    _, _, y_test_any, _ = any_model
    baseline = float(np.mean(y_test_any))

    for name, (model, X_test, y_test, sb_xg) in trained_models.items():
        y_pred = model.predict(X_test)
        xg_ours = model.predict_proba(X_test)[:, 1]

        print(f"\n=== {name} ===")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(classification_report(y_test, y_pred, digits=3))

        # ROC-AUC nostro
        auc_ours = roc_auc_score(y_test, xg_ours)
        print(f"ROC-AUC nostro xG: {auc_ours:.3f}")

        # ROC-AUC StatsBomb (solo su righe senza NaN)
        sb_xg = sb_xg.astype(float)
        mask = sb_xg.notna().values
        if mask.sum() > 0 and len(np.unique(y_test[mask])) > 1:
            auc_sb = roc_auc_score(y_test[mask], sb_xg[mask])
            print(f"ROC-AUC StatsBomb xG: {auc_sb:.3f}")

            # Correlazione (solo dove sb_xg è presente)
            corr = np.corrcoef(xg_ours[mask], sb_xg[mask])[0, 1]
            print(f"Correlazione xG (nostro vs StatsBomb): {corr:.3f}")
        else:
            print("ROC-AUC StatsBomb xG: N/A (valori mancanti o classi non presenti)")

        # Confusion matrix plot
        _plot_confusion_matrix(
            cm,
            labels=["NoGoal", "Goal"],
            title=f"Confusion Matrix - {name}",
            save_path=os.path.join(plots_dir, f"cm_{name}.png")
        )

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, xg_ours)
        plt.figure(fig_roc.number)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_ours:.3f})")

        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, xg_ours)
        ap = average_precision_score(y_test, xg_ours)
        plt.figure(fig_pr.number)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    # Diagonale ROC (random)
    plt.figure(fig_roc.number)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_all_models.png"), dpi=200)
    plt.close(fig_roc)

    # Baseline PR = prevalenza positiva
    plt.figure(fig_pr.number)
    plt.hlines(baseline, 0, 1, linestyles="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pr_all_models.png"), dpi=200)
    plt.close(fig_pr)

    print(f"\nGrafici salvati in: {plots_dir}/")
    print("- roc_all_models.png")
    print("- pr_all_models.png")
    print("- cm_<Modello>.png per ogni modello")
