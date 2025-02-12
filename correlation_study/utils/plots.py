import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

def plot_statistical_results(results):
    plt.figure(figsize=(8, 5))
    labels, p_values = zip(*[(k, v.pvalue if hasattr(v, 'pvalue') else v[1]) for k, v in results.items()])
    sns.barplot(x=p_values, y=labels, palette='viridis')
    plt.axvline(0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
    plt.xlabel("P-Value")
    plt.title("Statistical Test Results")
    plt.legend()
    plt.show()

def plot_correlation_heatmap(df, output_path="results/correlation_heatmap.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()

def plot_boxplots(df, output_path="results/boxplot.png"):
    plt.figure(figsize=(12, 8))
    df.boxplot(rot=90)
    plt.title("Feature Distribution")
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, output_path="results/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.close()

