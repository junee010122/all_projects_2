import pandas as pd
import yaml
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import f_oneway, ttest_ind, chi2_contingency, normaltest, mannwhitneyu, kruskal
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

from utils.plots import plot_statistical_results

def load_config(config_path="configs/params.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_statistical_tests(df, config):
    results = {}
    tests = config["analysis"].get("hypothesis_tests", [])
    
    # Determine data characteristics
    numerical_cols = [col for col in df.columns if col != "Pass/Fail" and np.issubdtype(df[col].dtype, np.number)]
    categorical_cols = [col for col in df.columns if col == "Pass/Fail" or not np.issubdtype(df[col].dtype, np.number)]
    
    for test in tqdm(tests, desc="Running Statistical Tests"):
        if test == "anova" and len(numerical_cols) > 1:
            # Check normality assumption
            p_values = [normaltest(df[col]).pvalue for col in numerical_cols]
            if all(p > 0.05 for p in p_values):  # Normal distribution assumed
                results["ANOVA"] = f_oneway(*[df[col] for col in numerical_cols])
            else:
                results["Kruskal-Wallis"] = kruskal(*[df[col] for col in numerical_cols])
        
        elif test == "t-test" and len(numerical_cols) >= 2:
            col1, col2 = numerical_cols[:2]
            # Check normality assumption
            p1, p2 = normaltest(df[col1]).pvalue, normaltest(df[col2]).pvalue
            if p1 > 0.05 and p2 > 0.05:
                results["T-Test"] = ttest_ind(df[col1], df[col2])
            else:
                results["Mann-Whitney"] = mannwhitneyu(df[col1], df[col2])
        
        elif test == "chi-square" and len(categorical_cols) >= 2:
            chi2, p, _, _ = chi2_contingency(pd.crosstab(df[categorical_cols[0]], df[categorical_cols[1]]))
            results["Chi-Square"] = (chi2, p)
    
    # Plot results
    #plot_statistical_results(results)
    from IPython import embed
    embed()

    return results


def train_model(X_train, y_train, config):
    model_type = config["model"]["type"]

    # Convert continuous target variable to categorical if a classification model is used
    if model_type in ["logistic_regression", "random_forest", "svm"]:
        if y_train.dtype == "float64" or y_train.dtype == "int64":
            y_train = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="quantile").fit_transform(y_train.values.reshape(-1, 1)).ravel()

    if model_type == "logistic_regression":
        model = LogisticRegression(C=config["model"]["hyperparameters"]["C"])
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config["model"]["hyperparameters"]["n_estimators"],
            max_depth=config["model"]["hyperparameters"]["max_depth"]
        )
    elif model_type == "svm":
        model = SVC(C=config["model"]["hyperparameters"]["C"])
    
    for _ in tqdm(range(1), desc="Training Model"):
        model.fit(X_train, y_train)
    
    return model

