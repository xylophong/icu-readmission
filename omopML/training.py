import numpy as np
import pandas as pd
from datetime import date, datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, auc, roc_auc_score, roc_curve, log_loss, f1_score, 
    precision_score, recall_score, make_scorer, fbeta_score,
    precision_recall_curve
)
import shap
from torch.utils.tensorboard import SummaryWriter
import plotly.express as px

class SingleLabelTrainer:
    """
    Randomly splits data in a training, validation and testing dataset, and trains an 
    XGBoost model for a single target, providing useful evaluation tools including a 
    Tensorboard callback.
    
    NB: Requires the presence of a "./data" folder and a "./data/models" subfolder

    Parameters
    ----------
    task_name : str
        Training task name, for saving purposes
    X : `DataFrame` object
        Dataset features in a pandas DataFrame format
    y : `Series` object
        Dataset labels in a pandas Series format
    test_size : float
        Proportion of the dataset to keep as the testing set
    val_size : float
        Proportion of the training set to keep as the validation set
    seed : int
        Dataset split random seed/state

    """

    def __init__(self, task_name, X, y, test_size, val_size, seed=42):
        self.task_name = task_name
        self.X = X
        self.y = y
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

        self.split_data()

    def split_data(self):
        test_gss = GroupShuffleSplit(
            n_splits=2, test_size=self.test_size, random_state=self.seed
        )
        self.pretrain_idx, self.test_idx = next(
            test_gss.split(
                self.X, 
                groups=self.X.index.get_level_values("person_id")
            )
        )

        self.X_pretrain = self.X.iloc[self.pretrain_idx]
        self.X_test = self.X.iloc[self.test_idx]
        self.y_pretrain = self.y.iloc[self.pretrain_idx]
        self.y_test = self.y.iloc[self.test_idx]

        val_gss = GroupShuffleSplit(
            n_splits=2, test_size=self.test_size, random_state=self.seed
        )
        self.train_idx, self.val_idx = next(
            val_gss.split(
                self.X_pretrain,
                groups=self.X_pretrain.index.get_level_values("person_id"),
            )
        )

        self.X_train = self.X_pretrain.iloc[self.train_idx]
        self.X_val = self.X_pretrain.iloc[self.val_idx]
        self.y_train = self.y_pretrain.iloc[self.train_idx]
        self.y_val = self.y_pretrain.iloc[self.val_idx]

    def train_model(self, model_params, early_stopping_rounds=5, verbose=0):
        """
        Launches model training, using the validation set to evaluate the model at
        each epoch and decide whether to stop training or continue. 
        
        NB: Requires the presence of a "./data" folder and a "./data/models" subfolder

        Parameters
        ----------
        model_params : dict
            Dictionary of XGBoost parameters (sklearn API syntax)
        early_stopping_rounds : int
            Number of rounds after which to stop training if performance on validation
            set has not improved
        verbose : int
            Verbose parameter to set the level of text to show at each epoch

        """
        self.model_params = model_params
        self.models = {}
        self.target = "event"
        self.probas = pd.DataFrame(columns=[self.target])
        self.val_probas = pd.DataFrame(columns=[self.target])
        self.default_predict = pd.DataFrame(columns=[self.target])

        logdir = "/export/home/dpnguyen/tensorboard_data/{}_{}_{}".format(
            self.task_name,
            datetime.now().strftime("%Y%m%d-%H%M"), 
            self.target
        )

        self.models[self.target] = XGBClassifier(**self.model_params)
        print("Training model...")
        self.models[self.target].fit(
            self.X_train.values,
            self.y_train,
            callbacks=[self.TensorBoardCallback(log_dir=logdir)],
            eval_metric=self.custom_metrics,
            eval_set=[
                (self.X_train.values, self.y_train),
                (self.X_val.values, self.y_val),
            ],
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        self.models[self.target].save_model(
            "./data/models/{}_{}_{}.bin".format(
                datetime.now().strftime("%Y%m%d-%H%M"), self.target, self.task_name
            )
        )

        self.probas.loc[:, self.target] = self.models[self.target].predict_proba(
            self.X_test.values
        )[:, 1]
        self.val_probas.loc[:, self.target] = self.models[self.target].predict_proba(
            self.X_val.values
        )[:, 1]
        self.default_predict.loc[:, self.target] = self.models[self.target].predict(
            self.X_test.values
        )

    def tune_thresholds(self, beta=1):
        """
        Optimizes the prediction threshold according to F-beta score  

        Parameters
        ----------
        beta : float
            Beta parameter for f-beta score

        """
        self.beta = beta
        print(f"Tuning thresholds to maximize f-{self.beta} score...")
        self.threshold = self.compute_threshold_f_beta(
            "f-beta", self.beta, self.y_val, self.val_probas
        )

    def evaluate(self, y_true, y_probas, cv=None, print_results=False):
        """
        Calculates and optionally prints evaluation metrics   

        Parameters
        ----------
        y_true : `Series` object
            True labels on which to evaluate results
        y_probas : `Series` object
            Probabilites from the model output
        cv : int
            Represents number of folds for cross validation. If cv==None, no cross
            validation is done.
        print_results : bool
            Whether to print results or not

        """

        y_pred = y_probas >= self.threshold

        self.metrics = {}
        self.metrics["Precision"] = precision_score(y_true, y_pred.event)
        self.metrics["Sensitivity/Recall"] = recall_score(y_true, y_pred.event)
        self.metrics["Specificity"] = recall_score(1 - y_true, 1 - y_pred.event)
        self.metrics["F1 score"] = f1_score(y_true, y_pred.event)
        self.metrics["ROC AUC"] = roc_auc_score(y_true, y_probas.event)
        self.metrics["Accuracy"] = accuracy_score(y_true, y_pred.event)

        if print_results:
            for score in self.metrics:
                print(f"{score}: {round(self.metrics[score], 4)}")

        if cv != None:
            custom_scorer = {
                "Accuracy": make_scorer(self.accuracy_score, needs_proba=True),
                "Precision": make_scorer(self.precision_score, needs_proba=True),
                "Sensitivity/Recall": make_scorer(self.recall_score, needs_proba=True),
                "Specificity": make_scorer(self.specificity_score, needs_proba=True),
                "F1 score": make_scorer(self.f1_score, needs_proba=True),
                "ROC AUC": make_scorer(roc_auc_score, needs_proba=True),
            }
            self.cv_metrics = cross_validate(
                XGBClassifier(**self.model_params),
                self.X_train.values,
                self.y_train.values,
                cv=GroupShuffleSplit(
                    n_splits=cv, test_size=self.test_size, random_state=self.seed
                ),
                groups=self.X_train.index.get_level_values("patient_id").values,
                scoring=custom_scorer,
            )

    def accuracy_score(self, y_true, y_probas):
        y_pred = y_probas >= self.threshold
        return accuracy_score(y_true, y_pred)

    def recall_score(self, y_true, y_probas):
        y_pred = y_probas >= self.threshold
        return recall_score(y_true, y_pred)

    def precision_score(self, y_true, y_probas):
        y_pred = y_probas >= self.threshold
        return precision_score(y_true, y_pred)

    def specificity_score(self, y_true, y_probas):
        y_pred = y_probas >= self.threshold
        return recall_score(1 - y_true, 1 - y_pred)

    def f1_score(self, y_true, y_probas):
        y_pred = y_probas >= self.threshold
        return f1_score(y_true, y_pred)

    def TensorBoardCallback(self, log_dir=None):
        writer = SummaryWriter(log_dir)

        def callback(env):
            results = []
            for k, v in env.evaluation_result_list:
                results.append(v)

            writer.add_scalar("Accuracy/train", results[2], env.iteration)
            writer.add_scalar("ROC-AUC/train", 1 - results[3], env.iteration)
            writer.add_scalar("Log-loss/train", results[4], env.iteration)

            writer.add_scalar("Accuracy/val", results[5], env.iteration)
            writer.add_scalar("ROC-AUC/val", 1 - results[6], env.iteration)
            writer.add_scalar("Log-loss/val", results[7], env.iteration)

        return callback

    def custom_metrics(self, output, dtrain):
        labels = dtrain.get_label()
        preds = output > 0.5
        results = [
            ("Accuracy", accuracy_score(labels, preds)),
            ("ROC-AUC", 1 - roc_auc_score(labels, output)),
            ("Log-loss", log_loss(labels, output, eps=1e-7)),
        ]
        return results

    def compute_threshold_f_beta(
        self, threshold_metric, threshold_parameter, y_val, val_predictions
    ):

        nb_steps = 200
        best_threshold = 1

        col_true = y_val
        col_pred = val_predictions

        score_reached = 0

        list_thresholds = np.percentile(
            col_pred, q=[i * 100.0 / nb_steps for i in range(0, nb_steps - 1)]
        )

        for threshold in list_thresholds:
            binary_predicted_values = (col_pred >= threshold) * 1

            if threshold_metric == "f-beta":
                score = fbeta_score(
                    col_true, binary_predicted_values, beta=threshold_parameter
                )
            else:
                score = 0

            if score > score_reached:
                score_reached = score
                best_threshold = threshold

        return best_threshold

    def plot_interactive_threshold(self, y_true, y_probas, width=700, height=500):
        fpr, tpr, thresh_roc = roc_curve(y_true, y_probas)
        pre, rec, thresh_pr = precision_recall_curve(y_true, y_probas)
        df_roc = pd.DataFrame(
            {"Specificity": 1 - fpr, "Sensitivity/Recall": tpr}, index=thresh_roc
        )
        df_pr = pd.DataFrame(
            {"Precision": pre[:-1], "Sensitivity/Recall": rec[:-1]}, index=thresh_pr
        )
        df = df_roc.merge(
            df_pr,
            on="Sensitivity/Recall",
            how="inner",
            left_index=True,
            right_index=True,
        )
        df.index.name = "Thresholds"
        df.columns.name = "Metric"
        fig_thresh = px.line(
            df,
            title="Sensitivity, Specificity and Precision trade-offs",
            labels={"value": "Performance"},
            width=width,
            height=height,
            template="plotly_white",
            line_dash="Metric",
            render_mode="svg"
        )
        return fig_thresh

    def plot_importance(self, ax=None):
        ax = ax or plt.gca()
        self.non_zero_importance = (
            pd.DataFrame(
                {
                    "feature": self.X_train.columns,
                    "importance": self.models["event"].feature_importances_,
                }
            )
            .sort_values("importance", ascending=True)
            .loc[lambda x: x.importance > 0, "feature"]
            .values
        )
        plot_importance(trainer.models["event"], ax=ax)
        ax.set_yticklabels(self.non_zero_importance);
    
    def compute_shap(self):
        self.explainer = shap.TreeExplainer(self.models[self.target])
        
        
    def plot_shap(self, data, show=True):
        if not hasattr(self, "explainer"):
            self.compute_shap()
        self.shap_values = self.explainer.shap_values(data)
        shap.summary_plot(self.shap_values, data, show=show)
        
    def plot_calibration(self, ax=None, n_bins=10, figsize=(8, 8)):
        ax = ax or plt.gca()
        plot_y, plot_x = calibration_curve(self.y_test, self.probas.loc[:, self.target], n_bins=n_bins)
        plt.plot(plot_x, plot_y, marker='o', linewidth=1, label='Calibration curve')

        # reference line, legends, and axis labels
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        ax.set_title('Calibration plot (realiability curve)', fontsize=16)
        ax.set_xlabel('Predicted probability', fontsize=16)
        ax.set_ylabel('Outcome proportion', fontsize=16)
        plt.legend(fontsize=16)
