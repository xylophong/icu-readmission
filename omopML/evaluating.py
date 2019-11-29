import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
from scipy import interp
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, auc, roc_auc_score, roc_curve, log_loss, f1_score, precision_score, recall_score, make_scorer
)

class ModelEvaluator:
    """
    A class to compute a list of metrics to evaluate a trained model
    ...

    Methods
    -------
    compute_metrics
        Computes the required metrics and saves the results in a dictionary
    print_metrics
        Prints the computed metrics; only available if the compute_metrics was
        called beforehand
    plot_roc
        Plots the Receiver operating characteristic (ROC) curve; only available
        if AUC was part of the computed metrics
    """
    
    def __init__(self, model, X_test, y_test, metrics_dict, threshold=0.5):
        """
        Parameters
        ----------
        model : 
            A model implemented with a `predict` method similar to Scikit-learn's
            conventions
        X_test : numpy.ndarray or pandas.DataFrame
            The testing set
        y_test : numpy.ndarray or pandas.Series
            The testing set labels
        metrics_dict : dict
            A dictionary containing the metrics to be computed, in the following
            form: {"metrics_name": metrics_function}
        threshold : float
            A float indicating the probability threshold over which a sample is
            classified in the corresponding category
        """
        
        self.model = model
        self.threshold = threshold
        self.y_test_ = y_test
        
        if hasattr(self.model, "predict_proba"):
            self.y_pred_proba_ = self.model.predict_proba(X_test)
            if hasattr(self.model, "history"): #hack to identify keras models; to improve
                self.y_pred_ = self.model.predict(X_test) > self.threshold
            else:
                self.y_pred_ = self.model.predict_proba(X_test)[:,1] > self.threshold
        else:
            if hasattr(self.model, "BL"): #hack to identify the DeepSuperLearner; to improve
                self.y_pred_ = self.model.predict(X_test)[:,1] > self.threshold
            elif hasattr(self.model, "history"):
                self.y_pred_ = self.model.predict(X_test) > self.threshold
            else:
                self.y_pred_ = self.model.predict(X_test)
        
            
        self.metrics_ = metrics_dict
        self.computed_metrics_ = {}
        
    def compute_metrics(self):
        """Computes the required metrics and saves the results in a dictionary"""
        
        for metric in self.metrics_:
            if metric == "neg_log_loss":
                self.computed_metrics_[metric] = (
                    self.metrics_[metric](self.y_test_, self.y_pred_proba_)
                )
                continue
            elif metric == "roc_auc":
                if hasattr(self, "y_pred_proba_"):
                    self.roc_curve_ = roc_curve(self.y_test_, self.y_pred_proba_[:,1])
                else:
                    self.roc_curve_ = roc_curve(self.y_test_, self.y_pred_)
                self.computed_metrics_[metric] = (
                    self.metrics_[metric](self.roc_curve_[0], self.roc_curve_[1])
                )
                continue
            self.computed_metrics_[metric] = (
                self.metrics_[metric](self.y_test_, self.y_pred_)
            )
    
    def print_metrics(self):
        """Prints the computed metrics; only available if the compute_metrics was
        called beforehand
        
        Raises
        ------
        NameError
            If metrics have not been computed yet via `compute_metrics`
        """
        
        if self.computed_metrics_:
            for metric in self.computed_metrics_:
                print(
                    "{}: {:.4f}".format(metric, self.computed_metrics_[metric])
                )
        else:
            raise NameError("Metrics have to be computed first")
            
    def cross_val(self, X_train, y_train, cv=10, plot_auc=False, figsize=(16, 9)):
        """Computes stratified K-Fold cross-validation of the metrics provided 
        to the class and returns a dictionnary with the results for each fold ; 
        if the `plot_auc` argument is set to `True`, then cross-validation will only 
        be run with AUC as the metric, and returns a plot with ROC curves for each 
        fold
        
        Parameters
        ------
        X_train : numpy.ndarray or pandas.DataFrame
            Data to train the cross-validation on
        y_train : numpy.ndarray or pandas.Series
            Labels for `X_train`
        cv : int
            Number of folds for cross-validation
        plot_auc : bool
            Indication to plot ROC curves for each fold
        figsize : int tuple
            Plot size
        """
        if plot_auc == True:
            cv = StratifiedKFold(n_splits=cv)
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            i = 0
            plt.figure(figsize=figsize)
            for train, test in cv.split(X_train, y_train):
                probas_ = self.model.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
                fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
                i += 1

            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                         label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color='b',
                        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
                        lw=2, alpha=.8)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve')
            plt.legend(loc="lower right")
        else:
            self.cv_scores_ = {}
            for metric in self.metrics_:
                if metric == "precision":
                    self.cv_scores_[metric] = cross_val_score(
                        self.model, X_train, y_train, cv=cv, scoring=self.custom_precision_score
                    )
                elif metric == "recall":
                    self.cv_scores_[metric] = cross_val_score(
                        self.model, X_train, y_train, cv=cv, scoring=self.custom_recall_score
                    )
                elif metric == "accuracy":
                    self.cv_scores_[metric] = cross_val_score(
                        self.model, X_train, y_train, cv=cv, scoring=self.custom_accuracy_score
                    )
                elif metric == "f1_score":
                    self.cv_scores_[metric] = cross_val_score(
                        self.model, X_train, y_train, cv=cv, scoring=self.custom_f1_score
                    )
                else:
                    self.cv_scores_[metric] = cross_val_score(
                        self.model, X_train, y_train, cv=cv, scoring=metric
                    )
            return self.cv_scores_
    
    def custom_precision_score(self, estimator, X, y):
        y_pred = self.model.predict_proba(X)[:,1] > self.threshold
        return(precision_score(y, y_pred))
    
    def custom_recall_score(self, estimator, X, y):
        y_pred = self.model.predict_proba(X)[:,1] > self.threshold
        return(recall_score(y, y_pred))
    
    def custom_accuracy_score(self, estimator, X, y):
        y_pred = self.model.predict_proba(X)[:,1] > self.threshold
        return(accuracy_score(y, y_pred))
    
    def custom_f1_score(self, estimator, X, y):
        y_pred = self.model.predict_proba(X)[:,1] > self.threshold
        return(f1_score(y, y_pred))
            
    def plot_roc(self, figsize=(8, 8)):
        """Plots the Receiver operating characteristic (ROC) curve; only available
        if AUC was part of the computed metrics
        
        Raises
        ------
        NameError
            If AUC is not part of the computed metrics
        """
        if self.roc_curve_:
            plt.figure(figsize=figsize)
            plt.plot(
                self.roc_curve_[0], 
                self.roc_curve_[1], 
                label='ROC curve (area = %0.3f)' % self.computed_metrics_["roc_auc"]
            )
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
        else:
            raise NameError("AUC has to be computed first")
            
    def plot_calibration(self, n_bins=10, figsize=(8, 8)):
        plot_y, plot_x = calibration_curve(self.y_test_, self.y_pred_proba_[:,1], n_bins=n_bins)
        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(plot_x, plot_y, marker='o', linewidth=1, label='Calibration curve')

        # reference line, legends, and axis labels
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        ax.set_title('Calibration plot (realiability curve)')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Outcome proportion')
        plt.legend()
        
        