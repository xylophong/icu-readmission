import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, data, features, label, time_to_readmission=30):
        self.data = data
        self.label_ = label
        self.time_to_readmission_ = time_to_readmission
        self.features_list_ = list(features.keys())
        self.n_features_ = len(self.features_list_)
        self.features_types_ = features
        
        self.add_label()
        
        self.y = self.data.loc[:, self.label_]
        self.X = self.data.loc[:, [x for x in self.features_list_ if x != "full_count"]]
        if "full_count" in self.features_list_ :
            self.X["full_count"] = self.X.apply(lambda x: x.count(), axis=1)
        
    def split(self, test_size=0.20, shuffle=True, random_state=42):
        if test_size == 0.:
            return self.X, self.y 
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                train_test_split(
                    self.X, 
                    self.y, 
                    test_size=test_size, 
                    shuffle=shuffle,
                    random_state=random_state
                )
            )
            return self.X_train, self.X_test, self.y_train, self.y_test
        
    def fit(self, data, labels, scaler=None, imputing_strategy=None, fill_value=None):
        self.fitted_data = data.loc[:, self.features_list_]
        if imputing_strategy == "iterative":
            self.imputer = IterativeImputer()
            self.imputer.fit(self.fitted_data)
        elif imputing_strategy in ["mean", "median", "most_frequent", "constant"]:
            self.imputer = SimpleImputer(strategy=imputing_strategy, fill_value=fill_value)
            self.imputer.fit_transform(self.fitted_data)
        else:
            self.fitted_data = self.fitted_data.dropna()
        self.fitted_labels = labels.loc[self.fitted_data.index]
            
        
        self.scaler = scaler
        if self.scaler is not None:
            self.scaler = self.scaler.fit(self.fitted_data)
                
    def transform(self, data):
        self.transformed_data = data.loc[:, self.features_list_]
        if self.imputer:
            self.transformed_data = self.imputer.transform(data)
            self.transformed_data = pd.DataFrame(
                self.transformed_data, columns=self.features_list_
            )
        else:
            self.transformed_data = data.dropna()
        
        for feature in self.features_list_:
            self.transformed_data.loc[:, feature] = (
                self.transformed_data.loc[:, feature].astype(self.features_types_[feature])
            )
            
        self.continuous_features_ = list(
            self.transformed_data.select_dtypes(["int32", "int64", "float32", "float64"]).columns
        )
        self.categorical_features_ = list(
            self.transformed_data.select_dtypes(["category"]).columns
        )
        
        if self.categorical_features_:
            self.transformed_data = pd.get_dummies(
                self.transformed_data, columns=self.categorical_features_
            )
        if self.scaler:
            self.transformed_data = self.scaler.transform(self.transformed_data)
            self.transformed_data = pd.DataFrame(
                self.transformed_data, columns=self.features_list_
            )
        return self.transformed_data
            
    def fit_transform(self, data, labels, scaler=None, imputing_strategy=None, fill_value=None):
        self.fit(data, labels, scaler, imputing_strategy, fill_value)
        return self.transform(data)
    
    def plot_corr_heatmap(self, data=None, features_list=None, figsize=(15, 12)):
        if data is None:
            data = self.X
        if features_list is None:
            features_list = self.features_list_
        plt.figure(figsize=figsize)
        sns.heatmap(data[features_list].corr(), annot=False);
        
    def plot_dist_label(self, data=None, label=None, figsize=None, nrow=None, ncol=3, height=5):
        if data is None:
            data = pd.concat([self.X, self.y], axis=1)
        if label is None:
            label = self.label_
        if figsize is None:
            figsize = (15, int(height * (self.n_features_ / ncol)))
        if nrow is None:
            nrow = int(np.ceil(self.n_features_ / ncol))
        fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
        for i, feature in enumerate(self.features_list_):
            self.distplot_with_hue(
                data, feature, label, height=height, 
                hist=False, kde=True, ax=ax[int(i/ncol),i%ncol]
            )
            plt.close(2)
    
    def add_label(self):
        if self.label_ == "death_during_visit":
            self.data[self.label_] = (
                self.data["discharge_to_concept_id"] == 4216643
            )
        elif self.label_ == "readmission":
            self.data["visit_detail_rank"] = self.data.groupby("visit_occurrence_id").cumcount()
            self.data = self.data.sort_values(["person_id", "visit_start_date"], axis=0)
            self.data["time_since_discharge"] = (
                self.data["visit_start_date"]
                - self.data.groupby("person_id")["visit_end_date"].shift()
            ).dt.days
            self.data["time_to_readmission"] = (
                self.data.groupby("person_id")["time_since_discharge"].shift(-1)
            )
            self.data["time_to_death"] = (
                self.data["death_date"] - self.data["visit_end_date"]
            ).dt.days
            self.data[self.label_] = (
                (self.data["time_to_readmission"] <= self.time_to_readmission_)
                | (self.data["time_to_death"] <= self.time_to_readmission_)
            )
            self.data[self.label_] = self.data[self.label_].fillna(False)
            self.data = self.data[self.data["discharge_to_concept_id"] != 4216643]
        else:
            raise NotImplementedError("This label is not implemented yet")
            
    def distplot_with_hue(self, data=None, x=None, hue=None, row=None, col=None, 
                          height=None, aspect=1, legend=True, **kwargs):
        _, bins = np.histogram(data[x].dropna())
        g = sns.FacetGrid(data, hue=hue, row=row, col=col, height=height, aspect=aspect)
        g.map(sns.distplot, x, **kwargs)