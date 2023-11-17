"""
Description:
This script runs a Logistic Regression model for survive in Titanic data.  

Author: Saeideh Kamgar [saeideh.kamgar@gmail.com]
Date: 14 Nov 2023
"""


import numpy as np 
import pandas as pd 

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14) #the default font size for text in Matplotlib plots
import seaborn as sns
##sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
import matplotlib.cm as cm

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.model_selection import GridSearchCV



#import warnings
#warnings.simplefilter(action='ignore')


# ----------------------------------------------------------------------------------------

def logistic_reg(
    working_path,
    train_df,
    test_df, 
    verbose=True,
    test_size=0.2,
    random_state=2023
):
    """
    Purpose:
    Delving into logistic regression through an in-depth examination, fine-tuning diverse parameters to discover the optimal model configuration
    
    Args::
        working_path (str): Working directory, contains the input files
        train_df (float): 
        test_df (float):
        ab_data_fname (str): filename of csv file of user info
        verbose (boolean): if True, return more reports on screen [Default: False]
        random_state (int): seed to generate the random number [Default: 2023]

    Example:
    Modify the working parameters and run the main script

    """

# Taking care of inputs/outputs
    # ------------------------------------------------------------------------------------

    # check if working_path exists
    assert Path(working_path).is_dir(), f'No {working_path} directory exists!'

    # Define paths for data, models, and figures
    data_path = Path(working_path) / "data"
    Path(data_path).mkdir(parents=True, exist_ok=True)

    model_path = Path(working_path) / "model"
    Path(model_path).mkdir(parents=True, exist_ok=True)

    fig_path = Path(working_path) / "figures"
    Path(fig_path).mkdir(parents=True, exist_ok=True)



    # Loading data into pandas DF
    train_df = data_path / train_df
    assert Path(train_df).is_file(), f'No transactions file named {train_df} found in {data_path}'
    test_df = data_path / test_df
    assert Path(test_df).is_file(), f'No labels file named {test_df} found in {data_path}'
    train_df = pd.read_csv(train_df)
    test_df = pd.read_csv(test_df)


    if verbose:
        # Display trans dataset information
        print(train_df.info())
        print(train_df.shape)
        print(train_df.head(20))
        print(train_df.apply(lambda x: x.nunique()))

        # Display label dataset information
        print(test_df.info())
        print(test_df.head(20))
        print(test_df.apply(lambda x: x.nunique()))
        print(test_df.isnull().sum())
  

# Data pre-processing: [part I]
    #Train DataFrame
    if verbose:
        print('The number of samples into the train data frame is {}.'.format(train_df.shape[0]))
    #remove rows which are duplicate values in the 'PassengerId' column.
    df1 = train_df.drop_duplicates(subset= 'PassengerId', keep= False)
    if verbose:
        print(df1.shape)

        

    #Test DataFrame
    if verbose:
        print('The number of samples into the test data frame is {}.'.format(test_df.shape[0]))
    #remove rows which are duplicate values in the 'PassengerId' column.
    df2 = test_df.drop_duplicates(subset= 'PassengerId', keep= False)
    if verbose:
        print(df2.shape)
        

    #I. Data Quality & Missing Value Assessment¶
    # Check missing values in train data
    if verbose:
        print(train_df.isnull().sum())
        
    # AGE: missing values
    if verbose:
        print('Percent of missing data for "Age" is %0.2f%%' % ((train_df['Age'].isnull().sum() / train_df.shape[0]) * 100))

    # Density plot of AGE
    ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
    train_df["Age"].plot(kind='density', color='teal')
    ax.set(xlabel='Age')
    plt.xlim(-10,85)
    figure_name = "AGE_plot.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()
 
    # Cabin: Missing Values
    if verbose:
        print('Percent of missing data for "Cabin" is %0.2f%%' % ((train_df['Cabin'].isnull().sum() / train_df.shape[0]) * 100))
    # Missing rate in Cabin is high, around 70%, So we omit Cabin from the furthur analysis. 
      

    # Embarked: Missing Values¶
    if verbose:
        print('Percent of missing data for "Embarked" is %0.2f%%' % ((train_df['Embarked'].isnull().sum() / train_df.shape[0]) * 100))
    if verbose:
        print(train_df['Embarked'].value_counts())
    sns.countplot(x='Embarked', data=train_df, palette='Set2')
    figure_name = "Embarked_plot.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()

    # Treatment of Missing values
    train_completed = train_df.copy()
    # Age dist is skewed and the missing rate is 20%, so median is used for imputation instead of mean
    train_completed["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
    #Embarked has low rate of missing, 2 percent, so the missing values is imputed by mode value. Note: Embarked is a categorical variable.   
    train_completed["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
    train_completed.drop('Cabin', axis=1, inplace=True)

    # Check if the completed datasets is full, without any NA values.
    if(train_completed.isnull().sum().max())>0: 
        print(
            "WARNING: Train Data is incomplete"
        )
        
    # Plot the available case of AGE and completed AGE in one figure
    plt.figure(figsize=(15,8))
    ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
    train_df["Age"].plot(kind='density', color='teal')
    ax = train_completed["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
    train_completed["Age"].plot(kind='density', color='orange')
    ax.legend(['Incomplete Age', 'Completed Age'])
    ax.set(xlabel='Age')
    plt.xlim(-10,85)
    figure_name = "AGE_NA_imp_plot.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()


   # Create binary categorical variable for traveling alone: 
   # Creating a new variable based on a condition on two other variables and then removing these two variables.
    train_completed['TravelAlone']=np.where((train_completed["SibSp"]+train_completed["Parch"])>0, 0, 1)
    train_completed.drop('SibSp', axis=1, inplace=True) #Remove SibSp 
    train_completed.drop('Parch', axis=1, inplace=True)

   # Convert some variables to dummy variables   
    training=pd.get_dummies(train_completed, columns=["Pclass","Embarked","Sex"])
    training.drop('Sex_female', axis=1, inplace=True)
    training.drop('PassengerId', axis=1, inplace=True)
    training.drop('Name', axis=1, inplace=True)
    training.drop('Ticket', axis=1, inplace=True)

    final_train = training

# Data pre-processing: [part II] apply the same changes to the test data.
# IMPORTANT: Impute NA values of Test data, using the imputed value or imputed model from train data. For example, using the mean or median or mode of train data. 
    test_df.isnull().sum()
    test_completed = test_df.copy()
    test_completed["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
    test_completed["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
    test_completed.drop('Cabin', axis=1, inplace=True)

    test_completed['TravelAlone']=np.where((test_completed["SibSp"]+test_completed["Parch"])>0, 0, 1)

    test_completed.drop('SibSp', axis=1, inplace=True)
    test_completed.drop('Parch', axis=1, inplace=True)

    testing = pd.get_dummies(test_completed, columns=["Pclass","Embarked","Sex"])
    testing.drop('Sex_female', axis=1, inplace=True)
    testing.drop('PassengerId', axis=1, inplace=True)
    testing.drop('Name', axis=1, inplace=True)
    testing.drop('Ticket', axis=1, inplace=True)

    final_test = testing
    
    

# Exploratory Data Analysis: Check if there is any relation between features and different categories of label ('Survived', 'Died').
# AGE vs Survived and Died
    plt.figure(figsize=(15,8))
    ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", fill=True)
    sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", fill=True)
    plt.legend(['Survived', 'Died'])
    plt.title('Density Plot of Age for Surviving Population and Deceased Population')
    ax.set(xlabel='Age')
    plt.xlim(-10,85)
    figure_name = "AGE_vs_label.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()

# AGE vs Survived
    plt.figure(figsize=(20,8))
    avg_survival_byage = final_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
    g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")
    figure_name = "AGE_vs_Survived.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()  

# There is a signifincant difference between Survived and Died for minor age. 
# Add a new dummy variable named IsMinor. 
    final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)
    final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)

# Fare vs label
    plt.figure(figsize=(15,8))
    ax = sns.kdeplot(final_train["Fare"][final_train.Survived == 1], color="darkturquoise", fill=True)
    sns.kdeplot(final_train["Fare"][final_train.Survived == 0], color="lightcoral", fill=True)
    plt.legend(['Survived', 'Died'])
    plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
    ax.set(xlabel='Fare')
    plt.xlim(-20,200)
    figure_name = "fare_vs_label.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()  

# Passenger Class¶ vs Survived
    sns.barplot(x='Pclass', y='Survived', data=train_df, color="darkturquoise")
    figure_name = "Pclass_vs_Survived.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()  

# Embarked Port vs Survived
    sns.barplot(x='Embarked', y='Survived', data=train_df, color="teal")
    figure_name = "Embarked_vs_Survived.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()  

# Traveling Alone or With Family vs Survived
    sns.barplot(x='TravelAlone', y='Survived', data=final_train, color="mediumturquoise")
    figure_name = "Travel_vs_Survived.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()

# Sex vs Survived
    sns.barplot(x='Sex', y='Survived', data=train_df, color="aquamarine")
    figure_name = "Sex_vs_Survived.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()
    # ------------------------------------------------------------------------------------


# Logistic Regression
#print(final_train.head())
#print(final_test.head())

## Feature selection
### 1. Recursive feature elimination



    cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
    X = final_train[cols]
    y = final_train['Survived']
    # Standardize the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Build a logreg and compute the feature importances
    # set the max_iter=1000 for convergance of the process of finding the optimal coefficients that minimize the logistic loss function.
    model = LogisticRegression(max_iter=1000)
    # create the RFE model and select 8 attributes
    rfe = RFE(model, n_features_to_select=8)
    rfe = rfe.fit(X, y)
    # summarize the selection of the attributes
    print('Selected features: %s' % list(X.columns[rfe.support_]))


### 2. RFE object and a cross-validated score.
    rfecv = RFECV(estimator=LogisticRegression(max_iter=1000), step=1, cv=10, scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X.columns[rfecv.support_]))

    # Plot number of features VS. cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross-validation score (number of correct classifications)")
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
    plt.xlim(-10, 85)  # Set x-axis limits
    figure_name = "feature_CV.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()


    Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 
                        'Embarked_S', 'Sex_male', 'IsMinor']
    X = final_train[Selected_features]

    plt.subplots(figsize=(8, 5))
    sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
    figure_name = "X_corr_fig.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()


# Model Evaluation
## check the performance of a model by calculationg the Testing accuracy

# 1. Model evaluation: simple train/test split
    # create X (based on the selected features )and y 
    X = final_train[Selected_features]
    y = final_train['Survived']

    # Perform simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # check classification scores of logistic regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_pred_proba = logreg.predict_proba(X_test)[:, 1]
    [fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
    print('Train/Test split results:')
    print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
    print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
    print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

    idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

    plt.figure()
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    figure_name = "ROC_curve.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()

    print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
        "and a specificity of %.3f" % (1-fpr[idx]) + 
        ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))


# GridSearchCV evaluating using multiple scorers simultaneously¶
    X = final_train[Selected_features]

    param_grid = {'C': np.arange(1e-05, 3, 0.1)}
    scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}

    gs = GridSearchCV(LogisticRegression(), return_train_score=True,
                    param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy')

    gs.fit(X, y)
    results = gs.cv_results_

    print('='*20)
    print("best params: " + str(gs.best_estimator_))
    print("best params: " + str(gs.best_params_))
    print('best score:', gs.best_score_)
    print('='*20)

    plt.figure(figsize=(10, 10))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)

    plt.xlabel("Inverse of regularization strength: C")
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    ax.set_xlim(0, param_grid['C'].max()) 
    ax.set_ylim(0.35, 0.95)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_C'].data, dtype=float)

    for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]
            
        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    figure_name = "allgrid_C.png"
    plt.savefig(str(fig_path / figure_name))  # Save as PNG format
    plt.close()

    # Assuming you have performed the GridSearchCV as in your first code snippet
    # gs.fit(X, y)

    # Get the best estimator from the GridSearchCV results
    best_logreg = gs.best_estimator_

    # Make predictions on the final_test dataset
    final_test_predictions = best_logreg.predict(final_test[Selected_features])

    # Create a DataFrame for submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_test_predictions
    })

    # Save the submission DataFrame to a CSV file
    submission.to_csv('submission.csv', index=False)
    print(submission.tail())


if __name__ == "__main__":
    # Set the input parameters (Refer to the README file for the description of each parameter)
    # -------------------------------------------------------------------------------------------------------------
    working_path = "/Users/saeideh/job_seek/Data_Science/UK_2023/Nov2023/Kaggle_project/Survival_prediction_Logistic_Reg"
    train_df = "train.csv"
    test_df = "test.csv"
    verbose = False
    test_size=0.2
    random_state = 2023

    # Call the process_data function with the argument values
    logistic_reg(
        working_path,
        train_df,
        test_df,
        verbose=verbose,
        test_size=0.2,
        random_state=2023
    )


