import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as cnf
from itertools import combinations
from sklearn.model_selection import GridSearchCV, cross_validate

def check_df(dataframe, head=5,non_numeric=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### DESCRIBE #####################")
    print(dataframe.describe())
    
    for col in dataframe.columns:
        if dataframe[col].isna().sum() <= 0:
            if dataframe[col].nunique() > 20:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()}\n')
            else:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()} --- unique: {dataframe[col].unique()}\n')
        else:
            if dataframe[col].nunique() > 20:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()} --- nan: {dataframe[col].isna().sum()}\n')
            else:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()} --- unique: {dataframe[col].unique()} --- nan: {dataframe[col].isna().sum()}\n')
    
    if non_numeric:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    
def grab_col_names(dataframe, cat_th=10, car_th=20,p=False):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    if p:
        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols,cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False,q1 = 0.01,q3 = 0.99):
    low, up = outlier_thresholds(dataframe, col_name,q1=q1,q3 = q3)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
    
def num_summary(dataframe, numerical_col, plot=False):
    """
        Numerik kolonlar input olarak verilmelidir.
        Sadece ekrana cikti veren herhangi bir degeri return etmeyen bir fonksiyondur.
        For dongusuyle calistiginda grafiklerde bozulma olmamaktadir.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
      
def plot_distributions(dataframe, columns,kde=False, log_transform=False, label_angle=0, 
                       figsize = (8,3) , order_cats= False, target_pie=False, alert=False,target=cnf.target): 

    if alert == True:
        pie_palette = cnf.alert_palette
    else:
        pie_palette = cnf.sequential_palette
        
    if target_pie == True:
#         colors = ['#ff6666', '#468499', '#ff7f50', '#ffdab9', 
#           '#00ced1', '#ffff66','#088da5','#daa520',
#           '#794044','#a0db8e','#b4eeb4','#c0d6e4','#065535','#d3ffce']
# fig1, ax1 = plt.subplots(figsize=(14,7))

# ax1.pie(data.enflasyon,labels=data.Tarih,colors=colors, autopct='%1.1f%%');
        ax = dataframe[columns].value_counts().plot.pie(autopct='%1.1f%%',
                                              textprops={'fontsize':10},
                                              colors=cnf.muted_palette
                                              ).set_title(f"{target} Distribution")
        plt.ylabel('')
        plt.show()

    else:
        for col in columns:
            if log_transform == True:
                x = np.log10(dataframe[col])
                title = f'{col} - Log Transformed'
            else:
                x = dataframe[col]
                title = f'{col}'
            
            if order_cats == True:
                
                print(pd.DataFrame({col: dataframe[col].value_counts(),
                            "Ratio": 100 * dataframe[col].value_counts() / len(dataframe)}))
            
                print("##########################################")
                
                print(f"NA in {col} : {dataframe[col].isnull().sum()}")
                
                print("##########################################")

                labels = dataframe[col].value_counts(ascending=False).index
                values = dataframe[col].value_counts(ascending=False).values
                
                plt.subplots(figsize=figsize)
                plt.tight_layout()
                plt.xticks(rotation=label_angle)
                sns.barplot(x=labels,
                            y=values,
                            palette = cnf.sequential_palette)
                        
            else:   
            
                quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
                print(dataframe[col].describe(quantiles).T)

                plt.subplots(figsize=figsize)
                plt.tight_layout()
                plt.xticks(rotation=label_angle)
                sns.histplot(x,
                        bins=50,
                        kde=kde,
                        color = cnf.sequential_palette[0])

    
            plt.title(title)
            plt.show()

def numcols_target_corr(dataframe, num_cols,target = cnf.target):
    numvar_combinations = list(combinations(num_cols, 2))
    
    for item in numvar_combinations:
        
        plt.subplots(figsize=(8,4))
        sns.scatterplot(x=dataframe[item[0]], 
                        y=dataframe[item[1]],
                        hue=dataframe[target],
                        palette=cnf.bright_palette
                       ).set_title(f'{item[0]}   &   {item[1]}')
        plt.grid(True)
        plt.show()            
            
def numeric_variables_boxplot(df, num_cols, target=None, alert=False):
    
    if alert == True:
        palette = cnf.alert_palette
    else:
        palette = cnf.bright_palette
        
    if target == None:
        
        fig, [ax1,ax2,ax3,ax4] = plt.subplots(1,4, figsize=(7,3))

        for col, ax, i in zip(num_cols, [ax1,ax2,ax3,ax4], range(4)):
            sns.boxplot(df[col], 
                        color=palette[i], 
                        ax=ax
                        ).set_title(col)
            
        for ax in [ax1,ax2,ax3,ax4]:
            ax.set_xticklabels([])
    else:
        for col in num_cols:
            plt.subplots(figsize=(7,3))
            sns.boxplot(x=df[target], 
                                y=df[col],
                                hue=df[target],
                                dodge=False, 
                                fliersize=3,
                                linewidth=0.7,
                                palette=palette)
            plt.title(col)
            plt.xlabel('')
            plt.ylabel('')
            plt.xticks(rotation=45)
            plt.legend('',frameon=False)

    plt.tight_layout()
    plt.show()
    
def plot_categorical_data(dataframe, x, hue, title='', label_angle=0):
    """
    Kategorik veri görselleştirmesi için alt grafikleri çizen bir fonksiyon. 
    """
    # Alt grafikleri yan yana düzenleme
    fig, ax = plt.subplots(1, figsize=(8, 3))

    # Grafik 1
    sns.countplot(data=dataframe, x=x, hue=hue, ax=ax, palette='husl')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)
    ax.legend(prop={'size': 10})

    # Grafikleri göster
    plt.tight_layout()
    plt.xticks(rotation=label_angle)
    plt.show(block=True)
    
def sample_sub(ypred):
  sample = pd.read_csv("sample_submission.csv")

  submission = pd.DataFrame({"PassengerId": sample["PassengerId"],
                             "Transported": ypred})
  submission["Transported"] = submission["Transported"].astype(bool)                          
  submission.to_csv("tekrar23.csv",index=False)
  
def plot_avg_numvars_by_target(dataframe, num_cols,agg='mean', round_ndigits=1, alert=False):
    
    if alert == True:
        palette = cnf.alert_palette
    else:
        palette = cnf.sequential_palette
        
    for col in num_cols:
        
        if agg == 'max':
            col_grouped = dataframe.groupby(cnf.target)[col].max().reset_index().sort_values(ascending=False,by=col)
        elif agg == 'min':
            col_grouped = dataframe.groupby(cnf.target)[col].min().reset_index().sort_values(ascending=True,by=col)
        elif agg == 'sum':
            col_grouped = dataframe.groupby(cnf.target)[col].sum().reset_index().sort_values(ascending=False,by=col)
        elif agg == 'std':
            col_grouped = dataframe.groupby(cnf.target)[col].std().reset_index().sort_values(ascending=False,by=col)
        else:
            col_grouped = dataframe.groupby(cnf.target)[col].mean().reset_index().sort_values(ascending=False,by=col)
        
        plt.subplots(figsize=(6,3))
        ax = sns.barplot(x=col_grouped[cnf.target], 
                     y=col_grouped[col], 
                     width=0.8,
                     palette=palette,
                     errorbar=None)
        ax.set_yticklabels([])  
    
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2., 
                    p.get_height(), 
                    round(p.get_height(),ndigits=round_ndigits), 
                    fontsize=10, color='black', ha='center', va='top')
            
        plt.xlabel('')  
        #plt.ylabel('')
        plt.xticks(rotation=45)
        plt.title(f'{agg} {col} - {cnf.target}')
        plt.show()   
        
def plot_avg_numvars_by_catvars(dataframe, agg='mean', num_col='Age',cat_cols=[]):
    
    for col in cat_cols:
        
        if agg == 'max':
            col_grouped = dataframe.groupby(col)[num_col].max().reset_index().sort_values(ascending=False,by=num_col)
        elif agg == 'min':
            col_grouped = dataframe.groupby(col)[num_col].min().reset_index().sort_values(ascending=True,by=num_col)
        elif agg == 'sum':
            col_grouped = dataframe.groupby(col)[num_col].sum().reset_index().sort_values(ascending=False,by=num_col)
        elif agg == 'std':
            col_grouped = dataframe.groupby(col)[num_col].std().reset_index().sort_values(ascending=False,by=num_col)
        else:
            col_grouped = dataframe.groupby(col)[num_col].mean().reset_index().sort_values(ascending=False,by=num_col)
        
        plt.subplots(figsize=(6,3))
        ax = sns.barplot(x=col_grouped[col], 
                     y=col_grouped[num_col], 
                     width=0.8,
                     palette=cnf.sequential_palette,
                     errorbar=None)
        ax.set_yticklabels([])  
    
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2., 
                    p.get_height(), 
                    round(p.get_height(),ndigits=1), 
                    fontsize=10, color='black', ha='center', va='top')
            
        plt.xlabel('')  
        #plt.ylabel('')
        plt.xticks(rotation=45)
        plt.title(f'{agg} {num_col} - {col}')
        plt.show(block=True)        
        
def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    """
    Grid search ile daha önce belirlenen parametreler ile hiperparametre optimizasyonu yapılıyor.
    """
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in cnf.classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()