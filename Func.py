import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import mstats


#Function to calculate IV
def calc_iv(df, feature, target, pr = False):
    '''
    Function to calculate Information value for Feature
    Input
    df: DataFrame
    Feature: Feature which need to be calculated IV
    Target: Predited variable
    '''
    lst = []
    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):    
        val = list(df[feature].unique())[i]   
        lst.append([feature,                                                           
                    val,                                                               
                    df[df[feature] == val].count()[feature],                           
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],     
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]      
                    ])

    data = pd.DataFrame(lst, columns = ['Variable', 'Value', 'All', 'Good', 'Bad'])   
    data['Share'] = data['All']/data['All'].sum()     
    data['Bad Rate'] = data['Bad']/data['All']
    data['Distribution Good'] = (data['All'] - data['Bad'])/(data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad']/ data['Bad'].sum()

    #WOE = ln(Distribution of Good/ Distribution of Bad)
    data['WOE'] = np.log(data['Distribution Good']/ data['Distribution Bad'])

    data = data.replace({'WOE': {np.inf: 0, -np.inf: 0}})

    #IV = WOE*(Distribution of Good - Distribution of Bad)
    data['IV'] = data['WOE']* (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by = ['Variable', 'Value'], ascending = [True, True])
    data.index = range(len(data.index)) 

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())

    iv = data['IV'].sum()   #IV of feature is sum IV of all attributes
    print('This variable IV is: ',iv)
    print(df[feature].value_counts())
    return iv, data  



#==============================================================================================
def convert_dummy(df, feature, rank = 0):
    '''
    Convert category variable into Dummy variable (N values will generate N-1 dummy features)
    df: DataFrame
    feature: Feature name
    rank = 0: Drop the biggest count feature with value[0]
    '''
    pos = pd.get_dummies(df[feature], prefix = feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest], axis = 1, inplace = True)
    #df.drop([feature], axis = 1, inplace = True)
    df = df.join(pos)
    return df


#==============================================================================================
def get_category(df, col, binsnum, labels, qcut = False):
    ''' 
    Function to binning feature into multiple bin
    df: DataFrame
    Col: Feature Name
    binsum: the number of bin or the range of mean, ex: [1,30,60,70,100,..]
    labels: Name of bin, by default labels = None
    qcut: If qcut = True, cut by quantitile, if not cut by equal length
    '''
    if qcut:
        localdf = pd.qcut(x = df[col], q = binsnum, labels = labels)   #quantile cut
    else:
        localdf = pd.cut(x = df[col], bins = binsnum, labels = labels) #equal- length cut
    
    localdf = pd.DataFrame(localdf)
    name = 'gp' + '_' + col
    localdf[name] = localdf[col]
    df = df.join(localdf[name])
    df[name] = df[name].astype(object)
    return df
        

#==============================================================================================
def plot_confusion_matrix (cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float')/ cm.sum(axis = 1)[:,np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j]),fmt)
        horizontalalignment = "center",
        color = "white" if cm[i,j] > thresh else "black"

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#==============================================================================================
def Winsorize_Feature(data, col, limits=None):
    ''' 
    Hàm xử lý outlier cho biến numerical
    
    Input:
    data: DataFrame cần xử lý
    feature: feature chứa outlier cần xử lý
    limits: Chọn phần trăm cắt (cutoff percentiles), ví dụ 5% ở cả hai đầu (2.5% ở mỗi đầu)
    '''
    if limits is None:
        limits = [0.025, 0.025]
    data[col] = mstats.winsorize(data[col], limits)
    return data

#==============================================================================================
def WOE_transform(woe_data, col, iv_table):
    # Create a DataFrame containing 'Value' and 'WOE' columns
    woe_df = iv_table[['Value', 'WOE']]

    name='woe_'+ col

    # Rename 'WOE' column for clarity
    woe_df.rename(columns={'WOE': name}, inplace=True)

    # Merge 'woe_data' with 'woe_df' based on 'col'
    woe_data = pd.merge(woe_data, woe_df, how='left', left_on=col, right_on='Value')

    # Drop the 'Value' column
    woe_data.drop(columns='Value', inplace=True)

    return woe_data


def check():
    print("Hello Alice in Wonderland!")
    