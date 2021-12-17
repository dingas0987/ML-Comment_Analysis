"""Imports for Project"""
"""Install nltk, langdetect"""
import re
import json
import nltk
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
from matplotlib.pyplot import figure
from collections import Counter
from langdetect import detect
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def df_from_json(file_name):
    
    """
    Retrieve pandas dataframe from json file, specific to json Luke sent
    
    2941629 lines in "comment_code_data2.json", reads 2936485, loss of 5144
    
    arge file_name: name of json file as str
    
    returns: pandas df
    """
    d = []
    with open(file_name, errors="ignore") as f:
        for line in f:
            try:
                d.append(json.loads(line))
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                break
            except: #Continue through errors
                continue

    data = []
    for x in d:
        try:
            data.append({**x, **x.pop('code')})
        except:
            continue

    df = pd.DataFrame(data)
    
    return df

""" Cleaning Comments Helper Functions """
def remove_punc(x):
    #To edit, edit [^a-z@-Z0-9\s]+ section
    ans = re.sub("[^a-z@-Z0-9\s]+", " ", x)
    return ans

def lowercase(x):
    ans = x.lower()
    return ans

def remove_stopwords_list(x, stop_words):
    x = x.split()
    ans = [w for w in x if not w in stop_words]
    return ans

def remove_stopwords_string(x, stop_words):
    x = x.split()
    ans = [w for w in x if not w in stop_words]
    ans = " ".join(ans)
    return ans

def rev_keywords(x, keywords):
    x = x.split()
    ans = [w for w in x if w in keywords]
    ans = " ".join(ans)
    return ans

def detect_lang(x):
    try:
        ans = detect(x)
    except KeyboardInterrupt:
        exit()
    except:
        return "unknown"
    return ans

def clean_comments(df):
    """
    Clean dataframe comments to just lowercase text and numbers.This
    can be adjusted within the remove_punc function in the [^a-z@-Z0-9\s]+
    specifier. Each char-char is just the spcified characters that will be 
    left alone based on ascii value. Currently, a-z,@-Z,0-9 are the characters
    ranges being preserved. Everything else is being replaced by a space (" ").
    
    arg df: pandas df
    
    returns: pandas df with clean column
    """
    df["clean"] = df.comment.apply(remove_punc).apply(lowercase)
    return df


def add_lang_col(df):
    """
    Language detection from comment applied to new language column,
    this is extremely slow.
    
    arg df: pandas df
    
    returns: pandas df with lang column
    """
    df["language"] = df.clean.apply(detect_lang)
    return df


def rmv_stpwds_ret_list(df, stop):
    """
    Remove stopwords and return list
    
    arg df: pandas df (with clean column from clean_comments)
    
    arg stop: list of stopwords to remove
    
    returns: pandas df with no stopwords column filled with
             list of comment without stopwords
    """
    df["no stopwords"] = df.clean.apply(remove_stopwords_list, stop_words = stop)
    return df
        
def rmv_stpwds_ret_str(df, stop):
    """
    Remove stopwords and return list
    
    arg df: pandas df (with clean column from clean_comments)
    
    arg stop: list of stopwords to remove
    
    returns: pandas df with no stopwords column filled with
             string of comment without stopwords
    """
    df["no stopwords"] = df.clean.apply(remove_stopwords_string, stop_words = stop)
    return df        
        

def keywords(df, key):
    """
    Remove everything but specified keyword and return list
    
    arg df: pandas df (with clean column from clean_comments)
    
    arg key: list of keywords to extract
    
    returns: pandas df with keywords column filled with
             list of keywords that appeared in comment
    """
    df["keywords"] = df.clean.apply(rev_keywords, keywords = key)
    return df

def frequency_plot(count, label):
    """
    List of frequency tuples creates bar plot
    
    arg count: list of tuples from most common string (string, count)
    
    arg label: desired label of bar plot
    
    output: displays barplot
    """
    figure(figsize = (17,5))

    labels, values = zip(*count)

    indexes = np.arange(len(labels))

    plt.bar(indexes, values, align = 'center', width = .75, color = 'red')
    plt.xticks(indexes, labels)
    plt.xlabel('word')
    plt.ylabel('# of occurences')
    plt.title(label)

    plt.show()      
        
def function_attributes(code):
    """
    Return function type, parameters, number of parameters from function header
    
    arg code: string of java function code
    
    returns: dictionary of function header attributes
             {"type": fun_type,"num_params":num_params,"params": params}
    """
    fun_head = code[:code.find(")")+1]
    paren_list = fun_head[fun_head.find("(")+1:len(fun_head)-1].split()
    if fun_head.count(',') > 0:
        params = [paren_list[x].rstrip(paren_list[x][-1]) for x in range(len(paren_list)) if paren_list[x][-1] == ',']
        try:
            params.append(paren_list[-1])
        except:
            return "format error"
    elif len(paren_list) == 0:
        params = []
    else:
        params = [paren_list[len(paren_list)-1]]
    num_params = len(params)
    try:
        fun_type = fun_head[:fun_head.find("(")].split()[-2]
    except:
        return "format error"
    return {"type": fun_type,"num_params":num_params,"params": params}

def comment_attributes(com):
    """
    Return comment attributes from comment string
    
    arg com: string of cleaned comment
    
    returns: dictionary of comment attribute
    {"words_after":words_after,"num_@param":at_params,"num_@return":at_returns,"head_len":head_len,"total_@'s":len(params)}
    """
    com = com.split()
    params = [x for x in com if x[0] == '@']
    if len(params) == 0:
        return {"words_after":[],"num_@param":0,"num_@return":0,"head_len":len(com),"total_@'s":0}
    head_len = com.index(params[0])
    words_after = []
    num = 0
    count = 0
    for x in com[com.index(params[0])+1:]:
        count+=1
        if x[0] == "@":
            words_after.append((params[num],count-1))
            num+=1
            count = 0
        elif x == com[len(com)-1]:
            words_after.append((params[num],count))
    if com[-1][0] == "@":
        words_after.append((com[-1],0))
    at_params = len([x for x,y in words_after if x == '@param'])
    at_returns = len([x for x,y in words_after if x == '@return'])
    return {"words_after":words_after,"num_@param":at_params,"num_@return":at_returns,"head_len":head_len,"total_@'s":len(params)}

def comment_eval(com, code, head_weight=3, param_match_weight=2, return_match_weight=2, len_desc_weight=3, head_max=25, desc_max=6):
    """
    Comment evaluator giving Score 1-10. Considers header length, parameter matches, @return
    matches, and length of description after @XXXXX word. 
    
    Header length score is calculated by [(comment head length)/(specified header max score)]*(header length weight),
    max score of (header length weight) if (comment head length) exceeds (specified header max score).
    
    Parameter match score is calculated by [(comment @params's)/(function header parameters)]*(parameter match weight),
    the larger value of (comment @params) and (function header parameters) will always be the denominator.
    
    Return match score is calculated by whether or not the function type should return a value and the presence of an
    @return in the comment e.g. function type is void, no @return in comment: true, so 1*(return match weight)
    
    Length of @XXXX description length calculated by: 
    [(# of words after @XXXX)/(spec. description max score)]*(description length weight),
    max score of (description length weight) if (# words after @XXXX length) 
    exceeds (specified description max score).
    
    Exception is present to avoid incorrectly formatted functions with a "format error" string in place of code attribute
    dictionary.
    
    arg com: comment attribute dictionary
    
    arg code: code attribute dictionary
    
    arg head_weight: weight of header consideration, default=3
    
    arg param_match_weight: weight of matching param count, default=2
    
    arg return_match_weight: weight of return match, default=2
    
    arg len_desc_weight: weight of @XXXX description length, default=3
    
    arg head_max: integer to calculate max value of comment header length, default=25
    
    arg desc_max: integer to calculate max value of @XXXX description length, default=6
    
    returns: float representing comment strength 0-10
    """
    try:
        #Calculate header score
        head_score = com['head_len']/head_max
        if head_score > 1:
            head_score = 1
            head_score *= head_weight
        else:
            head_score *= head_weight

        #Check if parameters in comment and code match
        if code['num_params'] > com['num_@param']:
            param_score = com['num_@param']/code['num_params']
        elif com['num_@param'] > code['num_params']:
            param_score = code['num_params']/com['num_@param']
        else:
            param_score = 1
        param_score *= param_match_weight

        #Check if type of function and @return presence match
        if code['type'] != 'void' and com['num_@return'] > 0:
            return_match_score = 1
        else:
            return_match_score = 0
        return_match_score *= return_match_weight

        #Determine quality of @XXXXX description
        lst = []
        if len(com['words_after']) > 0:
            for x,y in com['words_after']:
                s = y/desc_max
                if s > 1:
                    s=1
                    lst.append(s)
                else:
                    lst.append(s)
        try:
            len_desc_score = sum(lst)/len(lst)
        except ZeroDivisionError:
            len_desc_score = 0
        len_desc_score *= len_desc_weight

        #Return sum of scores
        return head_score + param_score + return_match_score + len_desc_score
    except TypeError:
        return "format error"
        
def add_score_cols(cleaned_df):
    """
    Add columns of comment and code characteristic dictionaries and score
    
    arg cleaned_df: pandas df with cleaned comments column
    
    returns: pandas df with comment_dict column filled with comment attribute dictionaries,
             code_dict column filled with code attribute dictionaries,
             score column filled with comment strength score
    """
    cleaned_df["comment_dict"] = cleaned_df.clean.apply(comment_attributes)
    cleaned_df["code_dict"] = cleaned_df.body.apply(function_attributes)
    cleaned_df["score"] = cleaned_df.apply(lambda x: comment_eval(x.comment_dict,x.code_dict), axis = 1)
    return cleaned_df
        
        
        
        
        
        
        
        
        
        
        
        
        
        