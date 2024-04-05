# -*- coding: utf-8 -*-
import pandas
import nltk

"""
Created on Mon Feb 12 18:21:31 2024

@author: pathouli
"""

def clean_txt(str_in):
    import re
    test_str_fix_o = None
    try:
        test_str_fix_o = re.sub("[^A-Za-z']+"," ", str_in).strip()
        test_str_fix_o = test_str_fix_o.lower()
        #return test_str_fix_o
    except:
        print ("Hey you, you can only input a string and not a", type(str_in))
        pass
    return test_str_fix_o

def word_fun(str_in):
    import collections
    tmp = clean_txt(str_in)
    the_w_f_o = dict(collections.Counter(tmp.split()))
    return the_w_f_o

def file_opener(path_i, file_n):
    f = open(path_i + file_n, "r", encoding="UTF8")
    tmp = f.read()
    f.close()
    tmp_clean = clean_txt(tmp)
    return tmp_clean

def file_crawler(the_path_in):
    import pandas as pd
    import os
    my_pd = pd.DataFrame()
    for root, dirs, files in os.walk(the_path_in):
        t = root.split("/")
        label_t = t[-1:][0]
        for name in files:
            try:
                tmp_f = file_opener(root + "/", name)
                tmp_pd = pd.DataFrame(
                    {"label": label_t, "body": tmp_f}, index=[0])
                my_pd = pd.concat([my_pd, tmp_pd],
                                  ignore_index=True)
            except:
                print (root + "/" + name)
                pass
    return my_pd

def rem_sw(str_in):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    #create a function called rem_sw that inputs arbitrary text, 
    #removes stopwords
    #and returns string contenated tokene that are left
    tmp_list = list()
    for w in str_in.split():
        if w not in sw:
            tmp_list.append(w)
    fin_txt = ' '.join(tmp_list)
    return fin_txt

def topic_fun(df_in, col_in):
    topic_dictionary= dict()
    for topic in df_in["label"].unique():
        tmp = df_in[df_in["label"] == topic]
        text = tmp[col_in].str.cat(sep=" ")
        topic_dictionary[topic] = word_fun(text)
    return topic_dictionary

def stem_fun(str_in):
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    tmp = [ps.stem(word) for word in str_in.split()]
    tmp = ' '.join(tmp)
    return tmp

def make_pickle(obj, name, path):
    import pickle 
    pickle.dump(obj, open(path + name + ".pk","wb")) 

def read_pickle(picklePath, pickleName):
    import pickle 
    the_data_t = pickle.load(open(picklePath + pickleName + ".pk", "rb")) 
    return the_data_t

"""
create a function, vec_fun, that takes a pandas column
and returns the vectorized pandas dataframe
give user ability to choose their own ngram range

count vectorizer, counts the number of tokens for each document. Turns into dataframe 
Each row is a corpus and the columns are the features, tokens or sequential token combinations 
Concepts used in text classification and determining significance of word based on word frequency 
"""

"""
refactor vec_fun to save vec_t to an arbitrary location 
with an arbitrary name 
"""
def vec_fun(df_in, m_in, n_in, countVecOption, fileName, out_path):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    if countVecOption: 
        vec_t = CountVectorizer(ngram_range=(m_in, n_in))
        
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pandas as pd
        vec_t = TfidfVectorizer(ngram_range=(m_in,n_in))
        
    xform_data_t = pd.DataFrame(vec_t.fit_transform(df_in).toarray())
    xform_data_t.columns = vec_t.get_feature_names_out()
    make_pickle(xform_data_t,fileName, out_path)
    return xform_data_t
