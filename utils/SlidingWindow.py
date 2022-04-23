import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
import warnings

warnings.filterwarnings("ignore")

def word2vec_train(lst, emb_dim = 150, seed = 42):
    """
    train a word2vec mode
    args: lst(list of string): sentences
          emb_dim(int): word2vec embedding dimensions
          seed(int): seed for word2vec
    return: word2vec model
    """
    # \w 匹配字母、数字、下划线
    # 按照自己定义的规则进行分词
    tokenizer = RegexpTokenizer(r'\w+')
    sentences = []
    # 按行读取日志文件
    # 然后对每行数据词源化 存入sentences中
    for i in lst:
        sentences.append([x.lower() for x in tokenizer.tokenize(str(i))])
    w2v = Word2Vec(sentences, vector_size=emb_dim, min_count=1, seed=seed)
    return w2v
def get_sentence_emb(sentence, w2v):
    """
    get a sentence embedding vector
    *automatic initial random value to the new word
    args: sentence(string): sentence of log message
          w2v: word2vec model
    return: sen_emb(list of int): vector for the sentence
    """
    tokenizer = RegexpTokenizer(r'\w+')
    lst = []
    tokens = [x.lower() for x in tokenizer.tokenize(str(sentence))]
    if tokens == []:
        tokens.append('EmptyParametersTokens')
    for i in range(len(tokens)):
        words = list(w2v.wv.index_to_key)
        # 如果这个词源在当前的words中则加入lst中，否者通过word2vec重新训练后再加入
        if tokens[i] in words:
            lst.append(w2v.wv[tokens[i]])
        else:
            #准备模型词汇
            w2v.build_vocab([[tokens[i]]], update = True)
           #训练单词向量
            w2v.train([tokens[i]], epochs=1, total_examples=len([tokens[i]]))
            lst.append(w2v.wv[tokens[i]])
    #判断这个句子
    drop = 1
    if len(np.array(lst).shape) >= 2:
        sen_emb = np.mean(np.array(lst), axis=0)
        if len(np.array(lst)) >= 5:
            drop = 0
    else:
        sen_emb = np.array(lst)
    return list(sen_emb), drop

def word2emb(df_source, df_target, train_size_s, train_size_t, step_size, emb_dim):
    # np.concatenate()沿现有轴连接一系列数组
    # [:] 截取字符串的一部分   [1:5] 截取1-5部分
    w2v = word2vec_train(np.concatenate((df_source.EventTemplate.values[:step_size*train_size_s], df_target.EventTemplate.values[:step_size*train_size_t])), emb_dim=emb_dim)
    print('Processing words in the source dataset')
    # 处理 source数据
    dic = {}
    # set集合是没有重复的对象集合，所有的元素都是唯一的 模板数据
    lst_temp = list(set(df_source.EventTemplate.values))
    for i in tqdm(range(len(lst_temp))):
        #得到句子嵌入向量和对应的drop数
        (temp_val, drop) = get_sentence_emb([lst_temp[i]], w2v)
        # 将对应的句子嵌入向量和drop值存入字典中
        dic[lst_temp[i]] = (temp_val, drop)
    lst_emb = []
    lst_drop = []
    for i in tqdm(range(len(df_source))):
        lst_emb.append(dic[df_source.EventTemplate.loc[i]][0])
        lst_drop.append(dic[df_source.EventTemplate.loc[i]][1])
    df_source['Embedding'] = lst_emb
    df_source['drop'] = lst_drop
    print('Processing words in the target dataset')
    #处理 target数据
    dic = {}
    lst_temp = list(set(df_target.EventTemplate.values))
    for i in tqdm(range(len(lst_temp))):
        (temp_val, drop) = get_sentence_emb([lst_temp[i]], w2v)
        dic[lst_temp[i]] = (temp_val, drop)
    lst_emb = []
    lst_drop = []
    for i in tqdm(range(len(df_target))):
        lst_emb.append(dic[df_target.EventTemplate.loc[i]][0])
        lst_drop.append(dic[df_target.EventTemplate.loc[i]][1])
    df_target['Embedding'] = lst_emb
    df_target['drop'] = lst_drop
    #将drop=1的句子嵌入向量丢弃
    # loc函数：通过行索引“index”中的具体值来取行数据
    df_source = df_source.loc[df_source['drop'] == 0]
    df_target = df_target.loc[df_target['drop'] == 0]
    #丢弃无单词日志后的源长度
    print(f'Source length after drop none word logs: {len(df_source)}')
    print(f'Target length after drop none word logs: {len(df_target)}')
    # 返回处理好的数据以及w2v
    return df_source, df_target, w2v

def sliding_window(df, window_size = 20, step_size = 4, target = 0, val_date = '2005.11.15'):
    df["Label"] = df["Label"].apply(lambda x: int(x != '-'))
    df = df[["Label", "Content", "Embedding", "Date"]]
    df['target'] = target
    df['val'] = 0
    #获取日志长度
    log_size = df.shape[0]
    #iloc函数：通过行号来取行数据 （0,0） （1,0） （3,0）
    label_data = df.iloc[:, 0]
    # （0,‘instruction cache parity error corrected’）（2,'machine check interrupt (bit=0x1d): L2 dcache unit data parity error')
    logkey_data = df.iloc[:, 1]
    # (0,[-0.4774356, -0.2533616, 0.14253572, 0.51238215, 0.12990837, 0.07629535, -0.1034914, 0.27132705, -0.12041668, -0.14161834, 0.192397, -0.1253474, 0.6705574, 0.329522, -0.098604, 0.50946885, -0.020411626, -0.025722498, 0.05654074, -0.015345657, 0.63066757, 1.14132, 0.13585927, -0.45857906, -0.17273347, 0.71507746, -0.0026506782, -0.15278031, -0.5056721, -0.48632222, 0.53128386, 0.070083, -0.15145102, 0.50135934, 0.27355155, 0.05374694, 0.06405538, 0.022331458, -0.569962, -0.34306005, -0.24165864, 0.1767532, 0.53125304, 0.32945734, -0.05223488, -0.31562242, 0.4795099, 1.005008, 0.27928126, 0.576686, -0.6507969, -0.47602606, -0.74957466, 0.4778091, -0.21496145, 0.27753642, 0.07783038, 0.22235087, 0.10105457, 0.7214346, -0.036871836, -0.08378242, 0.3065753, 0.7148466, 0.02527436, -0.24668713, 0.47928277, -0.029116193, 0.35796785, -0.0629101, -1.0676856, -0.868769, -0.43072358, -0.07846196, -0.77180445, -0.17986426, -0.25503045, 0.35563827, -0.021989226, -0.11137234, 0.21540189, -0.12968381..)
    emb_data = df.iloc[:, 2]
    # (0,'2005.06.03')
    date_data = df.iloc[:, 3]
    new_data = []
    index = 0
    while index <= log_size-window_size:
        if date_data.iloc[index] == val_date:
            new_data.append([
                max(label_data[index:index+window_size]),
                logkey_data[index:index+window_size].values,
                emb_data[index:index+window_size].values,
                date_data.iloc[index],
                target,
                1
            ])
            index += step_size
        else:
            new_data.append([
                max(label_data[index:index+window_size]),
                logkey_data[index:index+window_size].values,
                emb_data[index:index+window_size].values,
                date_data.iloc[index],
                target,
                0
            ])
            index += step_size
    return pd.DataFrame(new_data, columns=df.columns)

def get_datasets(df_source, df_target, options, val_date="2005.11.15"):
    # Get source data preprocessed
    #从配置文件中获取各种配置
    window_size = options["window_size"]
    step_size = options["step_size"]
    source = options["source_dataset_name"]
    target = options["target_dataset_name"]
    train_size_s = options["train_size_s"]
    train_size_t = options["train_size_t"]
    emb_dim = options["emb_dim"]
    times =  int(train_size_s/train_size_t) - 1

    df_source, df_target, w2v = word2emb(df_source, df_target, train_size_s, train_size_t, step_size, emb_dim)

    print(f'Start preprocessing for the source: {source} dataset')
    window_df = sliding_window(df_source, window_size, step_size, 0, val_date)
    #满足特定日期的数据
    r_s_val_df = window_df[window_df['val'] == 1]
    # 除去特定日期的数据
    window_df = window_df[window_df['val'] == 0]

    # Training normal data
    df_normal = window_df[window_df["Label"] == 0]

    # shuffle normal data 将数据和下标都进行打乱
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
    train_len = train_size_s

    train_normal_s = df_normal[:train_len]
    print("Source training size {}".format(len(train_normal_s)))

    # Test normal data
    test_normal_s = df_normal[train_len:]
    print("Source test normal size {}".format(len(test_normal_s)))

    # Testing abnormal data
    test_abnormal_s = window_df[window_df["Label"] == 1]
    print('Source test abnormal size {}'.format(len(test_abnormal_s)))

    print('------------------------------------------')
    print(f'Start preprocessing for the target: {target} dataset')
    # Get target data preprocessed
    window_df = sliding_window(df_target, window_size, step_size, 1, val_date)
    r_t_val_df = window_df[window_df['val'] == 1]
    window_df = window_df[window_df['val'] == 0]

    # Training normal data
    df_normal = window_df[window_df["Label"] == 0]
    # shuffle normal data
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
    train_len = train_size_t

    train_normal_t = df_normal[:train_len]
    print("Target training size {}".format(len(train_normal_t)))
    temp = train_normal_t[:]
    for _ in range(times):
        train_normal_t = pd.concat([train_normal_t, temp])

    # Testing normal data
    test_normal_t = df_normal[train_len:]
    print("Target test normal size {}".format(len(test_normal_t)))

    # Testing abnormal data
    test_abnormal_t = window_df[window_df["Label"] == 1]
    print('Target test abnormal size {}'.format(len(test_abnormal_t)))

    return train_normal_s, test_normal_s, test_abnormal_s, r_s_val_df, \
           train_normal_t, test_normal_t, test_abnormal_t, r_t_val_df, w2v