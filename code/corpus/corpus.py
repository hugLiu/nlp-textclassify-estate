import pandas as pd

from configs.config import Config

default_config = Config.yaml_config()

def load_data():
    train_left = pd.read_csv('corpus/tsv/train.query.tsv', sep='\t', header=None)
    train_left.columns = ['id', 'q']
    train_right = pd.read_csv('corpus/tsv/train.reply.tsv', sep='\t', header=None)
    train_right.columns = ['id', 'id_sub', 'a', 'label']
    df_train = train_left.merge(train_right, how='left')
    df_train['a'] = df_train['a'].fillna('好的')

    test_left = pd.read_csv('corpus/tsv/test.query.tsv', sep='\t', encoding='gbk', header=None)
    test_left.columns = ['id', 'q']
    test_right = pd.read_csv('corpus/tsv/test.reply.tsv', sep='\t', encoding='gbk', header=None)
    test_right.columns = ['id', 'id_sub', 'a']
    df_test = test_left.merge(test_right, how='left')

    if 'corpus_size' in default_config.keys():
        corpus_size = default_config['corpus_size']
        return df_train[:corpus_size], df_test[:corpus_size]

    return df_train, df_test
