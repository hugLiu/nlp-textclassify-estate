"""
1. 基础库
2. 神经网络框架库
3. 神经网络应用库
4. 本地库
"""
import time
import numpy as np
from sklearn.model_selection import GroupKFold

from utils.helper import (
    set_seed,
    search_f1,
    format_time
)
from corpus import corpus
from configs.config import Config
from datasets.dataset import dataloader_generator
from submits.submit import predict_to_file
from models.estate import EstateModel

# global config
config = Config.yaml_config()
# global corpus data
df_train, df_test = corpus.load_data()
print('df_train.shape', df_train.shape)
print('df_test.shape', df_test.shape)

def gkf_data():
    # The same group will not appear in two different folds
    # (the number of distinct groups has to be at least equal to the number of folds).
    # split: Generate indices to split data into training and test set
    group_kfold = GroupKFold(n_splits=config['k'])
    gkf = group_kfold.split(X=df_train, groups=df_train['id'])

    return gkf

def kfold_train():
    # calculate the train time
    t0 = time.time()
    eval_preds, test_preds = np.zeros((len(df_train), 1)), []
    # GroupKFold train
    for k, (train_idx, eval_idx) in enumerate(gkf_data()):
        print(f'fold train, {k} in {config["k"]}')
        train_dataloader = dataloader_generator(df_train.iloc[train_idx])
        eval_dataloader = dataloader_generator(df_train.iloc[eval_idx], mode='eval')

        # init model
        model = EstateModel()
        # train, evaluate
        model.fit(train_dataloader, eval_dataloader)

        # predict eval
        eval_pred = model.predict(eval_dataloader)
        eval_pred = np.expand_dims(np.array(eval_pred), axis=1)
        eval_preds[eval_idx] = eval_pred

        # predict test
        test_dataloader = dataloader_generator(df_test, mode='test')
        test_pred = model.predict(test_dataloader)
        test_preds.append(test_pred)

    total_time = format_time(time.time() - t0)
    print(f'train finished, took: {total_time}')

    return eval_preds, test_preds

def find_threshold(eval_preds):
    # compute the threshold by all train label and all predictions
    # eval_preds contain all train set
    threshold = search_f1(df_train['label'].values, eval_preds)

    return threshold

def main():
    # Set the seed value all over the place to make this reproducible.
    set_seed(config['seed'])
    # get preds by train
    eval_preds, test_preds = kfold_train()
    avg_test_preds = np.average(test_preds, axis=0)

    threshold = find_threshold(eval_preds)
    test_label = (avg_test_preds > threshold).astype(int)
    # update label feature
    df_test['label'] = test_label

    # generate csv file
    predict_to_file(df_test)

if __name__ == '__main__':
    main()
