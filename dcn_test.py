# -*- coding: utf-8 -*-
import random

from imblearn.over_sampling import SVMSMOTE, SMOTE, ADASYN, KMeansSMOTE, RandomOverSampler
import xlrd
import pandas as pd
import torch
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deep_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deep_torch.models import *

if __name__ == "__main__":
    data1 = pd.read_excel('data/data_test.xlsx',sheet_name='Sheet2',dtype=str)
    # data1 = pd.read_excel('data/1d_data.xlsx',sheet_name='一维数据',dtype=str)
    # 处理不平衡数据
    # 欠采样
    ros = RandomUnderSampler(random_state=0)
    # 过采样
    # ros = RandomOverSampler(random_state=0)
    x, y = ros.fit_resample(data1.drop(['label'], axis=1), data1['label'])
    data = pd.concat([x,y],axis=1)

    sparse_features = ['姓名','学院','专业','性别','民族','定向生','方向','名称','法人','地址']
    dense_features = ['注册资本','成立日期']

    data[sparse_features] = data[sparse_features].fillna('-', )
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2, random_state=2022)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DCN(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    # y = y.astype(float),
    history = model.fit(train_model_input, train[target].values.astype(float), batch_size=32, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))