# -*- coding: utf-8 -*-
import random

import xlwt
from imblearn.over_sampling import SVMSMOTE, SMOTE, ADASYN, KMeansSMOTE, RandomOverSampler
import xlrd
import numpy as np
import pandas as pd
import torch
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import *
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deep_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deep_torch.models import *

if __name__ == "__main__":
    data1 = pd.read_excel('data/data.xlsx',sheet_name='data',dtype=str)
    # 处理不平衡数据
    # 欠采样
    ros = RandomUnderSampler(random_state=0)
    # 过采样
    # ros = RandomOverSampler(random_state=0)
    x, y = ros.fit_resample(data1.drop(['label'], axis=1), data1['label'])
    data = pd.concat([x,y],axis=1)

    sparse_features = ['姓名','学院','专业','性别','民族','定向生','企业名称','企业行政属性','企业从事行业','法人','企业地址','企业经济性质','经营范围']
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
                l2_reg_embedding=1e-5,
                device=device,
                dnn_dropout=0.5,
                l2_reg_dnn = 0.01
                )

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc","mse","R","P","acc","F1"], )
    # y = y.astype(float),
    history = model.fit(train_model_input, train[target].values.astype(float), batch_size=32, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    ACC = accuracy_score(test[target].values.astype(np.int64), np.where(pred_ans > 0.5, 1, 0))
    print("test ACC", round(ACC, 4))
    P=precision_score(test[target].values.astype(np.int64), np.where(pred_ans > 0.5, 1, 0),zero_division=0)
    print("test P", round(P, 4))
    R=recall_score(test[target].values.astype(np.int64), np.where(pred_ans > 0.5, 1, 0))
    print("test R", round(R, 4))
    F1=f1_score(test[target].values.astype(np.int64), np.where(pred_ans > 0.5, 1, 0))
    print("test F1score", round(F1, 4))
    print(model)

    # df_recset = pd.DataFrame()
    # df_recset=pd.concat(test_model_input,pred_ans)
    # savepath = "../data/resultset.xls"  # 当前目录新建XLS，存储进去
    #
    # book = xlwt.Workbook(encoding="utf-8", style_compression=0)  # 创建workbook对象
    # sheet = book.add_sheet('input', cell_overwrite_ok=True)  # 创建工作表
    # sheet.write(test_model_input)
    # sheet1 = book.add_sheet('predict', cell_overwrite_ok=True)  # 创建工作表
    # sheet1.write(pred_ans)
    # book.save(savepath)  # 保存
