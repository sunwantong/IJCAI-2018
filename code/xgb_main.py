import pandas as pd
import xgboost as xgb
import scipy as sp
from scipy.stats import pearsonr
import operator

def xgb_model_val():

    #  'is_first_get_coupon','user_hour_count_label','context_timestamp_rank_desc_label'  user_item_brand_count  user_diff_shop_count
    train_set = pd.read_csv('../data/ftrain.csv', sep=",")
    validate_set = pd.read_csv('../data/fvalidate.csv', sep=",")

    train_x = train_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                              'is_trade','context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                              'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label',

                              'user_item_hour_count','is_last_get_coupon','user_shop_hour_count','gender_category_count','user_age_category_count',
                              'user_occupation_id_category_count','user_item_count','property'],axis=1)
    train_y = train_set['is_trade']

    # # 相关性分析
    # pearson_analysis_feature(train_x,train_y)
    # return

    val_x = validate_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                               'is_trade','context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                               'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label',

                              'user_item_hour_count','is_last_get_coupon','user_shop_hour_count','gender_category_count','user_age_category_count',
                              'user_occupation_id_category_count','user_item_count','property'], axis=1)
    val_y = validate_set['is_trade']

    xgb_train = xgb.DMatrix(train_x, label=train_y)
    xgb_val = xgb.DMatrix(val_x, label=val_y)

    print('feature number is:',len(train_x.columns))
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',  # 二分类的问题
              # 'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
              'max_depth': 5,  # 构建树的深度，越大越容易过拟合
              # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
              'subsample': 0.7,  # 随机采样训练样本
              'colsample_bytree': 0.7,  # 生成树时进行的列采样
              'min_child_weight': 3,
              # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
              # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
              # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
              'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
              'eta': 0.03,  # 如同学习率
              'nthread': 30,  # cpu 线程数
              'eval_metric': 'logloss'  # 评价方式
              }

    plst = list(params.items())
    num_rounds = 600  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    # early_stopping_rounds    当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds, watchlist)


    #-----------------------important of feature start-----------------------------------------
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1),reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    print(len(df))
    df.to_csv('feature_important.csv',index=None)
    # -----------------------important of feature end-----------------------------------------




def xgb_model_test():
    train_set = pd.read_csv('../data/ftrain.csv', sep=",")
    val_set = pd.read_csv('../data/fvalidate.csv', sep=",")

    test_set = pd.read_csv('../data/ftest.csv', sep=",")
    train_set = pd.concat([train_set,val_set],axis=0)  # user_item_count  property  364.37556559067

    train_x = train_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                              'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                              'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label','is_trade',

                              'user_item_hour_count','is_last_get_coupon','user_shop_hour_count','gender_category_count','user_age_category_count',
                              'user_occupation_id_category_count','user_item_count','property','user_item_brand_count'],axis=1)
    train_y = train_set['is_trade']

    test_x = test_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                            'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                            'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label',

                            'user_item_hour_count','is_last_get_coupon','user_shop_hour_count','gender_category_count','user_age_category_count',
                            'user_occupation_id_category_count','user_item_count','property','user_item_brand_count'],axis=1)


    # del_feature = ['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
    #                         'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
    #                         'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label',

    #                         'user_item_hour_count','is_last_get_coupon','user_shop_hour_count','gender_category_count','user_age_category_count',
    #                         'user_occupation_id_category_count','user_item_count','property','user_item_brand_count']

    [feature for feature in train_set.columns if feature not in del_feature]

    xgb_train = xgb.DMatrix(train_x.values, label=train_y.values)
    xgb_test = xgb.DMatrix(test_x.values)
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'max_depth': 5,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'min_child_weight': 3,  #该参数值越小，越容易 overfitting。,
              'silent': 0,
              'eta': 0.03,
              'nthread': 30,
              'eval_metric': 'logloss'
              }
    print('feature numbers:',len(train_x.columns))

    watchlist = [(xgb_train, 'train')]
    plst = list(params.items())
    num_rounds = 600  # 迭代次数
    model = xgb.train(plst, xgb_train, num_rounds,evals=watchlist)
    pred_value = model.predict(xgb_test)

    return pred_value,test_set



def gene_result(pred_value,test_range):
    tess = test_range[["instance_id"]]
    a = pd.DataFrame(pred_value, columns=["predicted_score"])
    res = pd.concat([tess, a["predicted_score"]], axis=1)
    res.to_csv("../submit/result_A_B_full(5.11).txt", index=None,sep=' ',line_terminator='\r')

def main():
    # xgb_model_val()

    pred_value,dframe_test = xgb_model_test()
    gene_result(pred_value,dframe_test)


if __name__ == '__main__':
    main()














# def pearson_analysis_feature(feature,label):
#     print(len(feature.columns))
#     res = []
#     print("feature number",len(feature.columns))
#     for pos in range(len(feature.columns)):
#         a = feature.iloc[:,pos]
#         b = feature.iloc[:, pos].name
#
#         # 第一个值是皮尔森相关系数,第二个是p-value,p值<0.05表示有相关性，
#         # 如果此时第一个值绝对值较大，这两个返回值一起表明有较强相关性
#         # p值不是完全可靠的，但对于大于500左右的数据集可能是合理的。
#         x,y = pearsonr(a,label)
#         res.append(b)
#         res.append(x)
#         print(b, x)
