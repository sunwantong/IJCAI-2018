import pandas as pd
import xgboost as xgb
import scipy as sp
from scipy.stats import pearsonr
import operator

def xgb_model():
    train_set = pd.read_csv('../data/ftrain.csv', sep=",") # maxminscalar
    test_set = pd.read_csv('../data/ftest.csv', sep=",")

    train_x = train_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                              'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                              'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label','is_trade',

                              'user_item_hour_countday31-05','user_item_hour_countday06','is_last_get_coupon','user_shop_hour_countday31-05',
                              'user_shop_hour_countday06','gender_category_count','user_age_category_count',
                              'user_occupation_id_category_count','user_item_countday31-05','user_item_countday06','property'],axis=1)
    train_y = train_set['is_trade']

    test_x = test_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                            'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                            'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label',

                            'user_item_hour_countday31-05','user_item_hour_countday06','is_last_get_coupon','user_shop_hour_countday31-05',
                            'user_shop_hour_countday06','gender_category_count','user_age_category_count',
                            'user_occupation_id_category_count','user_item_countday31-05','user_item_countday06','property'],axis=1)


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

    return pred_value,test_set  #52667.0719781482

def xgb_model_bak():
    train_set = pd.read_csv('../data/ftrain.csv', sep=",") # maxminscalar
    test_set = pd.read_csv('../data/ftest.csv', sep=",")

    train_x = train_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                              'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                              'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label','is_trade',
                              'user_item_hour_countday31-05','is_last_get_coupon','user_shop_hour_countday31-05','gender_category_count',
                              'user_age_category_count','user_occupation_id_category_count','user_item_countday31-05','property'],axis=1)
    train_y = train_set['is_trade']

    test_x = test_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                            'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                            'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label',
                            'user_item_hour_countday31-05','is_last_get_coupon','user_shop_hour_countday31-05','gender_category_count',
                            'user_age_category_count','user_occupation_id_category_count','user_item_countday31-05','property'],axis=1)


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

    return pred_value,test_set  #52667.0719781482


def gene_result(pred_value,test_range):
    tess = test_range[["instance_id"]]
    a = pd.DataFrame(pred_value, columns=["predicted_score"])
    res = pd.concat([tess, a["predicted_score"]], axis=1)
    res.to_csv("../submit/result_A_B_full(5.13).txt", index=None,sep=' ',line_terminator='\r')

def main():
    pred_value,dframe_test = xgb_model_bak()
    # pred_value,dframe_test = xgb_model()
    gene_result(pred_value,dframe_test)


if __name__ == '__main__':
    main()
