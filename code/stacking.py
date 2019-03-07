import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn import linear_model
import numpy as np
import sklearn.ensemble as sk

def load_csv():
    train_set = pd.read_csv('data/ftrain.csv', sep=",")
    val_set = pd.read_csv('data/fvalidate.csv', sep=",")
    test_set = pd.read_csv('data/ftest.csv', sep=",")

    train_set = pd.concat([train_set, val_set], axis=0)
    return train_set,test_set


def cross_validation(train_set,test_set):
    train_y = train_set['is_trade']
    train_x = train_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                              'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                              'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label','is_trade',

                              'user_item_hour_count','is_last_get_coupon','user_shop_hour_count','gender_category_count','user_age_category_count',
                              'user_occupation_id_category_count','user_item_count','property'],axis=1)

    test_x = test_set.drop(['instance_id', 'context_id', 'item_city_id', 'item_id', 'user_id', 'item_brand_id', 'shop_id',
                            'user_gender_id', 'user_occupation_id','context_timestamp', 'context_timestamp_and_dates', 'dates', 'day', 'hour',
                            'item_category_list','item_property_list', 'predict_category_property', 'is_first_get_coupon', 'context_timestamp_rank_desc_label',
                            'user_item_hour_count', 'is_last_get_coupon', 'user_shop_hour_count', 'gender_category_count',
                            'user_age_category_count','user_occupation_id_category_count', 'user_item_count', 'property'], axis=1)

    train_pred_value_list = []
    test_pred_value_list = []

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_x = xgb.DMatrix(test_x.values)

    skf = StratifiedKFold(n_splits=5,shuffle=False,random_state=None)
    for train_index,test_index in skf.split(train_x,train_y):
        print('Train: %s | test: %s' % (train_index, test_index))

        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        train_pred_value,model = xgb_model(X_train,y_train,X_test)

        print(train_pred_value)


        test_pred_value = model.predict(test_x)

        train_pred_value_list.extend(train_pred_value)
        test_pred_value_list.append(test_pred_value)

    train_set = train_set.reset_index(drop=True)
    a = pd.DataFrame(train_pred_value_list, columns=["xgb_label"])
    train_set = pd.concat([train_set, a["xgb_label"]], axis=1)

    test_pred_value_list = np.array(test_pred_value_list).T
    b = pd.DataFrame(test_pred_value_list, columns=["xgb_label1",'xgb_label2','xgb_label3','xgb_label4','xgb_label5'])
    test_set = pd.concat([test_set,b], axis=1)
    test_set['xgb_label'] = test_set['xgb_label1'] + test_set['xgb_label2'] + test_set['xgb_label3'] + test_set['xgb_label4'] + test_set['xgb_label5']
    test_set['xgb_label'] = test_set['xgb_label'] / 5
    test_set.drop(["xgb_label1",'xgb_label2','xgb_label3','xgb_label4','xgb_label5'],axis=1,inplace=True)

    return train_set,test_set

def xgb_model(X_train,y_train,X_test):
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test)
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'max_depth': 5,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'min_child_weight': 3,
              'silent': 0,
              'eta': 0.03,
              'nthread': 80,
              'eval_metric': 'logloss'
              }
    plst = list(params.items())
    num_rounds = 600  # 迭代次数
    model = xgb.train(plst, xgb_train, num_rounds)
    pred_value = model.predict(xgb_test)
    return pred_value,model



def xgb_model_final(train_set,test_set):
    test_x = test_set.drop(['instance_id', 'context_id', 'item_city_id', 'item_id', 'user_id', 'item_brand_id', 'shop_id',
         'user_gender_id', 'user_occupation_id',
         'context_timestamp', 'context_timestamp_and_dates', 'dates', 'day', 'hour', 'item_category_list',
         'item_property_list', 'predict_category_property', 'is_first_get_coupon', 'context_timestamp_rank_desc_label',

         'user_item_hour_count', 'is_last_get_coupon', 'user_shop_hour_count', 'gender_category_count',
         'user_age_category_count',
         'user_occupation_id_category_count', 'user_item_count', 'property'], axis=1)
    train_y = train_set['is_trade']
    train_x = train_set.drop(['instance_id', 'context_id', 'item_city_id', 'item_id', 'user_id', 'item_brand_id', 'shop_id',
         'user_gender_id', 'user_occupation_id',
         'context_timestamp', 'context_timestamp_and_dates', 'dates', 'day', 'hour', 'item_category_list',
         'item_property_list', 'predict_category_property', 'is_first_get_coupon', 'context_timestamp_rank_desc_label',
         'is_trade',

         'user_item_hour_count', 'is_last_get_coupon', 'user_shop_hour_count', 'gender_category_count',
         'user_age_category_count',
         'user_occupation_id_category_count', 'user_item_count', 'property'], axis=1)

    xgb_train = xgb.DMatrix(train_x, label=train_y)
    xgb_test = xgb.DMatrix(test_x)
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'max_depth': 5,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'min_child_weight': 3,
              'silent': 0,
              'eta': 0.03,
              'nthread': 80,
              'eval_metric': 'logloss'
              }
    plst = list(params.items())
    num_rounds = 600  # 迭代次数
    model = xgb.train(plst, xgb_train, num_rounds)
    pred_value = model.predict(xgb_test)
    return pred_value

def logistic_regression(train_set,test_set):
    test_set = test_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                              'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                              'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label',

                              'user_item_hour_count','is_last_get_coupon','user_shop_hour_count','gender_category_count','user_age_category_count',
                              'user_occupation_id_category_count','user_item_count','property'],axis=1)
    train_y = train_set['is_trade']
    train_x = train_set.drop(['instance_id','context_id','item_city_id','item_id','user_id','item_brand_id','shop_id','user_gender_id','user_occupation_id',
                              'context_timestamp','context_timestamp_and_dates','dates','day','hour','item_category_list',
                              'item_property_list','predict_category_property','is_first_get_coupon','context_timestamp_rank_desc_label','is_trade',

                              'user_item_hour_count','is_last_get_coupon','user_shop_hour_count','gender_category_count','user_age_category_count',
                              'user_occupation_id_category_count','user_item_count','property'],axis=1)

    print(len(train_x.columns),len(test_set.columns))

    lr = linear_model.LogisticRegression(fit_intercept=False)
    lr.fit(train_x, train_y)
    pred_value = lr.predict_proba(test_set)

    pred_value = pred_value[:, -1]
    return pred_value


def random_forest(train_set,test_set):
    test_x = test_set.drop(['instance_id', 'context_id', 'item_city_id', 'item_id', 'user_id', 'item_brand_id', 'shop_id',
         'user_gender_id', 'user_occupation_id',
         'context_timestamp', 'context_timestamp_and_dates', 'dates', 'day', 'hour', 'item_category_list',
         'item_property_list', 'predict_category_property', 'is_first_get_coupon', 'context_timestamp_rank_desc_label',

         'user_item_hour_count', 'is_last_get_coupon', 'user_shop_hour_count', 'gender_category_count',
         'user_age_category_count',
         'user_occupation_id_category_count', 'user_item_count', 'property'], axis=1)
    train_y = train_set['is_trade']
    train_x = train_set.drop(['instance_id', 'context_id', 'item_city_id', 'item_id', 'user_id', 'item_brand_id', 'shop_id',
         'user_gender_id', 'user_occupation_id',
         'context_timestamp', 'context_timestamp_and_dates', 'dates', 'day', 'hour', 'item_category_list',
         'item_property_list', 'predict_category_property', 'is_first_get_coupon', 'context_timestamp_rank_desc_label',
         'is_trade',

         'user_item_hour_count', 'is_last_get_coupon', 'user_shop_hour_count', 'gender_category_count',
         'user_age_category_count',
         'user_occupation_id_category_count', 'user_item_count', 'property'], axis=1)

    rf = sk.RandomForestClassifier(max_depth=15)
    rf.fit(train_x, train_y)
    pred_value = rf.predict_proba(test_x)
    return pred_value[:,-1]

def process_data(train_set,test_set):
    train_set.fillna(-1,inplace=True)
    test_set.fillna(-1, inplace=True)

    return train_set, test_set

def gene_result(pred_value,test_range):
    tess = test_range[["instance_id"]]
    a = pd.DataFrame(pred_value, columns=["predicted_score"])
    res = pd.concat([tess, a["predicted_score"]], axis=1)
    res.to_csv("submit/result_stacking.submit", index=None,sep=' ',line_terminator='\r')

def main():
    train_set, test_set = load_csv()

    train_set, test_set = cross_validation(train_set,test_set)
    train_set, test_set = process_data(train_set,test_set)
    pred_value = logistic_regression(train_set, test_set)

    # pred_value = xgb_model_final(train_set, test_set)
    # pred_value = random_forest(train_set, test_set)


    gene_result(pred_value,test_set)


if __name__ == '__main__':
    main()