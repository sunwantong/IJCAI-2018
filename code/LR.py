import pandas as pd
from sklearn import linear_model


def load_csv():
    train_set = pd.read_csv('data/ftrain.csv', sep=",")
    val_set = pd.read_csv('data/fvalidate.csv', sep=",")

    train_set = pd.concat([train_set, val_set], axis=0)
    test_set = pd.read_csv('data/ftest.csv', sep=",")

    return train_set,test_set

def linear_regression(train_set,test_set):
    train_x = train_set.drop(['instance_id', 'context_id', 'item_city_id', 'item_id', 'user_id','item_brand_id','shop_id',
                              'user_gender_id', 'user_occupation_id', 'context_timestamp','context_timestamp_and_dates',
                              'dates', 'day', 'hour', 'item_category_list', 'item_property_list', 'predict_category_property',
                              'is_first_get_coupon', 'context_timestamp_rank_desc_label','is_trade',

                              'user_item_hour_count', 'is_last_get_coupon', 'user_shop_hour_count', 'gender_category_count',
                              'user_age_category_count','user_occupation_id_category_count', 'user_item_count', 'property'], axis=1)

    train_y = train_set['is_trade']

    test_x = test_set.drop(['instance_id', 'context_id', 'item_city_id', 'item_id', 'user_id', 'item_brand_id', 'shop_id',
                            'user_gender_id', 'user_occupation_id','context_timestamp', 'context_timestamp_and_dates',
                            'dates', 'day', 'hour', 'item_category_list','item_property_list', 'predict_category_property',
                            'is_first_get_coupon', 'context_timestamp_rank_desc_label',

                            'user_item_hour_count', 'is_last_get_coupon', 'user_shop_hour_count', 'gender_category_count',
                            'user_age_category_count', 'user_occupation_id_category_count', 'user_item_count', 'property'], axis=1)

    print(len(train_x.columns), len(test_x.columns))
    lr = linear_model.LogisticRegression(fit_intercept=False)
    lr.fit(train_x,train_y)
    pred_value = lr.predict_proba(test_x)

    pred_value = pred_value[:,-1]

    # print(pred_value)

    return pred_value

def process_data(train_set,test_set):
    train_set.fillna(-1,inplace=True)
    test_set.fillna(-1, inplace=True)

    return train_set, test_set

def gene_result(pred_value,test_range):
    tess = test_range[["instance_id"]]
    a = pd.DataFrame(pred_value, columns=["predicted_score"])
    res = pd.concat([tess, a["predicted_score"]], axis=1)
    res.to_csv("submit/result_LR.submit", index=None, sep=' ', line_terminator='\r')

def main():
    train_set,test_set = load_csv()
    train_set, test_set = process_data(train_set,test_set)
    pred_value = linear_regression(train_set,test_set)
    gene_result(pred_value,test_set)

if __name__ == '__main__':
    main()