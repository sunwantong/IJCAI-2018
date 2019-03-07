import pandas as pd
import numpy as np
import datetime
import os as os
import pickle
import sklearn.preprocessing as preprocess

def load_csv():
    dframe = pd.read_csv('../round2_data/round2_train.csv', sep=",")
    dframe_test = pd.read_csv('../round2_data/round2_test.csv', sep=",")

    dframe,dframe_test = pre_process_data(dframe, dframe_test)
    train_feature_range_click_imp_and_rate,train_label_range = data_partition(dframe,dframe_test)

    return train_feature_range_click_imp_and_rate,train_label_range



def data_partition(dframe,dframe_test):
    # feature_range1
    train_feature_range_click_imp_and_rate = dframe[(dframe['context_timestamp'] >= '20180831000000') &
                                            (dframe['context_timestamp'] <= '20180906235959')]

    train_label_range_7day_mor = dframe[(dframe['context_timestamp'] >= '20180907000000') &
                                (dframe['context_timestamp'] <= '20180907115959')]

    train_label_range_test = dframe_test
    train_label_range_test['is_trade'] = -1

    train_label_range = train_label_range_7day_mor.append(train_label_range_test)

    return train_feature_range_click_imp_and_rate,train_label_range




# 交易数，总数，转化率
def day31today5_click_imp_and_rate(train_feature_range,train_label_range,flag):

    # 商铺出现的次数
    d1 = train_feature_range[['shop_id', 'item_sales_level']]
    ftrain = stat_feature_count(train_label_range, d1, ['shop_id'], 'item_sales_level', 'shop_count'+flag)

    # 用户和商铺出现的次数
    d2 = train_feature_range[['shop_id', 'user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d2, ["shop_id", 'user_id'], 'item_sales_level', 'user_shop_count' + flag)

    # 用户出现次数(总的次数)
    d3 = train_feature_range[['user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain,d3, ['user_id'], 'item_sales_level', 'user_count' + flag)

    # 用户出现次数(产生交易的次数)
    d4 = train_feature_range[train_feature_range['is_trade'] == 1][['user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d4, ['user_id'], 'item_sales_level', 'user_count_istrade' + flag)

    #空值替换
    ftrain['user_count_istrade'+flag].replace(np.nan,0,inplace=True)
    ftrain['user_count'+flag].replace(np.nan,0,inplace=True)
    #交易率
    ftrain['user_istrade_rate'+flag] = (ftrain['user_count_istrade'+flag].astype('float') + 0.00001) / (ftrain['user_count'+flag].astype('float') + 0.001)

    # 广告出现的次数
    d5 = train_feature_range[['item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d5, ['item_id'], 'item_sales_level', 'item_count' + flag)

    # 该商家的特定广告出现的次数
    d6 = train_feature_range[['item_id', 'shop_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d6, ['item_id','shop_id'], 'item_sales_level', 'shop_item_count' + flag)

    # 特定用户和该广告出现的次数
    d7 = train_feature_range[['user_id', 'item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d7, ['item_id', 'user_id'], 'item_sales_level', 'user_item_count' + flag)

    # 特定用户和该广告出现的次数(交易成功)
    d8 = train_feature_range[train_feature_range.is_trade == 1][['user_id', 'item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d8, ['item_id', 'user_id'], 'item_sales_level', 'user_item_count_istrade' + flag)

    # 空值替换
    ftrain['user_item_count'+flag].replace(np.nan,0, inplace=True)
    ftrain['user_item_count_istrade'+flag].replace(np.nan,0, inplace=True)
    # 交易率
    ftrain['user_item_count_istrade_rate'+flag] = (ftrain['user_item_count_istrade'+flag].astype('float') + 0.00001) / (ftrain['user_item_count'+flag].astype('float') + 0.001)

    # 用户和该用户的职业出现的次数
    d9 = train_feature_range[['user_id', 'user_occupation_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain,d9,['user_occupation_id', 'user_id'], 'item_sales_level', 'user_and_user_occupation_count' + flag)

    # 该职业出现的次数
    d13 = train_feature_range[['user_occupation_id', 'item_price_level']]
    ftrain = stat_feature_count(ftrain, d13, ['user_occupation_id'], 'item_price_level','user_occupation_id_count' + flag)

    # 广告价格等级出现的次数
    d14 = train_feature_range[['item_price_level', 'user_id']]
    ftrain = stat_feature_count(ftrain, d14, ['item_price_level'], 'user_id','item_price_level_count' + flag)

    # 性别对于是否交易的影响
    d15 = train_feature_range[['user_gender_id', 'user_id']]
    ftrain = stat_feature_count(ftrain, d15, ['user_gender_id'], 'user_id', 'user_gender_id_count' + flag)

    # 看当前日期是否是高峰期
    ftrain['is_high_trade'] = ftrain['context_timestamp'].apply(date2morning_afternoon)

    # 收藏级别出现的次数
    d17 = train_feature_range[['item_collected_level', 'item_price_level']]
    ftrain = stat_feature_count(ftrain, d17, ['item_collected_level'], 'item_price_level', 'item_collected_count' + flag)

    # 广告商品页面展示标号
    d18 = train_feature_range[['context_page_id', 'item_id', 'item_price_level']]
    ftrain = stat_feature_count(ftrain, d18, ["context_page_id", 'item_id'], 'item_price_level','context_page_item_count' + flag)

    # 用户和商铺,广告出现的次数
    d20 = train_feature_range[['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d20, ["shop_id", 'user_id', 'item_id'], 'item_sales_level','user_shop_item_count' + flag)

    train_feature_range['day'] = train_feature_range['context_timestamp'].map(lambda x: int(x[6:8]))
    train_feature_range['hour'] = train_feature_range['context_timestamp'].map(lambda x: int(x[8:10]))
    ftrain['day'] = ftrain['context_timestamp'].map(lambda x: int(x[6:8]))
    ftrain['hour'] = ftrain['context_timestamp'].map(lambda x: int(x[8:10]))

    # user和shop在每一个小时(24个小时)的的count个数
    d21 = train_feature_range[['user_id', 'shop_id', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d21, ['user_id', "shop_id", 'hour'], 'item_sales_level','user_shop_hour_count' + flag)

    # user在每一个小时(24个小时)的的count个数
    d22 = train_feature_range[['user_id','hour','item_sales_level']]
    ftrain = stat_feature_count(ftrain, d22, ['user_id','hour'], 'item_sales_level','user_hour_count' + flag)

    # shop在每一个小时(24个小时)的的count个数
    d23 = train_feature_range[['shop_id','hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d23, ['shop_id', 'hour'], 'item_sales_level', 'shop_hour_count' + flag)

    # item在每一个小时(24个小时)的的count个数
    d24 = train_feature_range[['item_id','hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d24, ['item_id', 'hour'], 'item_sales_level', 'item_hour_count' + flag)

    # user和item在每一个小时(24个小时)的的count个数
    d29 = train_feature_range[['user_id', 'item_id', 'hour','item_sales_level']]
    ftrain = stat_feature_count(ftrain, d29, ['user_id','item_id', 'hour'], 'item_sales_level', 'user_item_hour_count' + flag)

    # 用户和广告出现的次数
    d30 = train_feature_range[train_feature_range.is_trade == 1][['item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d30, ['item_id'], 'item_sales_level','item_count_istrade' + flag)

    # 空值替换
    ftrain['item_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_istrade_rate'+flag] = (ftrain['item_count_istrade'+flag].astype('float') + 0.00001) / (ftrain['item_count'+flag].astype('float') + 0.001)

    # shop交易count
    d33 = train_feature_range[train_feature_range.is_trade == 1][['shop_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d33,['shop_id'], 'item_sales_level', 'shop_count_istrade' + flag)

    ftrain['shop_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['shop_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['shop_istrade_rate'+flag] = (ftrain['shop_count_istrade'+flag].astype('float') + 0.00001) / (ftrain['shop_count'+flag].astype('float') + 0.001)

    # item_city_id的count
    d34 = train_feature_range[['item_city_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d34, ['item_city_id'], 'item_sales_level', 'item_city_id_count' + flag)

    # user_id,item_city_id的count
    d35 = train_feature_range[['user_id', 'item_city_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d35, ['item_city_id','user_id'], 'item_sales_level', 'user_item_city_id_count' + flag)

    # user和shop的次数
    d36 = train_feature_range[train_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d36, ['shop_id', 'user_id'], 'item_sales_level','user_shop_count_istrade' + flag)

    ftrain['user_shop_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_shop_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_shop_count_istrade_rate'+flag] = (ftrain['user_shop_count_istrade'+flag].astype('float') + 0.00001) / (ftrain['user_shop_count'+flag].astype('float') + 0.001)

    # 用户和广告出现的次数item_count  # 空值替换
    ftrain['item_count'+flag].replace(np.nan,0, inplace=True)
    ftrain['item_count_istrade'+flag].replace(np.nan,0, inplace=True)

    # user和item_brand_id的次数
    d37 = train_feature_range.groupby(['user_id','item_brand_id'], as_index=False)['item_sales_level'].agg({'user_item_brand_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d37, on=["item_brand_id", 'user_id'], how="left")

    # user with shop_star_level count
    d38 = train_feature_range.groupby(['user_id','shop_star_level'],as_index=False)['item_sales_level'].agg({'user_shop_star_level_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d38, on=["shop_star_level", 'user_id'], how="left")

    # user with item_sales_level count
    d39 = train_feature_range.groupby(['user_id', 'item_sales_level'], as_index=False)['item_sales_level'].agg({'user_item_sales_level_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d39, on=["item_sales_level", 'user_id'], how="left")

    # user with item_collected_level count
    d40 = train_feature_range.groupby(['user_id', 'item_collected_level'], as_index=False)['item_sales_level'].\
        agg({'user_item_collected_level_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d40, on=["item_collected_level", 'user_id'], how="left")
    #
    # user with category count
    d41 = train_feature_range.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d41, on=["category", 'user_id'], how="left")

    #
    d44 = train_feature_range.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d44, on=["item_city_id", 'user_id','item_id'], how="left")


    ######  new
    d45 = train_feature_range[train_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d45 = d45.groupby(['shop_id', 'user_id', 'item_id'], as_index=False)['item_sales_level'].agg({'user_shop_item_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d45, on=['shop_id', 'user_id', 'item_id'], how="left")

    ftrain['user_shop_item_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_shop_item_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_shop_item_count_istrade_rate'+flag] = (ftrain['user_shop_item_count_istrade'+flag].astype('float') + 0.00001) / \
                                                       (ftrain['user_shop_item_count'+flag].astype('float') + 0.001)

    ####
    d46 = train_feature_range[train_feature_range.is_trade == 1][['user_id', 'item_id', 'item_city_id', 'item_sales_level']]
    d46 = d46.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d46, on=['user_id', 'item_id', 'item_city_id'], how="left")

    ftrain['user_item_item_city_id_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_item_item_city_id_count_istrade'+flag].replace(np.nan, 0, inplace=True)

    ftrain['user_item_item_city_id_count_istrade_rate'+flag] = (ftrain['user_item_item_city_id_count_istrade'+flag].astype('float') + 0.00001) /\
                                                               (ftrain['user_item_item_city_id_count'+flag].astype('float') + 0.001)

    ### user category
    d47 = train_feature_range[train_feature_range.is_trade == 1][['user_id', 'category','item_sales_level']]
    d47 = d47.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d47, on=['user_id', 'category'], how="left")

    ftrain['user_category_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_category_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_category_count_istrade_rate'+flag] = (ftrain['user_category_count_istrade'+flag].astype('float') + 0.00001) /\
                                                      (ftrain['user_category_count'+flag].astype('float') + 0.001)

    # user_gender_id
    d48 = train_feature_range[train_feature_range.is_trade == 1][['user_gender_id', 'item_sales_level']]
    d48 = d48.groupby(['user_gender_id'], as_index=False)['item_sales_level'].agg({'user_gender_id_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d48, on=['user_gender_id'], how="left")

    ftrain['user_gender_id_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_gender_id_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_gender_id_count_istrade_rate'+flag] = (ftrain['user_gender_id_count_istrade'+flag].astype('float') + 0.00001) /\
                                                  (ftrain['user_gender_id_count'+flag].astype('float') + 0.001)

    # user_occupation_id
    d49 = train_feature_range[train_feature_range.is_trade == 1][['user_occupation_id', 'item_sales_level']]
    d49 = d49.groupby(['user_occupation_id'], as_index=False)['item_sales_level'].agg({'user_occupation_id_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d49, on=['user_occupation_id'], how="left")

    ftrain['user_occupation_id_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_occupation_id_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_occupation_id_count_istrade_rate'+flag] = (ftrain['user_occupation_id_count_istrade'+flag].astype('float') + 0.00001) / \
                                                      (ftrain['user_occupation_id_count'+flag].astype('float') + 0.001)

    # user_age_level
    d50 = train_feature_range[['user_age_level', 'item_sales_level']]
    d50 = d50.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d50, on=['user_age_level'], how="left")

    d51 = train_feature_range[train_feature_range.is_trade == 1][['user_age_level', 'item_sales_level']]
    d51 = d51.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d51, on=['user_age_level'], how="left")

    ftrain['user_age_level_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_age_level_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_age_level_count_istrade_rate'+flag] = (ftrain['user_age_level_count_istrade'+flag].astype('float') + 0.00001) /\
                                                       (ftrain['user_age_level_count'+flag].astype('float') + 0.001)


    # item_sales_level
    d52 = train_feature_range[['item_sales_level', 'context_timestamp']]
    d52 = d52.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d52, on=['item_sales_level'], how="left")

    d53 = train_feature_range[train_feature_range.is_trade == 1][['item_sales_level', 'context_timestamp']]
    d53 = d53.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d53, on=['item_sales_level'], how="left")

    ftrain['item_sales_level_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_sales_level_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_sales_level_count_istrade_rate'+flag] = (ftrain['item_sales_level_count_istrade'+flag].astype('float') + 0.00001) / \
                                                         (ftrain['item_sales_level_count'+flag].astype('float') + 0.001)


    # item_collected_level
    d54 = train_feature_range[train_feature_range.is_trade == 1][['item_collected_level', 'item_sales_level']]
    d54 = d54.groupby(['item_collected_level'], as_index=False)['item_sales_level'].agg({'item_collected_level_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d54, on=['item_collected_level'], how="left")

    ftrain['item_collected_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_collected_level_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_collected_level_count_istrade_rate'+flag] = (ftrain['item_collected_level_count_istrade'+flag].astype('float') + 0.00001) / \
                                                             (ftrain['item_collected_count'+flag].astype('float') + 0.001)

    # user_star_level
    d55 = train_feature_range[['user_star_level', 'item_sales_level']]
    d55 = d55.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d55, on=['user_star_level'], how="left")

    d56 = train_feature_range[train_feature_range.is_trade == 1][['user_star_level', 'item_sales_level']]
    d56 = d56.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d56, on=['user_star_level'], how="left")

    ftrain['user_star_level_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_star_level_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['user_star_level_count_istrade_rate'+flag] = (ftrain['user_star_level_count_istrade'+flag].astype('float') + 0.00001) /\
                                                        (ftrain['user_star_level_count'+flag].astype('float') + 0.001)

    # item_pv_level
    d57 = train_feature_range[['item_pv_level', 'item_sales_level']]
    d57 = d57.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d57, on=['item_pv_level'], how="left")

    d58 = train_feature_range[train_feature_range.is_trade == 1][['item_pv_level', 'item_sales_level']]
    d58 = d58.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d58, on=['item_pv_level'], how="left")

    ftrain['item_pv_level_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_pv_level_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_pv_level_count_istrade_rate'+flag] = (ftrain['item_pv_level_count_istrade'+flag].astype('float') + 0.00001) /\
                                                      (ftrain['item_pv_level_count'+flag].astype('float') + 0.001)

    # shop_review_num_level
    d59 = train_feature_range[['shop_review_num_level', 'item_sales_level']]
    d59 = d59.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count'+flag: 'count'})
    ftrain = pd.merge(ftrain, d59, on=['shop_review_num_level'], how="left")

    d60 = train_feature_range[train_feature_range.is_trade == 1][['shop_review_num_level', 'item_sales_level']]
    d60 = d60.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d60, on=['shop_review_num_level'], how="left")

    ftrain['shop_review_num_level_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['shop_review_num_level_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['shop_review_num_level_count_istrade_rate'+flag] = (ftrain['shop_review_num_level_count_istrade'+flag].astype('float') + 0.00001) / \
                                                              (ftrain['shop_review_num_level_count'+flag].astype('float') + 0.001)


    # item_price_level
    d62 = train_feature_range[train_feature_range.is_trade == 1][['item_price_level', 'item_sales_level']]
    d62 = d62.groupby(['item_price_level'], as_index=False)['item_sales_level'].agg({'item_price_level_count_istrade'+flag: 'count'})
    ftrain = pd.merge(ftrain, d62, on=['item_price_level'], how="left")

    ftrain['item_price_level_count'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_price_level_count_istrade'+flag].replace(np.nan, 0, inplace=True)
    ftrain['item_price_level_count_istrade_rate'+flag] = (ftrain['item_price_level_count_istrade'+flag].astype('float') + 0.00001) / \
                                                    (ftrain['item_price_level_count'+flag].astype('float') + 0.001)

    print("31-5 over!")

    return ftrain




# 交易数，总数
def day06_click_imp(train_feature_range,train_label_range,flag):
    # 商铺出现的次数
    d1 = train_feature_range[['shop_id', 'item_sales_level']]
    ftrain = stat_feature_count(train_label_range, d1, ['shop_id'], 'item_sales_level', 'shop_count' + flag)

    # 用户和商铺出现的次数
    d2 = train_feature_range[['shop_id', 'user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d2, ["shop_id", 'user_id'], 'item_sales_level', 'user_shop_count' + flag)

    # 用户出现次数(总的次数)
    d3 = train_feature_range[['user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d3, ['user_id'], 'item_sales_level', 'user_count' + flag)

    # 用户出现次数(产生交易的次数)
    d4 = train_feature_range[train_feature_range['is_trade'] == 1][['user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d4, ['user_id'], 'item_sales_level', 'user_count_istrade' + flag)

    # 空值替换
    ftrain['user_count_istrade' + flag].replace(np.nan, 0, inplace=True)
    ftrain['user_count' + flag].replace(np.nan, 0, inplace=True)

    # 广告出现的次数
    d5 = train_feature_range[['item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d5, ['item_id'], 'item_sales_level', 'item_count' + flag)

    # 该商家的特定广告出现的次数
    d6 = train_feature_range[['item_id', 'shop_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d6, ['item_id', 'shop_id'], 'item_sales_level', 'shop_item_count' + flag)

    # 特定用户和该广告出现的次数
    d7 = train_feature_range[['user_id', 'item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d7, ['item_id', 'user_id'], 'item_sales_level', 'user_item_count' + flag)

    # 特定用户和该广告出现的次数(交易成功)
    d8 = train_feature_range[train_feature_range.is_trade == 1][['user_id', 'item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d8, ['item_id', 'user_id'], 'item_sales_level','user_item_count_istrade' + flag)

    # 用户和该用户的职业出现的次数
    d9 = train_feature_range[['user_id', 'user_occupation_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d9, ['user_occupation_id', 'user_id'], 'item_sales_level','user_and_user_occupation_count' + flag)

    # 该职业出现的次数
    d13 = train_feature_range[['user_occupation_id', 'item_price_level']]
    ftrain = stat_feature_count(ftrain, d13, ['user_occupation_id'], 'item_price_level','user_occupation_id_count' + flag)

    # 广告价格等级出现的次数
    d14 = train_feature_range[['item_price_level', 'user_id']]
    ftrain = stat_feature_count(ftrain, d14, ['item_price_level'], 'user_id', 'item_price_level_count' + flag)

    # 性别对于是否交易的影响
    d15 = train_feature_range[['user_gender_id', 'user_id']]
    ftrain = stat_feature_count(ftrain, d15, ['user_gender_id'], 'user_id', 'user_gender_id_count' + flag)

    # 看当前日期是否是高峰期
    ftrain['is_high_trade'] = ftrain['context_timestamp'].apply(date2morning_afternoon)

    # 收藏级别出现的次数
    d17 = train_feature_range[['item_collected_level', 'item_price_level']]
    ftrain = stat_feature_count(ftrain, d17, ['item_collected_level'], 'item_price_level','item_collected_count' + flag)

    # 广告商品页面展示标号
    d18 = train_feature_range[['context_page_id', 'item_id', 'item_price_level']]
    ftrain = stat_feature_count(ftrain, d18, ["context_page_id", 'item_id'], 'item_price_level','context_page_item_count' + flag)

    # 用户和商铺,广告出现的次数
    d20 = train_feature_range[['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d20, ["shop_id", 'user_id', 'item_id'], 'item_sales_level','user_shop_item_count' + flag)

    train_feature_range['day'] = train_feature_range['context_timestamp'].map(lambda x: int(x[6:8]))
    train_feature_range['hour'] = train_feature_range['context_timestamp'].map(lambda x: int(x[8:10]))
    ftrain['day'] = ftrain['context_timestamp'].map(lambda x: int(x[6:8]))
    ftrain['hour'] = ftrain['context_timestamp'].map(lambda x: int(x[8:10]))

    # user和shop在每一个小时(24个小时)的的count个数
    d21 = train_feature_range[['user_id', 'shop_id', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d21, ['user_id', "shop_id", 'hour'], 'item_sales_level','user_shop_hour_count' + flag)

    # user在每一个小时(24个小时)的的count个数
    d22 = train_feature_range[['user_id', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d22, ['user_id', 'hour'], 'item_sales_level', 'user_hour_count' + flag)

    # shop在每一个小时(24个小时)的的count个数
    d23 = train_feature_range[['shop_id', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d23, ['shop_id', 'hour'], 'item_sales_level', 'shop_hour_count' + flag)

    # item在每一个小时(24个小时)的的count个数
    d24 = train_feature_range[['item_id', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d24, ['item_id', 'hour'], 'item_sales_level', 'item_hour_count' + flag)

    # user和item在每一个小时(24个小时)的的count个数
    d29 = train_feature_range[['user_id', 'item_id', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d29, ['user_id', 'item_id', 'hour'], 'item_sales_level','user_item_hour_count' + flag)

    # 用户和广告出现的次数
    d30 = train_feature_range[train_feature_range.is_trade == 1][['item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d30, ['item_id'], 'item_sales_level', 'item_count_istrade' + flag)

    # 空值替换
    ftrain['item_count_istrade' + flag].replace(np.nan, 0, inplace=True)
    ftrain['item_count' + flag].replace(np.nan, 0, inplace=True)

    # shop交易count
    d33 = train_feature_range[train_feature_range.is_trade == 1][['shop_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d33, ['shop_id'], 'item_sales_level', 'shop_count_istrade' + flag)

    ftrain['shop_count_istrade' + flag].replace(np.nan, 0, inplace=True)
    ftrain['shop_count' + flag].replace(np.nan, 0, inplace=True)

    # item_city_id的count
    d34 = train_feature_range[['item_city_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d34, ['item_city_id'], 'item_sales_level', 'item_city_id_count' + flag)

    # user_id,item_city_id的count
    d35 = train_feature_range[['user_id', 'item_city_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d35, ['item_city_id', 'user_id'], 'item_sales_level','user_item_city_id_count' + flag)

    # user和shop的次数
    d36 = train_feature_range[train_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d36, ['shop_id', 'user_id'], 'item_sales_level','user_shop_count_istrade' + flag)


    # 用户和广告出现的次数item_count  # 空值替换
    ftrain['item_count' + flag].replace(np.nan, 0, inplace=True)
    ftrain['item_count_istrade' + flag].replace(np.nan, 0, inplace=True)

    # user和item_brand_id的次数
    d37 = train_feature_range.groupby(['user_id', 'item_brand_id'], as_index=False)['item_sales_level'].agg({'user_item_brand_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d37, on=["item_brand_id", 'user_id'], how="left")

    # user with shop_star_level count
    d38 = train_feature_range.groupby(['user_id', 'shop_star_level'], as_index=False)['item_sales_level'].agg({'user_shop_star_level_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d38, on=["shop_star_level", 'user_id'], how="left")

    # user with item_sales_level count
    d39 = train_feature_range.groupby(['user_id', 'item_sales_level'], as_index=False)['item_sales_level'].agg({'user_item_sales_level_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d39, on=["item_sales_level", 'user_id'], how="left")

    # user with item_collected_level count
    d40 = train_feature_range.groupby(['user_id', 'item_collected_level'], as_index=False)['item_sales_level'].agg({'user_item_collected_level_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d40, on=["item_collected_level", 'user_id'], how="left")
    #
    # user with category count
    d41 = train_feature_range.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d41, on=["category", 'user_id'], how="left")

    #
    d44 = train_feature_range.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d44, on=["item_city_id", 'user_id', 'item_id'], how="left")

    ######  new
    d45 = train_feature_range[train_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d45 = d45.groupby(['shop_id', 'user_id', 'item_id'], as_index=False)['item_sales_level'].agg({'user_shop_item_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d45, on=['shop_id', 'user_id', 'item_id'], how="left")


    ####
    d46 = train_feature_range[train_feature_range.is_trade == 1][['user_id', 'item_id', 'item_city_id', 'item_sales_level']]
    d46 = d46.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d46, on=['user_id', 'item_id', 'item_city_id'], how="left")


    ### user category
    d47 = train_feature_range[train_feature_range.is_trade == 1][['user_id', 'category', 'item_sales_level']]
    d47 = d47.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d47, on=['user_id', 'category'], how="left")

    # user_gender_id
    d48 = train_feature_range[train_feature_range.is_trade == 1][['user_gender_id', 'item_sales_level']]
    d48 = d48.groupby(['user_gender_id'], as_index=False)['item_sales_level'].agg({'user_gender_id_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d48, on=['user_gender_id'], how="left")


    # user_occupation_id
    d49 = train_feature_range[train_feature_range.is_trade == 1][['user_occupation_id', 'item_sales_level']]
    d49 = d49.groupby(['user_occupation_id'], as_index=False)['item_sales_level'].agg({'user_occupation_id_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d49, on=['user_occupation_id'], how="left")

    # user_age_level
    d50 = train_feature_range[['user_age_level', 'item_sales_level']]
    d50 = d50.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d50, on=['user_age_level'], how="left")

    d51 = train_feature_range[train_feature_range.is_trade == 1][['user_age_level', 'item_sales_level']]
    d51 = d51.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d51, on=['user_age_level'], how="left")

    # item_sales_level
    d52 = train_feature_range[['item_sales_level', 'context_timestamp']]
    d52 = d52.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d52, on=['item_sales_level'], how="left")

    d53 = train_feature_range[train_feature_range.is_trade == 1][['item_sales_level', 'context_timestamp']]
    d53 = d53.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d53, on=['item_sales_level'], how="left")


    # item_collected_level
    d54 = train_feature_range[train_feature_range.is_trade == 1][['item_collected_level', 'item_sales_level']]
    d54 = d54.groupby(['item_collected_level'], as_index=False)['item_sales_level'].agg({'item_collected_level_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d54, on=['item_collected_level'], how="left")

    # user_star_level
    d55 = train_feature_range[['user_star_level', 'item_sales_level']]
    d55 = d55.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d55, on=['user_star_level'], how="left")

    d56 = train_feature_range[train_feature_range.is_trade == 1][['user_star_level', 'item_sales_level']]
    d56 = d56.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d56, on=['user_star_level'], how="left")

    # item_pv_level
    d57 = train_feature_range[['item_pv_level', 'item_sales_level']]
    d57 = d57.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d57, on=['item_pv_level'], how="left")

    d58 = train_feature_range[train_feature_range.is_trade == 1][['item_pv_level', 'item_sales_level']]
    d58 = d58.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d58, on=['item_pv_level'], how="left")


    # shop_review_num_level
    d59 = train_feature_range[['shop_review_num_level', 'item_sales_level']]
    d59 = d59.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count' + flag: 'count'})
    ftrain = pd.merge(ftrain, d59, on=['shop_review_num_level'], how="left")

    d60 = train_feature_range[train_feature_range.is_trade == 1][['shop_review_num_level', 'item_sales_level']]
    d60 = d60.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d60, on=['shop_review_num_level'], how="left")

    # item_price_level
    d62 = train_feature_range[train_feature_range.is_trade == 1][['item_price_level', 'item_sales_level']]
    d62 = d62.groupby(['item_price_level'], as_index=False)['item_sales_level'].agg({'item_price_level_count_istrade' + flag: 'count'})
    ftrain = pd.merge(ftrain, d62, on=['item_price_level'], how="left")


    print("day06 over!")

    return ftrain



# rank
def day07_rank_click(train_label_range,ftrain):
    i = 100
    # 商铺出现的次数
    d1_label = train_label_range[['shop_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d1_label, ['shop_id'], 'item_sales_level','shop_count_label')

    # 用户和商铺出现的次数
    d2_label = train_label_range[['shop_id', 'user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d2_label, ['shop_id','user_id'], 'item_sales_level', 'user_shop_count_label')

    d2 = d2_label.groupby(['user_id'], as_index=False)['shop_id'].agg({'user_diff_shop_count': 'count'})
    ftrain = pd.merge(ftrain, d2, on=["user_id"], how="left")

    d3 = d2_label.groupby(['shop_id'], as_index=False)['user_id'].agg({'shop_diff_user_count': 'count'})
    ftrain = pd.merge(ftrain, d3, on=["shop_id"], how="left")

    # 用户出现次数
    d3_label = train_label_range[['user_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d3_label, ['user_id'], 'item_sales_level', 'user_count_label')

    # 广告出现的次数
    d4_label = train_label_range[['item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d4_label, ['item_id'], 'item_sales_level', 'item_count_label')

    # 该商家的特定广告出现的次数
    d5_label = train_label_range[['item_id', 'shop_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d5_label, ['item_id','shop_id'], 'item_sales_level', 'shop_item_count_label')

    # 特定用户和该广告出现的次数
    d6_label = train_label_range[['user_id', 'item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d6_label, ['item_id', 'user_id'], 'item_sales_level', 'user_item_count_label')

    d6 = d6_label.groupby(['user_id'], as_index=False)['item_id'].agg({'user_diff_item_count': 'count'})
    ftrain = pd.merge(ftrain, d6, on=["user_id"], how="left")

    d7 = d6_label.groupby(['item_id'], as_index=False)['user_id'].agg({'item_diff_user_count': 'count'})
    ftrain = pd.merge(ftrain, d7, on=["item_id"], how="left")

    # 用户和该用户的职业出现的次数
    d7_label = train_label_range[['user_id', 'user_occupation_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d7_label, ['user_occupation_id', 'user_id'], 'item_sales_level', 'user_and_user_occupation_count_label')

    # 用户和商铺,广告出现的次数
    d8_label = train_label_range[['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d8_label, ["shop_id", 'user_id', 'item_id'], 'item_sales_level','user_shop_item_count_label')

    train_label_range['day'] = train_label_range['context_timestamp'].map(lambda x: int(x[6:8]))
    train_label_range['hour'] = train_label_range['context_timestamp'].map(lambda x: int(x[8:10]))

    # user在每一个小时(24个小时)的的count个数
    d10_label = train_label_range[['user_id', 'day', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d10_label, ["user_id", 'day', 'hour'], 'item_sales_level','user_hour_count_label')

    # shop在每一个小时(24个小时)的的count个数
    d11_label = train_label_range[['shop_id', 'day', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d11_label, ["shop_id", 'day', 'hour'], 'item_sales_level','shop_hour_count_label')

    # item在每一个小时(24个小时)的的count个数
    d12_label = train_label_range[['item_id', 'day', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d12_label, ["item_id", 'day', 'hour'], 'item_sales_level','item_hour_count_label')



    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_user_item_rank_label'] = rank

    else:
        # user和item的context_stamp的排名(升序)
        ftrain['context_timestamp_user_item_rank_label'] = ftrain.groupby(['user_id', 'item_id'])['context_timestamp'].rank(ascending=True)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_user_item_rank_label'], open(path, 'wb'))




    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_user_item_rank_label_desc'] = rank
    else:
        # user和item的context_stamp的排名(降序)
        ftrain['context_timestamp_user_item_rank_label_desc'] = ftrain.groupby(['user_id', 'item_id'])['context_timestamp'].rank(ascending=False)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_user_item_rank_label_desc'], open(path, 'wb'))




    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_user_shop_rank_label'] = rank
    else:
        # user和shop的context_stamp的排名(升序)
        ftrain['context_timestamp_user_shop_rank_label'] = ftrain.groupby(['user_id', 'shop_id'])['context_timestamp'].rank(ascending=True)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_user_shop_rank_label'], open(path, 'wb'))



    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_user_shop_rank_label_desc'] = rank
    else:
        # user和shop的context_stamp的排名(降序)
        ftrain['context_timestamp_user_shop_rank_label_desc'] = ftrain.groupby(['user_id', 'shop_id'])['context_timestamp'].rank(ascending=False)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_user_shop_rank_label_desc'], open(path, 'wb'))



    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_rank_label'] = rank
    else:
        # 对同一个用户的时间进行排序(升序)
        ftrain['context_timestamp_rank_label'] = ftrain.groupby(['user_id'])['context_timestamp'].rank(ascending=True)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_rank_label'], open(path, 'wb'))



    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_rank_desc_label'] = rank
    else:
        # 对同一个用户的时间进行排序(降序)
        ftrain['context_timestamp_rank_desc_label'] = ftrain.groupby(['user_id'])['context_timestamp'].rank(ascending=False)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_rank_desc_label'], open(path, 'wb'))



    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_shop_rank_label'] = rank
    else:
        # 对同一个shop的时间进行排序(升序)
        ftrain['context_timestamp_shop_rank_label'] = ftrain.groupby(['shop_id'])['context_timestamp'].rank(ascending=True)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_shop_rank_label'], open(path, 'wb'))


    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_shop_rank_desc_label'] = rank
    else:
        # 对同一个shop的时间进行排序(降序)
        ftrain['context_timestamp_shop_rank_desc_label'] = ftrain.groupby(['shop_id'])['context_timestamp'].rank(ascending=False)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_shop_rank_desc_label'], open(path, 'wb'))


    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_item_rank_label'] = rank
    else:
        # item的context_stamp的排名(升序)
        ftrain['context_timestamp_item_rank_label'] = ftrain.groupby(['item_id'])['context_timestamp'].rank(ascending=True)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_item_rank_label'], open(path, 'wb'))



    i += 1
    path = '../rank_data/valfile' + str(i) + '.pkl'
    if os.path.exists(path):
        rank = pickle.load(open(path, 'rb'))
        rank = rank.values.reshape(-1, 1)
        rank = preprocess.MinMaxScaler().fit_transform(rank)
        ftrain['context_timestamp_item_rank_label_desc'] = rank
    else:
        # item的context_stamp的排名(降序)
        ftrain['context_timestamp_item_rank_label_desc'] = ftrain.groupby(['item_id'])['context_timestamp'].rank(ascending=False)
        ftrain.drop_duplicates(inplace=True)
        pickle.dump(ftrain['context_timestamp_item_rank_label_desc'], open(path, 'wb'))




    ftrain['aa_user_item_shop_rank_label'] = ftrain.groupby(['user_id', 'item_id', 'shop_id'])['context_timestamp'].rank(ascending=True)
    ftrain.drop_duplicates(inplace=True)

    ftrain['aa_user_item_shop_rank_desc_label'] = ftrain.groupby(['user_id', 'item_id', 'shop_id'])['context_timestamp'].rank(ascending=False)
    ftrain.drop_duplicates(inplace=True)

    # 广告商品页面展示标号
    d14_label = train_label_range[['context_page_id', 'item_id', 'item_price_level']]
    ftrain = stat_feature_count(ftrain, d14_label, ["context_page_id", 'item_id'], 'item_price_level','context_page_item_count_label')

    # user和shop店铺评价的均值
    d15_label = train_label_range[['user_id', 'shop_id', 'shop_review_num_level']]
    ftrain = stat_feature_count(ftrain, d15_label, ["user_id", 'shop_id'], 'shop_review_num_level','user_shop_mean_label')

    # user和item在每一个小时(24个小时)的的count个数
    d17_label = train_label_range[['user_id', 'item_id', 'day', 'hour', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d17_label, ['user_id', "item_id", 'day', 'hour'], 'item_sales_level','user_item_hour_count_label')

    # user和shop在每一个小时(24个小时)的的count个数
    d18_label = train_label_range[['user_id', 'shop_id', 'hour', 'day', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d18_label, ['user_id', "shop_id", 'hour', 'day'], 'item_sales_level','user_shop_hour_count_label')

    # user_id,item_city_id的count
    d20_label = train_label_range[['user_id', 'item_city_id', 'item_sales_level']]
    ftrain = stat_feature_count(ftrain, d20_label, ["item_city_id", 'user_id'], 'item_sales_level','user_item_city_id_count_label')

    # 用户上下一次点击广告的时间间隔
    d24_label = train_label_range[['item_id', 'user_id', 'context_timestamp']]
    d24_label = d24_label.groupby(['user_id', 'item_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index()
    d24_label.rename(columns={'context_timestamp': 'dates'}, inplace=True)
    ftrain = pd.merge(ftrain, d24_label, on=["user_id", "item_id"], how="left")


    ftrain['context_timestamp_and_dates'] = ftrain.context_timestamp.astype('str') + '-' + ftrain.dates
    ftrain['user_before_day_click_item_gap'] = ftrain.context_timestamp_and_dates.apply(get_day_gap_before)
    ftrain['user_after_day_click_item_gap'] = ftrain.context_timestamp_and_dates.apply(get_day_gap_after)

    # 用户是否是第一次点击特定广告
    ftrain["is_first_get_coupon"] = ftrain.context_timestamp_and_dates.apply(is_first_get_coupon)
    # 用户是否是最后一次点击特定广告
    ftrain["is_last_get_coupon"] = ftrain.context_timestamp_and_dates.apply(is_last_get_coupon)

    d25_label = train_label_range.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count_label': 'count'})
    ftrain = pd.merge(ftrain, d25_label, on=["category", 'user_id'], how="left")

    # process property
    ftrain['property_count'] = ftrain['item_property_list'].apply(get_property_info)
    ftrain['property'] = ftrain['property_count'].apply(lambda x: x.split(';')[0])
    ftrain['property_max_count'] = ftrain['property_count'].apply(lambda x: x.split(';')[1])
    del ftrain['property_count']

    # user with category rank
    ftrain['user_and_category_rank'] = ftrain.groupby(['user_id', 'category'])['context_timestamp'].rank(ascending=True)
    ftrain['user_and_category_rank_desc'] = ftrain.groupby(['user_id', 'category'])['context_timestamp'].rank(ascending=False)

    # gender_id with category
    test_range = train_label_range[train_label_range.user_gender_id != -1]
    d26_label = test_range.groupby(['user_gender_id', 'category'], as_index=False)['item_sales_level'].agg({'gender_category_count': 'count'})
    ftrain = pd.merge(ftrain, d26_label, on=["category", 'user_gender_id'], how="left")

    d27_label = train_label_range.groupby(['user_age_level', 'category'], as_index=False)['item_sales_level'].agg({'user_age_category_count': 'count'})
    ftrain = pd.merge(ftrain, d27_label, on=["category", 'user_age_level'], how="left")

    d28_label = train_label_range.groupby(['user_occupation_id', 'category'], as_index=False)['item_sales_level'].agg({'user_occupation_id_category_count': 'count'})
    ftrain = pd.merge(ftrain, d28_label, on=["category", 'user_occupation_id'], how="left")


    print('label_rank finished!')

    return ftrain



def stat_feature_count(data_label,data,key,value,feature_name):
    feature = data.groupby(key, as_index=False)[value].agg({feature_name: 'count'})
    ftrain = pd.merge(data_label, feature, on=key, how="left")
    return ftrain


def stat_feature_rate(click,imp):
    click = click.astype('float')
    imp = imp.astype('float')
    return click / imp


def stat_feature_rank(data,key,sort_method):
    rank = data.groupby(key)['context_timestamp'].rank(ascending=sort_method)
    return rank



def split_train_test(label):
    ftest = label[label.is_trade == -1]   # is_trade
    ftrain = label[label.is_trade != -1]

    del ftest['is_trade']
    ftest.to_csv('../data/ftest.csv', index=None)
    ftrain.to_csv('../data/ftrain.csv', index=None)


def get_category_intesection(strs):
    s = strs.split('-')
    s[0] = s[0].split(';')
    s[1] = s[1].split(';')
    res = list(set(list(s[0])).intersection(set(list(s[1]))))
    return len(res)


def get_property_intesection(strs):
    s = strs.split('-')
    s[0] = s[0].split(';')
    s[1] = s[1].split(',')
    res = list(set(list(s[0])).intersection(set(list(s[1]))))
    return len(res)

def date2morning_afternoon(context_timestamp):
    hour = context_timestamp[8:10]
    hour = int(hour)
    if (hour > 19 & hour < 21) or (hour > 10 & hour < 15):  #是高峰期
        return True
    else:
        return False


"""
是：1
否：0
"""

def is_first_get_coupon(strs):
    date_received, dates = strs.split("-")
    dates = dates.split(':')
    date_received = str(date_received)
    gaps = []
    for dt in dates:
        dt = str(dt)
        gap_days = datetime.datetime.strptime(dt, '%Y%m%d%H%M%S') - datetime.datetime.strptime(date_received, '%Y%m%d%H%M%S')
        gap_days = gap_days.days
        if gap_days < 0:
            return 0  # 不是第一次
        gaps.append(gap_days)
    if len(gaps) == 1:  # 是第一次，也是最后一次
        return 1
    return 1

"""
是最后一次：1
不是最后一次：0
"""

def is_last_get_coupon(strs):
    date_received, dates = strs.split("-")
    dates = dates.split(':')
    date_received = str(date_received)
    gaps = []
    for dt in dates:
        dt = str(dt)
        gap_days = datetime.datetime.strptime(date_received, '%Y%m%d%H%M%S') - datetime.datetime.strptime(dt, '%Y%m%d%H%M%S')
        gap_days = gap_days.days
        if gap_days < 0:
            return 0  # 不是最后一次
        gaps.append(gap_days)
    if len(gaps) == 1:  # 是第一次，也是最后一次
        return 1
    return 1


def get_day_gap_before(strs):
    date_received, dates = strs.split('-')
    dates = dates.split(':')
    gaps = []
    date_received = str(date_received)
    # print(date_received)
    for dt in dates:
        dt = str(dt)
        gap_days = datetime.datetime.strptime(date_received,'%Y%m%d%H%M%S') - datetime.datetime.strptime(dt,'%Y%m%d%H%M%S')
        gap_days = gap_days.seconds
        gap_days = round(gap_days / 3600, 2)

        if gap_days > 0.0:
            gaps.append(gap_days)
    if len(gaps) == 0.0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(strs):
    date_received, dates = strs.split('-')
    dates = dates.split(':')
    date_received = str(date_received)
    gaps = []
    print()
    for dt in dates:
        dt = str(dt)
        gap_days = datetime.datetime.strptime(dt, '%Y%m%d%H%M%S') - datetime.datetime.strptime(date_received, '%Y%m%d%H%M%S')
        gap_days = gap_days.seconds
        gap_days = round(gap_days/3600,2)

        if gap_days > 0.0:
            gaps.append(gap_days)
    if len(gaps) == 0.0:
        return -1
    else:
        return min(gaps)



def get_category_list(strs):
    arrays = strs.split(';')
    category_list = ''
    strlen = len(arrays)
    i = 0
    for ele in arrays:
        s = ele.split(':')
        if i == strlen-1:
            category_list = category_list + s[0]
        else:
            category_list = category_list + s[0] + ";"
        i += 1
    return category_list

def get_property_list(strs):
    arrays = strs.split(';')
    property_list = ''
    strlen = len(arrays)
    i = 0
    for ele in arrays:
        s = ele.split(':')
        if i == strlen-1:
            property_list = property_list + '-1'
        else:
            property_list = property_list + s[1] + ","
        i += 1
    return property_list

def get_category_rank(strs):
    count = 0
    s = strs.split("-")
    s2 = s[1].split(';')
    for ele in s2:
        count += 1
        if s[0] == ele:
            return count


def get_property_info(s):
    str_arrays = s.split(';')
    word_count = {}
    for ele in str_arrays:
        if ele in word_count:
            word_count[ele] += 1
        else:
            word_count[ele] = 1

    # reverse = true,降序,反之
    word_infos = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    property_count = str(word_infos[0][0]) + ";" + str(word_infos[0][1])
    return property_count

def pre_process_data(dframe, dframe_test):
    dframe['category'] = dframe['item_category_list'].apply(lambda x: x.split(";")[2] if len(x.split(";")) == 3 else x.split(";")[1])
    dframe_test['category'] = dframe_test['item_category_list'].apply(lambda x: x.split(";")[2] if len(x.split(";")) == 3 else x.split(";")[1])
    # dframe['category'] = dframe['item_category_list'].apply(lambda x: x.split(";")[1])
    # dframe_test['category'] = dframe_test['item_category_list'].apply(lambda x: x.split(";")[1])

    # 对全局数据集做category处理
    dframe['category_list'] = dframe['predict_category_property'].apply(get_category_list)
    dframe['category_list'] = dframe['category_list'].astype('str')
    dframe['category_category_list'] = dframe['category'] + '-' + dframe['category_list']
    dframe['category_rank'] = dframe['category_category_list'].apply(get_category_rank)
    # 对全局数据集做category处理
    dframe_test['category_list'] = dframe_test['predict_category_property'].apply(get_category_list)
    dframe_test['category_list'] = dframe_test['category_list'].astype('str')
    dframe_test['category_category_list'] = dframe_test['category'] + '-' + dframe_test['category_list']
    dframe_test['category_rank'] = dframe_test['category_category_list'].apply(get_category_rank)


    # inserction item_category_list with categoryo_list
    dframe['item_category_preditct_category'] = dframe['item_category_list'] + '-' + dframe['category_list']
    dframe['item_category_with_predict_category_intersection_len'] = dframe['item_category_preditct_category'].apply(get_category_intesection)
    dframe_test['item_category_preditct_category'] = dframe_test['item_category_list'] + '-' + dframe_test['category_list']
    dframe_test['item_category_with_predict_category_intersection_len'] = dframe_test['item_category_preditct_category'].apply(get_category_intesection)
    dframe['preditct_property_list'] = dframe['predict_category_property'].apply(get_property_list)
    dframe['item_property_preditct_property'] = dframe['item_property_list'] + '-' + dframe['preditct_property_list']
    dframe['predict_property_jiaoji_item_property'] = dframe['item_property_preditct_property'].apply(get_property_intesection)
    dframe_test['preditct_property_list'] = dframe_test['predict_category_property'].apply(get_property_list)
    dframe_test['item_property_preditct_property'] = dframe_test['item_property_list'] + '-' + dframe_test['preditct_property_list']
    dframe_test['predict_property_jiaoji_item_property'] = dframe_test['item_property_preditct_property'].apply(get_property_intesection)

    del dframe['item_property_preditct_property']
    del dframe_test['item_property_preditct_property']
    del dframe['preditct_property_list']
    del dframe_test['preditct_property_list']
    del dframe['category_category_list']
    del dframe_test['category_list']
    del dframe['item_category_preditct_category']
    del dframe_test['item_category_preditct_category']
    del dframe['category_list']
    del dframe_test['category_category_list']

    dframe['category_rank'].replace(np.nan, -1, inplace=True)
    dframe_test['category_rank'].replace(np.nan, -1, inplace=True)

    dframe['item_category_with_predict_category_intersection_len'].replace(np.nan, -1, inplace=True)
    dframe_test['item_category_with_predict_category_intersection_len'].replace(np.nan, -1, inplace=True)

    dframe['predict_property_jiaoji_item_property'].replace(np.nan, -1, inplace=True)
    dframe_test['predict_property_jiaoji_item_property'].replace(np.nan, -1, inplace=True)

    dframe_test['context_timestamp'] = dframe_test['context_timestamp'].astype("str")
    dframe['context_timestamp'] = dframe['context_timestamp'].astype("str")

    return dframe,dframe_test




def main():
    train_feature_range_click_imp_and_rate,train_label_range = load_csv()

    ftrain = day31today5_click_imp_and_rate(train_feature_range_click_imp_and_rate,train_label_range,'day31-05')

    ftrain = day07_rank_click(train_label_range,ftrain)

    split_train_test(ftrain)

if __name__ == '__main__':
    main()