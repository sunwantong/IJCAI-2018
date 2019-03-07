import pandas as pd
import numpy as np
import xgboost as xgb
import scipy as sp
import time
import datetime
from com.sun.tong.IJCAI.Baysian import *
import os as os
import pickle
from concurrent.futures import ProcessPoolExecutor,wait, as_completed

# def new_split_data():
#     dframe = pd.read_csv('data/train.csv', sep=",")
#     dframe_test = pd.read_csv('data/round1_ijcai_18_test_b_20180418.csv', sep=",")
#
#     dframe['category'] = dframe['item_category_list'].apply(lambda x: x.split(";")[2] if len(x.split(";")) == 3 else x.split(";")[1])
#     dframe_test['category'] = dframe_test['item_category_list'].apply(lambda x: x.split(";")[2] if len(x.split(";")) == 3 else x.split(";")[1])
#
#     # dframe['category'] = dframe['item_category_list'].apply(lambda x: x.split(";")[1])
#     # dframe_test['category'] = dframe_test['item_category_list'].apply(lambda x: x.split(";")[1])
#
#     # 对全局数据集做category处理
#     dframe['category_list'] = dframe['predict_category_property'].apply(get_category_list)
#     dframe['category_list'] = dframe['category_list'].astype('str')
#     dframe['category_category_list'] = dframe['category'] + '-' + dframe['category_list']
#     dframe['category_rank'] = dframe['category_category_list'].apply(get_category_rank)
#
#     # 对全局数据集做category处理
#     dframe_test['category_list'] = dframe_test['predict_category_property'].apply(get_category_list)
#     dframe_test['category_list'] = dframe_test['category_list'].astype('str')
#     dframe_test['category_category_list'] = dframe_test['category'] + '-' + dframe_test['category_list']
#     dframe_test['category_rank'] = dframe_test['category_category_list'].apply(get_category_rank)
#
#     # inserction item_category_list with categoryo_list
#     dframe['item_category_preditct_category'] = dframe['item_category_list'] + '-' + dframe['category_list']
#     dframe['item_category_with_predict_category_intersection_len'] = dframe['item_category_preditct_category'].apply(get_category_intesection)
#
#     dframe_test['item_category_preditct_category'] = dframe_test['item_category_list'] + '-' + dframe_test['category_list']
#     dframe_test['item_category_with_predict_category_intersection_len'] = dframe_test['item_category_preditct_category'].apply(get_category_intesection)
#
#     del dframe['category_category_list']
#     del dframe_test['category_list']
#
#     del dframe['item_category_preditct_category']
#     del dframe_test['item_category_preditct_category']
#
#     del dframe['category_list']
#     del dframe_test['category_category_list']
#
#     dframe['category_rank'].replace(np.nan, -1, inplace=True)
#     dframe_test['category_rank'].replace(np.nan, -1, inplace=True)
#
#     dframe['item_category_with_predict_category_intersection_len'].replace(np.nan, -1, inplace=True)
#     dframe_test['item_category_with_predict_category_intersection_len'].replace(np.nan, -1, inplace=True)
#
#
#     dframe_test['context_timestamp'] = dframe_test['context_timestamp'].astype("str")
#     dframe['context_timestamp'] = dframe['context_timestamp'].astype("str")
#
#
#     # 训练2
#     train_feature_range2 = dframe[(dframe['context_timestamp'] >= '20180920000000') &
#                                   (dframe['context_timestamp'] <= '20180922235959')]
#     train_label_range2 = dframe[(dframe['context_timestamp'] >= '20180922000000') &
#                                 (dframe['context_timestamp'] <= '20180922235959')]
#     # 训练3
#     train_feature_range3 = dframe[(dframe['context_timestamp'] >= '20180919000000') &
#                                   (dframe['context_timestamp'] <= '20180921235959')]
#     train_label_range3 = dframe[(dframe['context_timestamp'] >= '20180921000000') &
#                                 (dframe['context_timestamp'] <= '20180921235959')]
#     # 训练4
#     train_feature_range4 = dframe[(dframe['context_timestamp'] >= '20180918000000') &
#                                   (dframe['context_timestamp'] <= '20180920235959')]
#     train_label_range4 = dframe[(dframe['context_timestamp'] >= '20180920000000') &
#                                 (dframe['context_timestamp'] <= '20180920235959')]
#
#     #验证
#     validate_feature_range = dframe[(dframe['context_timestamp'] >= '20180921000000') &
#                                     (dframe['context_timestamp'] <= '20180923235959')]
#     validate_label_range = dframe[(dframe['context_timestamp'] >= '20180924000000') &
#                                   (dframe['context_timestamp'] <= '20180924235959')]
#
#     # 测试
#     test_feature_range = dframe[(dframe['context_timestamp'] >= '20180922000000') &
#                                 (dframe['context_timestamp'] <= '20180924235959')]
#     test_label_range = dframe_test
#
#
#
#     return train_feature_range2,train_label_range2, \
#            train_feature_range3, train_label_range3, \
#            train_feature_range4, train_label_range4, \
#            validate_feature_range,validate_label_range,\
#            test_feature_range,test_label_range


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

def load_data_one():
    dframe = pd.read_csv('data/train.csv', sep=",")
    dframe_test = pd.read_csv('data/test_a_b.csv', sep=",")

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

    #训练1
    train_feature_range1 = dframe[(dframe['context_timestamp'] >= '20180921000000') &
                                 (dframe['context_timestamp'] <= '20180922235959')]
    train_label_range1 = dframe[(dframe['context_timestamp'] >= '20180923000000') &
                               (dframe['context_timestamp'] <= '20180923235959')]
    # 训练2
    train_feature_range2 = dframe[(dframe['context_timestamp'] >= '20180920000000') &
                                  (dframe['context_timestamp'] <= '20180921235959')]
    train_label_range2 = dframe[(dframe['context_timestamp'] >= '20180922000000') &
                                (dframe['context_timestamp'] <= '20180922235959')]
    # 训练3
    train_feature_range3 = dframe[(dframe['context_timestamp'] >= '20180919000000') &
                                  (dframe['context_timestamp'] <= '20180920235959')]
    train_label_range3 = dframe[(dframe['context_timestamp'] >= '20180921000000') &
                                (dframe['context_timestamp'] <= '20180921235959')]
    # 训练4
    train_feature_range4 = dframe[(dframe['context_timestamp'] >= '20180918000000') &
                                  (dframe['context_timestamp'] <= '20180919235959')]
    train_label_range4 = dframe[(dframe['context_timestamp'] >= '20180920000000') &
                                (dframe['context_timestamp'] <= '20180920235959')]

    #验证
    validate_feature_range = dframe[(dframe['context_timestamp'] >= '20180922000000') &
                                    (dframe['context_timestamp'] <= '20180923235959')]
    validate_label_range = dframe[(dframe['context_timestamp'] >= '20180924000000') &
                                  (dframe['context_timestamp'] <= '20180924235959')]

    # 测试
    test_feature_range = dframe[(dframe['context_timestamp'] >= '20180923000000') &
                                (dframe['context_timestamp'] <= '20180924235959')]
    test_label_range = dframe_test



    return train_feature_range1,train_label_range1,\
           train_feature_range2,train_label_range2, \
           train_feature_range3, train_label_range3, \
           train_feature_range4, train_label_range4, \
           validate_feature_range,validate_label_range,\
           test_feature_range,test_label_range

def get_rank_context_timestamp(df):
    df['context_timestamp_rank'] = df['context_timestamp'].rank(ascending=True, method="average")
    df = df.reset_index(drop=True)
    return df[['context_timestamp','context_timestamp_rank']]


def get_rank_context_timestamp_descend(df):
    df['context_timestamp_rank_desc'] = df['context_timestamp'].rank(ascending=False, method="average")
    df = df.reset_index(drop=True)
    return df[['context_timestamp','context_timestamp_rank_desc']]

def date2morning_afternoon(context_timestamp):
    hour = context_timestamp[8:10]
    hour = int(hour)
    if (hour > 19 & hour < 21) or (hour > 10 & hour < 15):  #是高峰期
        return True
    else:
        return False

def is_weekday(context_timestamp):
    # str -> 日期
    def string_toDatetime(string):
        return datetime.datetime.strptime(string, "%Y%m%d%H%M%S")

    context_timestamp = string_toDatetime(context_timestamp)
    day = context_timestamp.weekday()
    if day == 5 or day == 6:
        return 1
    else:
        return 0


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


# 贝叶斯平滑
def Bayesian_smoooth(ftrain_istrade_count,ftrain_total_count,flag):
    path = 'data/file'+str(flag)+'.pkl'
    if os.path.exists(path):
        smooth_rate = pickle.load(open(path, 'rb'))
    else:
        bs = BayesianSmoothing(1, 1)
        bs.update(ftrain_total_count.values, ftrain_istrade_count.values, 1000, 0.001)
        smooth_rate = (ftrain_istrade_count + bs.alpha) / (ftrain_total_count + bs.alpha + bs.beta)
        pickle.dump(smooth_rate,open(path,'wb'))

    smooth_rate.replace(np.nan, -1, inplace=True)
    return smooth_rate

def extract_train_feature(train_feature_range,train_label_range,flag):

    """
        特征区间提取特征
    """
    # 商铺出现的次数
    d1 = train_feature_range[['shop_id', 'item_sales_level']]
    d1 = d1.groupby(['shop_id']).agg('count').reset_index()
    d1.rename(columns={'item_sales_level': 'shop_count'}, inplace=True)
    ftrain = pd.merge(train_label_range, d1, on="shop_id", how="left")

    # 用户和商铺出现的次数
    d2 = train_feature_range[['shop_id', 'user_id', 'item_sales_level']]
    d2 = d2.groupby(['shop_id', 'user_id']).agg('count').reset_index()
    d2.rename(columns={'item_sales_level': 'user_shop_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d2, on=["shop_id", 'user_id'], how="left")


    # 用户出现次数(总的次数)
    d3 = train_feature_range[['user_id', 'item_sales_level']]
    d3 = d3.groupby(['user_id']).agg('count').reset_index()
    d3.rename(columns={'item_sales_level': 'user_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d3, on="user_id", how="left")


    # 用户出现次数(产生交易的次数)
    d4 = train_feature_range[train_feature_range['is_trade'] == 1][['user_id', 'item_sales_level']]
    d4 = d4.groupby(['user_id']).agg('count').reset_index()
    d4.rename(columns={'item_sales_level': 'user_count_istrade'}, inplace=True)
    ftrain = pd.merge(ftrain, d4, on="user_id", how="left")


    #空值替换
    ftrain['user_count_istrade'].replace(np.nan,0,inplace=True)
    ftrain['user_count'].replace(np.nan,0,inplace=True)

    #交易率
    ftrain['user_istrade_rate'] = ftrain['user_count_istrade'].astype('float') / ftrain['user_count'].astype('float')

    #贝叶斯平滑
    ftrain['user_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['user_count_istrade'],ftrain['user_count'],'train1'+flag)
    del ftrain['user_istrade_rate']


    # 广告出现的次数
    d5 = train_feature_range[['item_id', 'item_sales_level']]
    d5 = d5.groupby(['item_id']).agg('count').reset_index()
    d5.rename(columns={'item_sales_level': 'item_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d5, on="item_id", how="left")

    # 该商家的特定广告出现的次数
    d6 = train_feature_range[['item_id', 'shop_id', 'item_sales_level']]
    d6 = d6.groupby(['item_id', 'shop_id']).agg('count').reset_index()
    d6.rename(columns={'item_sales_level': 'shop_item_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d6, on=["item_id", 'shop_id'], how="left")

    # 特定用户和该广告出现的次数
    d7 = train_feature_range[['user_id', 'item_id', 'item_sales_level']]
    d7 = d7.groupby(['user_id', 'item_id']).agg('count').reset_index()
    d7.rename(columns={'item_sales_level': 'user_item_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d7, on=["user_id", 'item_id'], how="left")

    # 特定用户和该广告出现的次数(交易成功)
    d8 = train_feature_range[train_feature_range.is_trade == 1][['user_id', 'item_id', 'item_sales_level']]
    d8 = d8.groupby(['user_id', 'item_id']).agg('count').reset_index()
    d8.rename(columns={'item_sales_level': 'user_item_count_istrade'}, inplace=True)
    ftrain = pd.merge(ftrain, d8, on=["user_id", 'item_id'], how="left")


    # 空值替换
    ftrain['user_item_count'].replace(np.nan,0, inplace=True)
    ftrain['user_item_count_istrade'].replace(np.nan,0, inplace=True)

    # 交易率
    ftrain['user_item_count_istrade_rate'] = ftrain['user_item_count_istrade'].astype('float') / ftrain['user_item_count'].astype('float')

    # # 贝叶斯平滑
    ftrain['user_item_count_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['user_item_count_istrade'], ftrain['user_item_count'], 'train2'+flag)
    del ftrain['user_item_count_istrade_rate']

    # 用户和该用户的职业出现的次数
    d9 = train_feature_range[['user_id', 'user_occupation_id', 'item_sales_level']]
    d9 = d9.groupby(['user_id', 'user_occupation_id']).agg('count').reset_index()
    d9.rename(columns={'item_sales_level': 'user_and_user_occupation_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d9, on=["user_id", 'user_occupation_id'], how="left")

    # 该职业出现的次数
    d13 = train_feature_range[['user_occupation_id', 'item_price_level']]
    d13 = d13.groupby(['user_occupation_id']).agg('count').reset_index()
    d13.rename(columns={'item_price_level': 'user_occupation_id_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d13, on="user_occupation_id", how="left")

    # 广告价格等级出现的次数
    d14 = train_feature_range[['item_price_level', 'user_id']]
    d14 = d14.groupby(['item_price_level']).agg('count').reset_index()
    d14.rename(columns={'user_id': 'item_price_level_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d14, on="item_price_level", how="left")

    # 性别对于是否交易的影响
    d15 = train_feature_range[['user_gender_id', 'user_id']]
    d15 = d15.groupby(['user_gender_id']).agg('count').reset_index()
    d15.rename(columns={'user_id': 'user_gender_id_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d15, on="user_gender_id", how="left")

    # 看当前日期是否是高峰期
    ftrain['is_high_trade'] = ftrain['context_timestamp'].apply(date2morning_afternoon)

    # 收藏级别出现的次数
    d17 = train_feature_range[['item_collected_level', 'item_price_level']]
    d17 = d17.groupby(['item_collected_level']).agg('count').reset_index()
    d17.rename(columns={'item_price_level': 'item_collected_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d17, on="item_collected_level", how="left")

    # 广告商品页面展示标号
    d18 = train_feature_range[['context_page_id', 'item_id', 'item_price_level']]
    d18 = d18.groupby(['context_page_id', 'item_id']).agg('count').reset_index()
    d18.rename(columns={'item_price_level': 'context_page_item_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d18, on=["context_page_id", 'item_id'], how="left")

    # 用户和商铺,广告出现的次数
    d20 = train_feature_range[['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d20 = d20.groupby(['shop_id', 'user_id', 'item_id']).agg('count').reset_index()
    d20.rename(columns={'item_sales_level': 'user_shop_item_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d20, on=["shop_id", 'user_id', 'item_id'], how="left")


    train_feature_range['day'] = train_feature_range['context_timestamp'].map(lambda x: int(x[6:8]))
    train_feature_range['hour'] = train_feature_range['context_timestamp'].map(lambda x: int(x[8:10]))

    ftrain['day'] = ftrain['context_timestamp'].map(lambda x: int(x[6:8]))
    ftrain['hour'] = ftrain['context_timestamp'].map(lambda x: int(x[8:10]))

    # user和shop在每一个小时(24个小时)的的count个数
    d21 = train_feature_range[['user_id', 'shop_id', 'hour', 'item_sales_level']]
    d21 = d21.groupby(['user_id', 'shop_id', 'hour']).agg('count').reset_index()
    d21.rename(columns={'item_sales_level': 'user_shop_hour_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d21, on=['user_id', "shop_id", 'hour'], how="left")

    # user在每一个小时(24个小时)的的count个数
    d22 = train_feature_range[['user_id','hour','item_sales_level']]
    d22 = d22.groupby(['user_id','hour']).agg('count').reset_index()
    d22.rename(columns={'item_sales_level': 'user_hour_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d22, on=["user_id",'hour'], how="left")

    # shop在每一个小时(24个小时)的的count个数
    d23 = train_feature_range[['shop_id','hour', 'item_sales_level']]
    d23 = d23.groupby(['shop_id', 'hour']).agg('count').reset_index()
    d23.rename(columns={'item_sales_level': 'shop_hour_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d23, on=["shop_id",'hour'], how="left")

    # item在每一个小时(24个小时)的的count个数
    d24 = train_feature_range[['item_id','hour', 'item_sales_level']]
    d24 = d24.groupby(['item_id','hour']).agg('count').reset_index()
    d24.rename(columns={'item_sales_level': 'item_hour_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d24, on=["item_id",'hour'], how="left")

    # user和item在每一个小时(24个小时)的的count个数
    d29 = train_feature_range[['user_id', 'item_id', 'hour','item_sales_level']]
    d29 = d29.groupby(['user_id', 'item_id', 'hour']).agg('count').reset_index()
    d29.rename(columns={'item_sales_level': 'user_item_hour_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d29, on=['user_id', "item_id", 'hour'], how="left")

    # 用户和广告出现的次数
    d30 = train_feature_range[train_feature_range.is_trade == 1][['item_id', 'item_sales_level']]
    d30 = d30.groupby(['item_id']).agg('count').reset_index()
    d30.rename(columns={'item_sales_level': 'item_count_istrade'}, inplace=True)
    ftrain = pd.merge(ftrain, d30, on=["item_id"], how="left")

    # 空值替换
    ftrain['item_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['item_count'].replace(np.nan, 0, inplace=True)

    ftrain['item_istrade_rate'] = ftrain.item_count_istrade.astype('float') / ftrain.item_count.astype('float')

    # # 贝叶斯平滑
    ftrain['item_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['item_count_istrade'],ftrain['item_count'], 'train3'+flag)
    del ftrain['item_istrade_rate']

    # shop交易count
    d33 = train_feature_range[train_feature_range.is_trade == 1][['shop_id', 'item_sales_level']]
    d33 = d33.groupby(['shop_id']).agg('count').reset_index()
    d33.rename(columns={'item_sales_level': 'shop_count_istrade'}, inplace=True)
    ftrain = pd.merge(ftrain, d33, on=["shop_id"], how="left")

    ftrain['shop_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['shop_count'].replace(np.nan, 0, inplace=True)

    ftrain['shop_istrade_rate'] = ftrain.shop_count_istrade.astype('float') / ftrain.shop_count.astype('float')

    # # 贝叶斯平滑
    ftrain['shop_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['shop_count_istrade'], ftrain['shop_count'],'train4'+flag)
    del ftrain['shop_istrade_rate']


    # item_city_id的count
    d34 = train_feature_range[['item_city_id', 'item_sales_level']]
    d34 = d34.groupby(['item_city_id']).agg('count').reset_index()
    d34.rename(columns={'item_sales_level': 'item_city_id_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d34, on="item_city_id", how="left")

    # user_id,item_city_id的count
    d35 = train_feature_range[['user_id', 'item_city_id', 'item_sales_level']]
    d35 = d35.groupby(['user_id', 'item_city_id']).agg('count').reset_index()
    d35.rename(columns={'item_sales_level': 'user_item_city_id_count'}, inplace=True)
    ftrain = pd.merge(ftrain, d35, on=["item_city_id", 'user_id'], how="left")

    # user和shop的次数
    d36 = train_feature_range[train_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_sales_level']]
    d36 = d36.groupby(['shop_id', 'user_id']).agg('count').reset_index()
    d36.rename(columns={'item_sales_level': 'user_shop_count_istrade'}, inplace=True)
    ftrain = pd.merge(ftrain, d36, on=["shop_id", 'user_id'], how="left")


    ftrain['user_shop_count'].replace(np.nan, 0, inplace=True)
    ftrain['user_shop_count_istrade'].replace(np.nan, 0, inplace=True)

    # ftrain['user_shop_count_minus_user_shop_istrade'] = ftrain.user_shop_count.astype('int') - ftrain.user_shop_count_istrade.astype('int')
    ftrain['user_shop_count_istrade_rate'] = ftrain.user_shop_count_istrade.astype('float') / ftrain.user_shop_count.astype('float')

    # # 贝叶斯平滑
    ftrain['user_shop_count_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['user_shop_count_istrade'], ftrain['user_shop_count'],'train5'+flag)
    del ftrain['user_shop_count_istrade_rate']


    # 用户和广告出现的次数item_count  # 空值替换
    ftrain['item_count'].replace(np.nan,0, inplace=True)
    ftrain['item_count_istrade'].replace(np.nan,0, inplace=True)
    #
    # ftrain['item_count_minus_item_istrade'] = ftrain.item_count.astype('int') - ftrain.item_count_istrade.astype('int')
    # ftrain['user_item_count_minus_user_item_istrade'] = ftrain['user_item_count'].astype('int') - ftrain['user_item_count_istrade'].astype('int')
    # ftrain['user_count_minus_user_count_istrade'] = ftrain['user_count'].astype('int') - ftrain['user_count_istrade'].astype('int')
    #
    # user和item_brand_id的次数
    d37 = train_feature_range.groupby(['user_id','item_brand_id'], as_index=False)['item_sales_level'].agg({'user_item_brand_count': 'count'})
    ftrain = pd.merge(ftrain, d37, on=["item_brand_id", 'user_id'], how="left")

    # user with shop_star_level count
    d38 = train_feature_range.groupby(['user_id','shop_star_level'],as_index=False)['item_sales_level'].agg({'user_shop_star_level_count': 'count'})
    ftrain = pd.merge(ftrain, d38, on=["shop_star_level", 'user_id'], how="left")

    # user with item_sales_level count
    d39 = train_feature_range.groupby(['user_id', 'item_sales_level'], as_index=False)['item_sales_level'].agg({'user_item_sales_level_count': 'count'})
    ftrain = pd.merge(ftrain, d39, on=["item_sales_level", 'user_id'], how="left")

    # user with item_collected_level count
    d40 = train_feature_range.groupby(['user_id', 'item_collected_level'], as_index=False)['item_sales_level'].\
        agg({'user_item_collected_level_count': 'count'})
    ftrain = pd.merge(ftrain, d40, on=["item_collected_level", 'user_id'], how="left")
    #
    # user with category count
    d41 = train_feature_range.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count': 'count'})
    ftrain = pd.merge(ftrain, d41, on=["category", 'user_id'], how="left")

    #
    d44 = train_feature_range.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count': 'count'})
    ftrain = pd.merge(ftrain, d44, on=["item_city_id", 'user_id','item_id'], how="left")



    ######  new
    d45 = train_feature_range[train_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d45 = d45.groupby(['shop_id', 'user_id', 'item_id'], as_index=False)['item_sales_level'].agg({'user_shop_item_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d45, on=['shop_id', 'user_id', 'item_id'], how="left")

    ftrain['user_shop_item_count'].replace(np.nan, 0, inplace=True)
    ftrain['user_shop_item_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['user_shop_item_count_istrade_rate'] = ftrain.user_shop_item_count_istrade.astype('float') / ftrain.user_shop_item_count.astype('float')

    # # 贝叶斯平滑
    ftrain['user_shop_item_count_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['user_shop_item_count_istrade'],ftrain['user_shop_item_count'], 'train6' + flag)
    del ftrain['user_shop_item_count_istrade_rate']


    ####
    d46 = train_feature_range[train_feature_range.is_trade == 1][['user_id', 'item_id', 'item_city_id', 'item_sales_level']]
    d46 = d46.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d46, on=['user_id', 'item_id', 'item_city_id'], how="left")

    ftrain['user_item_item_city_id_count'].replace(np.nan, 0, inplace=True)
    ftrain['user_item_item_city_id_count_istrade'].replace(np.nan, 0, inplace=True)

    ftrain['user_item_item_city_id_count_istrade_rate'] = ftrain.user_item_item_city_id_count_istrade.astype('float') / ftrain.user_item_item_city_id_count.astype('float')

    #  贝叶斯平滑
    ftrain['user_item_item_city_id_count_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['user_item_item_city_id_count_istrade'],ftrain['user_item_item_city_id_count'],
                                                                                  'train7' + flag)
    del ftrain['user_item_item_city_id_count_istrade_rate']

    ### user category
    d47 = train_feature_range[train_feature_range.is_trade == 1][['user_id', 'category','item_sales_level']]
    d47 = d47.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d47, on=['user_id', 'category'], how="left")

    ftrain['user_category_count'].replace(np.nan, 0, inplace=True)
    ftrain['user_category_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['user_category_count_istrade_rate'] = ftrain.user_category_count_istrade.astype('float') / ftrain.user_category_count.astype('float')
    #  贝叶斯平滑
    ftrain['user_category_count_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['user_category_count_istrade'], ftrain['user_category_count'],'train8' + flag)
    del ftrain['user_category_count_istrade_rate']

    # user_gender_id
    d48 = train_feature_range[train_feature_range.is_trade == 1][['user_gender_id', 'item_sales_level']]
    d48 = d48.groupby(['user_gender_id'], as_index=False)['item_sales_level'].agg({'user_gender_id_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d48, on=['user_gender_id'], how="left")

    ftrain['user_gender_id_count'].replace(np.nan, 0, inplace=True)
    ftrain['user_gender_id_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['user_gender_id_count_istrade_rate'] = ftrain.user_gender_id_count_istrade.astype('float') / ftrain.user_gender_id_count.astype('float')

    # user_occupation_id
    d49 = train_feature_range[train_feature_range.is_trade == 1][['user_occupation_id', 'item_sales_level']]
    d49 = d49.groupby(['user_occupation_id'], as_index=False)['item_sales_level'].agg({'user_occupation_id_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d49, on=['user_occupation_id'], how="left")

    ftrain['user_occupation_id_count'].replace(np.nan, 0, inplace=True)
    ftrain['user_occupation_id_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['user_occupation_id_count_istrade_rate'] = ftrain.user_occupation_id_count_istrade.astype('float') / ftrain.user_occupation_id_count.astype('float')

    # user_age_level
    d50 = train_feature_range[['user_age_level', 'item_sales_level']]
    d50 = d50.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count': 'count'})
    ftrain = pd.merge(ftrain, d50, on=['user_age_level'], how="left")

    d51 = train_feature_range[train_feature_range.is_trade == 1][['user_age_level', 'item_sales_level']]
    d51 = d51.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d51, on=['user_age_level'], how="left")

    ftrain['user_age_level_count'].replace(np.nan, 0, inplace=True)
    ftrain['user_age_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['user_age_level_count_istrade_rate'] = ftrain.user_age_level_count_istrade.astype('float') / ftrain.user_age_level_count.astype('float')


    # item_sales_level
    d52 = train_feature_range[['item_sales_level', 'context_timestamp']]
    d52 = d52.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count': 'count'})
    ftrain = pd.merge(ftrain, d52, on=['item_sales_level'], how="left")

    d53 = train_feature_range[train_feature_range.is_trade == 1][['item_sales_level', 'context_timestamp']]
    d53 = d53.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d53, on=['item_sales_level'], how="left")

    ftrain['item_sales_level_count'].replace(np.nan, 0, inplace=True)
    ftrain['item_sales_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['item_sales_level_count_istrade_rate'] = ftrain.item_sales_level_count_istrade.astype('float') / ftrain.item_sales_level_count.astype('float')


    # item_collected_level
    d54 = train_feature_range[train_feature_range.is_trade == 1][['item_collected_level', 'item_sales_level']]
    d54 = d54.groupby(['item_collected_level'], as_index=False)['item_sales_level'].agg({'item_collected_level_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d54, on=['item_collected_level'], how="left")

    ftrain['item_collected_count'].replace(np.nan, 0, inplace=True)
    ftrain['item_collected_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['item_collected_level_count_istrade_rate'] = ftrain.item_collected_level_count_istrade.astype('float') / ftrain.item_collected_count.astype('float')

    # user_star_level
    d55 = train_feature_range[['user_star_level', 'item_sales_level']]
    d55 = d55.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count': 'count'})
    ftrain = pd.merge(ftrain, d55, on=['user_star_level'], how="left")

    d56 = train_feature_range[train_feature_range.is_trade == 1][['user_star_level', 'item_sales_level']]
    d56 = d56.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d56, on=['user_star_level'], how="left")

    ftrain['user_star_level_count'].replace(np.nan, 0, inplace=True)
    ftrain['user_star_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['user_star_level_count_istrade_rate'] = ftrain.user_star_level_count_istrade.astype('float') / ftrain.user_star_level_count.astype('float')

    # item_pv_level
    d57 = train_feature_range[['item_pv_level', 'item_sales_level']]
    d57 = d57.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count': 'count'})
    ftrain = pd.merge(ftrain, d57, on=['item_pv_level'], how="left")

    d58 = train_feature_range[train_feature_range.is_trade == 1][['item_pv_level', 'item_sales_level']]
    d58 = d58.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d58, on=['item_pv_level'], how="left")

    ftrain['item_pv_level_count'].replace(np.nan, 0, inplace=True)
    ftrain['item_pv_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['item_pv_level_count_istrade_rate'] = ftrain.item_pv_level_count_istrade.astype('float') / ftrain.item_pv_level_count.astype('float')

    #  贝叶斯平滑
    ftrain['item_pv_level_count_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['item_pv_level_count_istrade'],ftrain['item_pv_level_count'], 'train9' + flag)
    del ftrain['item_pv_level_count_istrade_rate']


    # shop_review_num_level
    d59 = train_feature_range[['shop_review_num_level', 'item_sales_level']]
    d59 = d59.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count': 'count'})
    ftrain = pd.merge(ftrain, d59, on=['shop_review_num_level'], how="left")

    d60 = train_feature_range[train_feature_range.is_trade == 1][['shop_review_num_level', 'item_sales_level']]
    d60 = d60.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d60, on=['shop_review_num_level'], how="left")

    ftrain['shop_review_num_level_count'].replace(np.nan, 0, inplace=True)
    ftrain['shop_review_num_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['shop_review_num_level_count_istrade_rate'] = ftrain.shop_review_num_level_count_istrade.astype('float') / ftrain.shop_review_num_level_count.astype('float')

    #  贝叶斯平滑
    ftrain['shop_review_num_level_count_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['shop_review_num_level_count_istrade'],ftrain['shop_review_num_level_count'],'train10' + flag)
    del ftrain['shop_review_num_level_count_istrade_rate']

    # item_price_level
    d62 = train_feature_range[train_feature_range.is_trade == 1][['item_price_level', 'item_sales_level']]
    d62 = d62.groupby(['item_price_level'], as_index=False)['item_sales_level'].agg({'item_price_level_count_istrade': 'count'})
    ftrain = pd.merge(ftrain, d62, on=['item_price_level'], how="left")

    ftrain['item_price_level_count'].replace(np.nan, 0, inplace=True)
    ftrain['item_price_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftrain['item_price_level_count_istrade_rate'] = ftrain.item_price_level_count_istrade.astype('float') / ftrain.item_price_level_count.astype('float')
    #  贝叶斯平滑
    ftrain['item_price_level_count_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['item_price_level_count_istrade'],ftrain['item_price_level_count'], 'train11'+flag)
    del ftrain['item_price_level_count_istrade_rate']


    # # shop_star_level
    # d63 = train_feature_range[['shop_star_level', 'item_sales_level']]
    # d63 = d63.groupby(['shop_star_level'], as_index=False)['item_sales_level'].agg({'shop_star_level_count': 'count'})
    # ftrain = pd.merge(ftrain, d63, on=['shop_star_level'], how="left")
    #
    # d64 = train_feature_range[train_feature_range.is_trade == 1][['shop_star_level', 'item_sales_level']]
    # d64 = d64.groupby(['shop_star_level'], as_index=False)['item_sales_level'].agg({'shop_star_level_count_istrade': 'count'})
    # ftrain = pd.merge(ftrain, d64, on=['shop_star_level'], how="left")
    #
    # ftrain['shop_star_level_count'].replace(np.nan, 0, inplace=True)
    # ftrain['shop_star_level_count_istrade'].replace(np.nan, 0, inplace=True)
    # ftrain['shop_star_level_count_istrade_rate'] = ftrain.shop_star_level_count_istrade.astype('float') / ftrain.shop_star_level_count.astype('float')
    # #  贝叶斯平滑
    # ftrain['shop_star_level_count_istrade_rate_smooth'] = Bayesian_smoooth(ftrain['shop_star_level_count_istrade'],ftrain['shop_star_level_count'],'train12'+flag)
    # del ftrain['shop_star_level_count_istrade_rate']






    print("训练集特征区间结束!")
    """
    标签区间提取特征
    """
    ftrain = get_train_label(ftrain, train_label_range)
    print("训练集标签区间结束!")
    return ftrain


def get_train_label(ftrain, train_label_range):

    # 商铺出现的次数
    d1_label = train_label_range[['shop_id', 'item_sales_level']]
    d1_label = d1_label.groupby(['shop_id']).agg('count').reset_index()
    d1_label.rename(columns={'item_sales_level': 'shop_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d1_label, on="shop_id", how="left")

    # 用户和商铺出现的次数
    d2_label = train_label_range[['shop_id', 'user_id', 'item_sales_level']]
    d2_label = d2_label.groupby(['shop_id', 'user_id']).agg('count').reset_index()
    d2_label.rename(columns={'item_sales_level': 'user_shop_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d2_label, on=["shop_id", 'user_id'], how="left")

    d2 = d2_label.groupby(['user_id'], as_index=False)['shop_id'].agg({'user_diff_shop_count': 'count'})
    ftrain = pd.merge(ftrain, d2, on=["user_id"], how="left")

    d3 = d2_label.groupby(['shop_id'], as_index=False)['user_id'].agg({'shop_diff_user_count': 'count'})
    ftrain = pd.merge(ftrain, d3, on=["shop_id"], how="left")


    # 用户出现次数
    d3_label = train_label_range[['user_id', 'item_sales_level']]
    d3_label = d3_label.groupby(['user_id']).agg('count').reset_index()
    d3_label.rename(columns={'item_sales_level': 'user_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d3_label, on="user_id", how="left")

    # 广告出现的次数
    d4_label = train_label_range[['item_id', 'item_sales_level']]
    d4_label = d4_label.groupby(['item_id']).agg('count').reset_index()
    d4_label.rename(columns={'item_sales_level': 'item_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d4_label, on="item_id", how="left")

    # 该商家的特定广告出现的次数
    d5_label = train_label_range[['item_id', 'shop_id', 'item_sales_level']]
    d5_label = d5_label.groupby(['item_id', 'shop_id']).agg('count').reset_index()
    d5_label.rename(columns={'item_sales_level': 'shop_item_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d5_label, on=["item_id", 'shop_id'], how="left")


    # 特定用户和该广告出现的次数
    d6_label = train_label_range[['user_id', 'item_id', 'item_sales_level']]
    d6_label = d6_label.groupby(['user_id', 'item_id']).agg('count').reset_index()
    d6_label.rename(columns={'item_sales_level': 'user_item_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d6_label, on=["user_id", 'item_id'], how="left")

    d6 = d6_label.groupby(['user_id'], as_index=False)['item_id'].agg({'user_diff_item_count': 'count'})
    ftrain = pd.merge(ftrain, d6, on=["user_id"], how="left")

    d7 = d6_label.groupby(['item_id'], as_index=False)['user_id'].agg({'item_diff_user_count': 'count'})
    ftrain = pd.merge(ftrain, d7, on=["item_id"], how="left")

    # 用户和该用户的职业出现的次数
    d7_label = train_label_range[['user_id', 'user_occupation_id', 'item_sales_level']]
    d7_label = d7_label.groupby(['user_id', 'user_occupation_id']).agg('count').reset_index()
    d7_label.rename(columns={'item_sales_level': 'user_and_user_occupation_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d7_label, on=["user_id", 'user_occupation_id'], how="left")

    # 用户和商铺,广告出现的次数
    d8_label = train_label_range[['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d8_label = d8_label.groupby(['shop_id', 'user_id', 'item_id']).agg('count').reset_index()
    d8_label.rename(columns={'item_sales_level': 'user_shop_item_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d8_label, on=["shop_id", 'user_id', 'item_id'], how="left")


    train_label_range['day'] = train_label_range['context_timestamp'].map(lambda x: int(x[6:8]))
    train_label_range['hour'] = train_label_range['context_timestamp'].map(lambda x: int(x[8:10]))


    # user在每一个小时(24个小时)的的count个数
    d10_label = train_label_range[['user_id', 'day', 'hour', 'item_sales_level']]
    d10_label = d10_label.groupby(['user_id', 'day', 'hour']).agg('count').reset_index()
    d10_label.rename(columns={'item_sales_level': 'user_hour_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d10_label, on=["user_id", 'day', 'hour'], how="left")

    # shop在每一个小时(24个小时)的的count个数
    d11_label = train_label_range[['shop_id', 'day', 'hour', 'item_sales_level']]
    d11_label = d11_label.groupby(['shop_id', 'day', 'hour']).agg('count').reset_index()
    d11_label.rename(columns={'item_sales_level': 'shop_hour_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d11_label, on=["shop_id", 'day', 'hour'], how="left")

    # item在每一个小时(24个小时)的的count个数
    d12_label = train_label_range[['item_id', 'day', 'hour', 'item_sales_level']]
    d12_label = d12_label.groupby(['item_id', 'day', 'hour']).agg('count').reset_index()
    d12_label.rename(columns={'item_sales_level': 'item_hour_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d12_label, on=["item_id", 'day', 'hour'], how="left")


    # user和item的context_stamp的排名(升序)
    ftrain['context_timestamp_user_item_rank_label'] = ftrain.groupby(['user_id','item_id'])['context_timestamp'].rank(
        ascending=True)
    ftrain.drop_duplicates(inplace=True)
    # user和item的context_stamp的排名(降序)
    ftrain['context_timestamp_user_item_rank_label_desc'] = ftrain.groupby(['user_id', 'item_id'])['context_timestamp'].rank(
        ascending=False)
    ftrain.drop_duplicates(inplace=True)


    # user和shop的context_stamp的排名(升序)
    ftrain['context_timestamp_user_shop_rank_label'] = ftrain.groupby(['user_id', 'shop_id'])['context_timestamp'].rank(ascending=True)
    ftrain.drop_duplicates(inplace=True)
    # user和shop的context_stamp的排名(降序)
    ftrain['context_timestamp_user_shop_rank_label_desc'] = ftrain.groupby(['user_id', 'shop_id'])['context_timestamp'].rank(ascending=False)
    ftrain.drop_duplicates(inplace=True)


    # 对同一个用户的时间进行排序(升序)
    ftrain['context_timestamp_rank_label'] = ftrain.groupby(['user_id'])['context_timestamp'].rank(ascending=True)
    ftrain.drop_duplicates(inplace=True)
    # 对同一个用户的时间进行排序(降序)
    ftrain['context_timestamp_rank_desc_label'] = ftrain.groupby(['user_id'])['context_timestamp'].rank(ascending=False)
    ftrain.drop_duplicates(inplace=True)


    # 对同一个shop的时间进行排序(升序)
    ftrain['context_timestamp_shop_rank_label'] = ftrain.groupby(['shop_id'])['context_timestamp'].rank(ascending=True)
    ftrain.drop_duplicates(inplace=True)
    # 对同一个shop的时间进行排序(降序)
    ftrain['context_timestamp_shop_rank_desc_label'] = ftrain.groupby(['shop_id'])['context_timestamp'].rank(ascending=False)
    ftrain.drop_duplicates(inplace=True)



    # item的context_stamp的排名(升序)
    ftrain['context_timestamp_item_rank_label'] = ftrain.groupby(['item_id'])['context_timestamp'].rank(ascending=True)
    ftrain.drop_duplicates(inplace=True)
    # item的context_stamp的排名(降序)
    ftrain['context_timestamp_item_rank_label_desc'] = ftrain.groupby(['item_id'])['context_timestamp'].rank(ascending=False)
    ftrain.drop_duplicates(inplace=True)

    ftrain['aa_user_item_shop_rank_label'] = ftrain.groupby(['user_id', 'item_id', 'shop_id'])['context_timestamp'].rank(ascending=True)
    ftrain.drop_duplicates(inplace=True)

    ftrain['aa_user_item_shop_rank_desc_label'] = ftrain.groupby(['user_id', 'item_id', 'shop_id'])['context_timestamp'].rank(ascending=False)
    ftrain.drop_duplicates(inplace=True)

    # 广告商品页面展示标号
    d14_label = train_label_range[['context_page_id', 'item_id', 'item_price_level']]
    d14_label = d14_label.groupby(['context_page_id', 'item_id']).agg('count').reset_index()
    d14_label.rename(columns={'item_price_level': 'context_page_item_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d14_label, on=["context_page_id", 'item_id'], how="left")

    # user和shop店铺评价的均值
    d15_label = train_label_range[['user_id', 'shop_id', 'shop_review_num_level']]
    d15_label = d15_label.groupby(['user_id', 'shop_id']).agg('mean').reset_index()
    d15_label.rename(columns={'shop_review_num_level': 'user_shop_mean_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d15_label, on=["user_id", 'shop_id'], how="left")


    # user和item在每一个小时(24个小时)的的count个数
    d17_label = train_label_range[['user_id', 'item_id','hour','day','item_sales_level']]
    d17_label = d17_label.groupby(['user_id', 'item_id','hour','day']).agg('count').reset_index()
    d17_label.rename(columns={'item_sales_level': 'user_item_hour_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d17_label, on=['user_id', "item_id",'hour','day'], how="left")

    # user和shop在每一个小时(24个小时)的的count个数
    d18_label = train_label_range[['user_id', 'shop_id', 'hour', 'day', 'item_sales_level']]
    d18_label = d18_label.groupby(['user_id', 'shop_id', 'hour', 'day']).agg('count').reset_index()
    d18_label.rename(columns={'item_sales_level': 'user_shop_hour_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d18_label, on=['user_id', "shop_id", 'hour', 'day'], how="left")


    # user_id,item_city_id的count
    d20_label = train_label_range[['user_id', 'item_city_id', 'item_sales_level']]
    d20_label = d20_label.groupby(['user_id', 'item_city_id']).agg('count').reset_index()
    d20_label.rename(columns={'item_sales_level': 'user_item_city_id_count_label'}, inplace=True)
    ftrain = pd.merge(ftrain, d20_label, on=["item_city_id", 'user_id'], how="left")


    # 用户上下一次点击广告的时间间隔
    d24_label = train_label_range[['item_id', 'user_id','context_timestamp']]
    d24_label = d24_label.groupby(['user_id', 'item_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index()
    d24_label.rename(columns={'context_timestamp': 'dates'}, inplace=True)
    ftrain = pd.merge(ftrain, d24_label, on=["user_id", "item_id"], how="left")
    ftrain['context_timestamp_and_dates'] = ftrain.context_timestamp.astype('str') + '-' + ftrain.dates
    #
    ftrain['user_before_day_click_item_gap'] = ftrain.context_timestamp_and_dates.apply(get_day_gap_before)
    ftrain['user_after_day_click_item_gap'] = ftrain.context_timestamp_and_dates.apply(get_day_gap_after)


    # 用户是否是第一次点击特定广告
    ftrain["is_first_get_coupon"] = ftrain.context_timestamp_and_dates.apply(is_first_get_coupon)
    # 用户是否是最后一次点击特定广告
    ftrain["is_last_get_coupon"] = ftrain.context_timestamp_and_dates.apply(is_last_get_coupon)
    #
    d25_label = train_label_range.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count_label': 'count'})
    ftrain = pd.merge(ftrain, d25_label, on=["user_id", 'category'], how="left")

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

    return ftrain


def extract_validate_feature(validate_feature_range,validate_label_range):

    """
        特征区间提取特征
    """
    # 商铺出现的次数
    d1 = validate_feature_range[['shop_id', 'item_sales_level']]
    d1 = d1.groupby(['shop_id']).agg('count').reset_index()
    d1.rename(columns={'item_sales_level': 'shop_count'}, inplace=True)
    fvalidate = pd.merge(validate_label_range, d1, on="shop_id", how="left")

    # 用户和商铺出现的次数
    d2 = validate_feature_range[['shop_id', 'user_id', 'item_sales_level']]
    d2 = d2.groupby(['shop_id', 'user_id']).agg('count').reset_index()
    d2.rename(columns={'item_sales_level': 'user_shop_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d2, on=["shop_id", 'user_id'], how="left")

    # 用户出现次数(总的次数 )
    d3 = validate_feature_range[['user_id', 'item_sales_level']]
    d3 = d3.groupby(['user_id']).agg('count').reset_index()
    d3.rename(columns={'item_sales_level': 'user_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d3, on="user_id", how="left")

    # 用户出现次数(交易的次数)
    d4 = validate_feature_range[validate_feature_range['is_trade'] == 1][['user_id', 'item_sales_level']]
    d4 = d4.groupby(['user_id']).agg('count').reset_index()
    d4.rename(columns={'item_sales_level': 'user_count_istrade'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d4, on="user_id", how="left")

    # 空值替换
    fvalidate['user_count_istrade'].replace(np.nan,0, inplace=True)
    fvalidate['user_count'].replace(np.nan,0, inplace=True)


    fvalidate['user_istrade_rate'] = fvalidate['user_count_istrade'].astype('float') / fvalidate['user_count'].astype('float')

    # # 贝叶斯平滑
    fvalidate['user_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['user_count_istrade'],fvalidate['user_count'],'validate1')
    del fvalidate['user_istrade_rate']

    # 广告出现的次数
    d5 = validate_feature_range[['item_id', 'item_sales_level']]
    d5 = d5.groupby(['item_id']).agg('count').reset_index()
    d5.rename(columns={'item_sales_level': 'item_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d5, on="item_id", how="left")

    # 该商家的特定广告出现的次数
    d6 = validate_feature_range[['item_id', 'shop_id', 'item_sales_level']]
    d6 = d6.groupby(['item_id', 'shop_id']).agg('count').reset_index()
    d6.rename(columns={'item_sales_level': 'shop_item_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d6, on=["item_id", 'shop_id'], how="left")

    # 特定用户和该广告出现的次数
    d7 = validate_feature_range[['user_id', 'item_id', 'item_sales_level']]
    d7 = d7.groupby(['user_id', 'item_id']).agg('count').reset_index()
    d7.rename(columns={'item_sales_level': 'user_item_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d7, on=["user_id", 'item_id'], how="left")

    # 特定用户和该广告出现的次数(交易成功)
    d8 = validate_feature_range[validate_feature_range.is_trade == 1][['user_id', 'item_id', 'item_sales_level']]
    d8 = d8.groupby(['user_id', 'item_id']).agg('count').reset_index()
    d8.rename(columns={'item_sales_level': 'user_item_count_istrade'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d8, on=["user_id", 'item_id'], how="left")

    # 空值替换
    fvalidate['user_item_count'].replace(np.nan, 0, inplace=True)
    fvalidate['user_item_count_istrade'].replace(np.nan,0, inplace=True)

    # 交易率
    fvalidate['user_item_count_istrade_rate'] = fvalidate['user_item_count_istrade'].astype('float') / fvalidate['user_item_count'].astype('float')

    # # 贝叶斯平滑
    fvalidate['user_item_count_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['user_item_count_istrade'], fvalidate['user_item_count'],'validate2')
    del fvalidate['user_item_count_istrade_rate']

    # 用户和该用户的职业出现的次数
    d9 = validate_feature_range[['user_id', 'user_occupation_id', 'item_sales_level']]
    d9 = d9.groupby(['user_id', 'user_occupation_id']).agg('count').reset_index()
    d9.rename(columns={'item_sales_level': 'user_and_user_occupation_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d9, on=["user_id", 'user_occupation_id'], how="left")

    # 该职业出现的次数
    d13 = validate_feature_range[['user_occupation_id', 'item_price_level']]
    d13 = d13.groupby(['user_occupation_id']).agg('count').reset_index()
    d13.rename(columns={'item_price_level': 'user_occupation_id_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d13, on="user_occupation_id", how="left")

    # 广告价格等级出现的次数
    d14 = validate_feature_range[['item_price_level', 'user_id']]
    d14 = d14.groupby(['item_price_level']).agg('count').reset_index()
    d14.rename(columns={'user_id': 'item_price_level_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d14, on="item_price_level", how="left")

    # 性别对于是否交易的影响
    d15 = validate_feature_range[['user_gender_id', 'user_id']]
    d15 = d15.groupby(['user_gender_id']).agg('count').reset_index()
    d15.rename(columns={'user_id': 'user_gender_id_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d15, on="user_gender_id", how="left")

    # 看当前日期是否是高峰期
    fvalidate['is_high_trade'] = fvalidate['context_timestamp'].apply(date2morning_afternoon)

    # 收藏级别出现的次数
    d17 = validate_feature_range[['item_collected_level', 'item_price_level']]
    d17 = d17.groupby(['item_collected_level']).agg('count').reset_index()
    d17.rename(columns={'item_price_level': 'item_collected_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d17, on="item_collected_level", how="left")

    # 广告商品页面展示标号
    d18 = validate_feature_range[['context_page_id', 'item_id', 'item_price_level']]
    d18 = d18.groupby(['context_page_id', 'item_id']).agg('count').reset_index()
    d18.rename(columns={'item_price_level': 'context_page_item_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d18, on=["context_page_id", 'item_id'], how="left")


    # 用户和商铺,广告出现的次数
    d20 = validate_feature_range[['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d20 = d20.groupby(['shop_id', 'user_id', 'item_id']).agg('count').reset_index()
    d20.rename(columns={'item_sales_level': 'user_shop_item_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d20, on=["shop_id", 'user_id', 'item_id'], how="left")


    validate_feature_range['day'] = validate_feature_range['context_timestamp'].map(lambda x: int(x[6:8]))
    validate_feature_range['hour'] = validate_feature_range['context_timestamp'].map(lambda x: int(x[8:10]))

    fvalidate['day'] = fvalidate['context_timestamp'].map(lambda x: int(x[6:8]))
    fvalidate['hour'] = fvalidate['context_timestamp'].map(lambda x: int(x[8:10]))

    # user和shop在每一个小时(24个小时)的的count个数
    d21 = validate_feature_range[['user_id', 'shop_id', 'hour', 'item_sales_level']]
    d21 = d21.groupby(['user_id', 'shop_id', 'hour']).agg('count').reset_index()
    d21.rename(columns={'item_sales_level': 'user_shop_hour_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d21, on=['user_id', "shop_id", 'hour'], how="left")

    # user在每一个小时(24个小时)的的count个数
    d22 = validate_feature_range[['user_id','hour', 'item_sales_level']]
    d22 = d22.groupby(['user_id','hour']).agg('count').reset_index()
    d22.rename(columns={'item_sales_level': 'user_hour_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d22, on=["user_id",'hour'], how="left")

    # shop在每一个小时(24个小时)的的count个数
    d23 = validate_feature_range[['shop_id','hour', 'item_sales_level']]
    d23 = d23.groupby(['shop_id','hour']).agg('count').reset_index()
    d23.rename(columns={'item_sales_level': 'shop_hour_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d23, on=["shop_id",'hour'], how="left")

    # item在每一个小时(24个小时)的的count个数
    d24 = validate_feature_range[['item_id','hour', 'item_sales_level']]
    d24 = d24.groupby(['item_id','hour']).agg('count').reset_index()
    d24.rename(columns={'item_sales_level': 'item_hour_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d24, on=["item_id",'hour'], how="left")

    # user和item在每一个小时(24个小时)的的count个数
    d29 = validate_feature_range[['user_id', 'item_id','hour', 'item_sales_level']]
    d29 = d29.groupby(['user_id', 'item_id','hour']).agg('count').reset_index()
    d29.rename(columns={'item_sales_level': 'user_item_hour_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d29, on=['user_id', "item_id",'hour'], how="left")

    # 用户和广告出现的次数
    d30 = validate_feature_range[validate_feature_range.is_trade == 1][['item_id', 'item_sales_level']]
    d30 = d30.groupby(['item_id']).agg('count').reset_index()
    d30.rename(columns={'item_sales_level': 'item_count_istrade'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d30, on=["item_id"], how="left")

    # 空值替换
    fvalidate['item_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['item_count'].replace(np.nan, 0, inplace=True)

    fvalidate['item_istrade_rate'] = fvalidate.item_count_istrade.astype('float') / fvalidate.item_count.astype('float')

    # # 贝叶斯平滑
    fvalidate['item_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['item_count_istrade'],fvalidate['item_count'], 'validate3')
    del fvalidate['item_istrade_rate']

    # shop交易count
    d33 = validate_feature_range[validate_feature_range.is_trade == 1][['shop_id', 'item_sales_level']]
    d33 = d33.groupby(['shop_id']).agg('count').reset_index()
    d33.rename(columns={'item_sales_level': 'shop_count_istrade'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d33, on=["shop_id"], how="left")

    fvalidate['shop_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['shop_count'].replace(np.nan, 0, inplace=True)

    fvalidate['shop_istrade_rate'] = fvalidate.shop_count_istrade.astype('float') / fvalidate.shop_count.astype('float')

    # # 贝叶斯平滑
    fvalidate['shop_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['shop_count_istrade'], fvalidate['shop_count'],'validate4')
    del fvalidate['shop_istrade_rate']

    # item_city_id的count
    d34 = validate_feature_range[['item_city_id', 'item_sales_level']]
    d34 = d34.groupby(['item_city_id']).agg('count').reset_index()
    d34.rename(columns={'item_sales_level': 'item_city_id_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d34, on="item_city_id", how="left")

    # user_id,item_city_id的count
    d35 = validate_feature_range[['user_id', 'item_city_id', 'item_sales_level']]
    d35 = d35.groupby(['user_id', 'item_city_id']).agg('count').reset_index()
    d35.rename(columns={'item_sales_level': 'user_item_city_id_count'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d35, on=["item_city_id", 'user_id'], how="left")

    # user和shop的次数
    d36 = validate_feature_range[validate_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_sales_level']]
    d36 = d36.groupby(['shop_id', 'user_id']).agg('count').reset_index()
    d36.rename(columns={'item_sales_level': 'user_shop_count_istrade'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d36, on=["shop_id", 'user_id'], how="left")

    fvalidate['user_shop_count'].replace(np.nan, 0, inplace=True)
    fvalidate['user_shop_count_istrade'].replace(np.nan, 0, inplace=True)

    # fvalidate['user_shop_count_minus_user_shop_istrade'] = fvalidate.user_shop_count.astype('int') - fvalidate.user_shop_count_istrade.astype('int')
    fvalidate['user_shop_count_istrade_rate'] = fvalidate.user_shop_count_istrade.astype('float') / fvalidate.user_shop_count.astype('float')
    #
    # # 贝叶斯平滑
    fvalidate['user_shop_count_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['user_shop_count_istrade'], fvalidate['user_shop_count'],'validate5')
    del fvalidate['user_shop_count_istrade_rate']


    # 用户和广告出现的次数item_count  # 空值替换
    fvalidate['item_count'].replace(np.nan,0, inplace=True)
    fvalidate['item_count_istrade'].replace(np.nan,0, inplace=True)

    # # 用户和广告出现的次数item_count
    # fvalidate['item_count_minus_item_istrade'] = fvalidate.item_count.astype('int') - fvalidate.item_count_istrade.astype('int')
    # fvalidate['user_item_count_minus_user_item_istrade'] = fvalidate['user_item_count'].astype('int') - fvalidate['user_item_count_istrade'].astype('int')
    # fvalidate['user_count_minus_user_count_istrade'] = fvalidate['user_count'].astype('int') - fvalidate['user_count_istrade'].astype('int')
    #
    # user和item_brand_id的次数
    d37 = validate_feature_range.groupby(['user_id', 'item_brand_id'], as_index=False)['item_sales_level'].agg({'user_item_brand_count': 'count'})
    fvalidate = pd.merge(fvalidate, d37, on=["item_brand_id", 'user_id'], how="left")

    # user with shop_star_level count
    d38 = validate_feature_range.groupby(['user_id', 'shop_star_level'], as_index=False)['item_sales_level'].agg({'user_shop_star_level_count': 'count'})
    fvalidate = pd.merge(fvalidate, d38, on=["shop_star_level", 'user_id'], how="left")

    # user with item_sales_level count
    d39 = validate_feature_range.groupby(['user_id', 'item_sales_level'], as_index=False)['item_sales_level'].agg({'user_item_sales_level_count': 'count'})
    fvalidate = pd.merge(fvalidate, d39, on=["item_sales_level", 'user_id'], how="left")

    # user with item_collected_level count
    d40 = validate_feature_range.groupby(['user_id', 'item_collected_level'], as_index=False)['item_sales_level'].agg({'user_item_collected_level_count': 'count'})
    fvalidate = pd.merge(fvalidate, d40, on=["item_collected_level", 'user_id'], how="left")
    #
    # user with category count
    d41 = validate_feature_range.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count': 'count'})
    fvalidate = pd.merge(fvalidate, d41, on=["category", 'user_id'], how="left")
    #
    d44 = validate_feature_range.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count': 'count'})
    fvalidate = pd.merge(fvalidate, d44, on=["item_city_id", 'user_id', 'item_id'], how="left")


    ######  new
    d45 = validate_feature_range[validate_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d45 = d45.groupby(['shop_id', 'user_id', 'item_id'], as_index=False)['item_sales_level'].agg({'user_shop_item_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d45, on=['shop_id', 'user_id', 'item_id'], how="left")

    fvalidate['user_shop_item_count'].replace(np.nan, 0, inplace=True)
    fvalidate['user_shop_item_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['user_shop_item_count_istrade_rate'] = fvalidate.user_shop_item_count_istrade.astype('float') / fvalidate.user_shop_item_count.astype('float')
    # # 贝叶斯平滑
    fvalidate['user_shop_item_count_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['user_shop_item_count_istrade'],fvalidate['user_shop_item_count'],'validate6')
    del fvalidate['user_shop_item_count_istrade_rate']

    ####
    d46 = validate_feature_range[validate_feature_range.is_trade == 1][['user_id', 'item_id', 'item_city_id', 'item_sales_level']]
    d46 = d46.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d46, on=['user_id', 'item_id', 'item_city_id'], how="left")

    fvalidate['user_item_item_city_id_count'].replace(np.nan, 0, inplace=True)
    fvalidate['user_item_item_city_id_count_istrade'].replace(np.nan, 0, inplace=True)

    fvalidate['user_item_item_city_id_count_istrade_rate'] = fvalidate.user_item_item_city_id_count_istrade.astype('float') / fvalidate.user_item_item_city_id_count.astype('float')
    #  贝叶斯平滑
    fvalidate['user_item_item_city_id_count_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['user_item_item_city_id_count_istrade'], fvalidate['user_item_item_city_id_count'],
                                                                                     'validate7')
    del fvalidate['user_item_item_city_id_count_istrade_rate']

    ### user category
    d47 = validate_feature_range[validate_feature_range.is_trade == 1][['user_id', 'category', 'item_sales_level']]
    d47 = d47.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d47, on=['user_id', 'category'], how="left")

    fvalidate['user_category_count'].replace(np.nan, 0, inplace=True)
    fvalidate['user_category_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['user_category_count_istrade_rate'] = fvalidate.user_category_count_istrade.astype('float') / fvalidate.user_category_count.astype('float')
    #  贝叶斯平滑
    fvalidate['user_category_count_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['user_category_count_istrade'],fvalidate['user_category_count'], 'validate8')
    del fvalidate['user_category_count_istrade_rate']

    # user_gender_id
    d48 = validate_feature_range[validate_feature_range.is_trade == 1][['user_gender_id', 'item_sales_level']]
    d48 = d48.groupby(['user_gender_id'], as_index=False)['item_sales_level'].agg({'user_gender_id_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d48, on=['user_gender_id'], how="left")

    fvalidate['user_gender_id_count'].replace(np.nan, 0, inplace=True)
    fvalidate['user_gender_id_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['user_gender_id_count_istrade_rate'] = fvalidate.user_gender_id_count_istrade.astype('float') / fvalidate.user_gender_id_count.astype('float')

    # user_occupation_id
    d49 = validate_feature_range[validate_feature_range.is_trade == 1][['user_occupation_id', 'item_sales_level']]
    d49 = d49.groupby(['user_occupation_id'], as_index=False)['item_sales_level'].agg({'user_occupation_id_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d49, on=['user_occupation_id'], how="left")

    fvalidate['user_occupation_id_count'].replace(np.nan, 0, inplace=True)
    fvalidate['user_occupation_id_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['user_occupation_id_count_istrade_rate'] = fvalidate.user_occupation_id_count_istrade.astype('float') / fvalidate.user_occupation_id_count.astype('float')

    # user_age_level
    d50 = validate_feature_range[['user_age_level', 'item_sales_level']]
    d50 = d50.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count': 'count'})
    fvalidate = pd.merge(fvalidate, d50, on=['user_age_level'], how="left")

    d51 = validate_feature_range[validate_feature_range.is_trade == 1][['user_age_level', 'item_sales_level']]
    d51 = d51.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d51, on=['user_age_level'], how="left")

    fvalidate['user_age_level_count'].replace(np.nan, 0, inplace=True)
    fvalidate['user_age_level_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['user_age_level_count_istrade_rate'] = fvalidate.user_age_level_count_istrade.astype('float') / fvalidate.user_age_level_count.astype('float')

    # item_sales_level
    d52 = validate_feature_range[['item_sales_level', 'context_timestamp']]
    d52 = d52.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count': 'count'})
    fvalidate = pd.merge(fvalidate, d52, on=['item_sales_level'], how="left")

    d53 = validate_feature_range[validate_feature_range.is_trade == 1][['item_sales_level', 'context_timestamp']]
    d53 = d53.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d53, on=['item_sales_level'], how="left")

    fvalidate['item_sales_level_count'].replace(np.nan, 0, inplace=True)
    fvalidate['item_sales_level_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['item_sales_level_count_istrade_rate'] = fvalidate.item_sales_level_count_istrade.astype('float') / fvalidate.item_sales_level_count.astype('float')

    # item_collected_level
    d54 = validate_feature_range[validate_feature_range.is_trade == 1][['item_collected_level', 'item_sales_level']]
    d54 = d54.groupby(['item_collected_level'], as_index=False)['item_sales_level'].agg({'item_collected_level_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d54, on=['item_collected_level'], how="left")

    fvalidate['item_collected_count'].replace(np.nan, 0, inplace=True)
    fvalidate['item_collected_level_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['item_collected_level_count_istrade_rate'] = fvalidate.item_collected_level_count_istrade.astype('float') / fvalidate.item_collected_count.astype('float')

    # user_star_level
    d55 = validate_feature_range[['user_star_level', 'item_sales_level']]
    d55 = d55.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count': 'count'})
    fvalidate = pd.merge(fvalidate, d55, on=['user_star_level'], how="left")

    d56 = validate_feature_range[validate_feature_range.is_trade == 1][['user_star_level', 'item_sales_level']]
    d56 = d56.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d56, on=['user_star_level'], how="left")

    fvalidate['user_star_level_count'].replace(np.nan, 0, inplace=True)
    fvalidate['user_star_level_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['user_star_level_count_istrade_rate'] = fvalidate.user_star_level_count_istrade.astype('float') / fvalidate.user_star_level_count.astype('float')

    # item_pv_level
    d57 = validate_feature_range[['item_pv_level', 'item_sales_level']]
    d57 = d57.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count': 'count'})
    fvalidate = pd.merge(fvalidate, d57, on=['item_pv_level'], how="left")

    d58 = validate_feature_range[validate_feature_range.is_trade == 1][['item_pv_level', 'item_sales_level']]
    d58 = d58.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d58, on=['item_pv_level'], how="left")

    fvalidate['item_pv_level_count'].replace(np.nan, 0, inplace=True)
    fvalidate['item_pv_level_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['item_pv_level_count_istrade_rate'] = fvalidate.item_pv_level_count_istrade.astype('float') / fvalidate.item_pv_level_count.astype('float')

    #  贝叶斯平滑
    fvalidate['item_pv_level_count_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['item_pv_level_count_istrade'],fvalidate['item_pv_level_count'], 'validate9')
    del fvalidate['item_pv_level_count_istrade_rate']

    # shop_review_num_level
    d59 = validate_feature_range[['shop_review_num_level', 'item_sales_level']]
    d59 = d59.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count': 'count'})
    fvalidate = pd.merge(fvalidate, d59, on=['shop_review_num_level'], how="left")

    d60 = validate_feature_range[validate_feature_range.is_trade == 1][['shop_review_num_level', 'item_sales_level']]
    d60 = d60.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d60, on=['shop_review_num_level'], how="left")

    fvalidate['shop_review_num_level_count'].replace(np.nan, 0, inplace=True)
    fvalidate['shop_review_num_level_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['shop_review_num_level_count_istrade_rate'] = fvalidate.shop_review_num_level_count_istrade.astype('float') / fvalidate.shop_review_num_level_count.astype('float')

    #  贝叶斯平滑
    fvalidate['shop_review_num_level_count_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['shop_review_num_level_count_istrade'],fvalidate['shop_review_num_level_count'],'validate10')
    del fvalidate['shop_review_num_level_count_istrade_rate']

    # item_price_level
    d62 = validate_feature_range[validate_feature_range.is_trade == 1][['item_price_level', 'item_sales_level']]
    d62 = d62.groupby(['item_price_level'], as_index=False)['item_sales_level'].agg({'item_price_level_count_istrade': 'count'})
    fvalidate = pd.merge(fvalidate, d62, on=['item_price_level'], how="left")

    fvalidate['item_price_level_count'].replace(np.nan, 0, inplace=True)
    fvalidate['item_price_level_count_istrade'].replace(np.nan, 0, inplace=True)
    fvalidate['item_price_level_count_istrade_rate'] = fvalidate.item_price_level_count_istrade.astype('float') / fvalidate.item_price_level_count.astype('float')
    #  贝叶斯平滑
    fvalidate['item_price_level_count_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['item_price_level_count_istrade'],fvalidate['item_price_level_count'],'validate11')
    del fvalidate['item_price_level_count_istrade_rate']

    # # shop_star_level
    # d63 = validate_feature_range[['shop_star_level', 'item_sales_level']]
    # d63 = d63.groupby(['shop_star_level'], as_index=False)['item_sales_level'].agg({'shop_star_level_count': 'count'})
    # fvalidate = pd.merge(fvalidate, d63, on=['shop_star_level'], how="left")
    #
    # d64 = validate_feature_range[validate_feature_range.is_trade == 1][['shop_star_level', 'item_sales_level']]
    # d64 = d64.groupby(['shop_star_level'], as_index=False)['item_sales_level'].agg({'shop_star_level_count_istrade': 'count'})
    # fvalidate = pd.merge(fvalidate, d64, on=['shop_star_level'], how="left")
    #
    # fvalidate['shop_star_level_count'].replace(np.nan, 0, inplace=True)
    # fvalidate['shop_star_level_count_istrade'].replace(np.nan, 0, inplace=True)
    # fvalidate['shop_star_level_count_istrade_rate'] = fvalidate.shop_star_level_count_istrade.astype('float') / fvalidate.shop_star_level_count.astype('float')
    # #  贝叶斯平滑
    # fvalidate['shop_star_level_count_istrade_rate_smooth'] = Bayesian_smoooth(fvalidate['shop_star_level_count_istrade'],fvalidate['shop_star_level_count'],'validate12')
    # del fvalidate['shop_star_level_count_istrade_rate']


    print("验证集特征区间结束!")
    """
        标签区间提取特征
    """
    fvalidate = get_validate_label(fvalidate, validate_label_range)
    print("验证集标签区间结束!")
    return fvalidate



def get_validate_label(fvalidate, validate_label_range):
    # 商铺出现的次数
    d1_label = validate_label_range[['shop_id', 'item_sales_level']]
    d1_label = d1_label.groupby(['shop_id']).agg('count').reset_index()
    d1_label.rename(columns={'item_sales_level': 'shop_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d1_label, on="shop_id", how="left")

    # 用户和商铺出现的次数
    d2_label = validate_label_range[['shop_id', 'user_id', 'item_sales_level']]
    d2_label = d2_label.groupby(['shop_id', 'user_id']).agg('count').reset_index()
    d2_label.rename(columns={'item_sales_level': 'user_shop_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d2_label, on=["shop_id", 'user_id'], how="left")

    d2 = d2_label.groupby(['user_id'], as_index=False)['shop_id'].agg({'user_diff_shop_count': 'count'})
    fvalidate = pd.merge(fvalidate, d2, on=["user_id"], how="left")

    d3 = d2_label.groupby(['shop_id'], as_index=False)['user_id'].agg({'shop_diff_user_count': 'count'})
    fvalidate = pd.merge(fvalidate, d3, on=["shop_id"], how="left")

    # 用户出现次数
    d3_label = validate_label_range[['user_id', 'item_sales_level']]
    d3_label = d3_label.groupby(['user_id']).agg('count').reset_index()
    d3_label.rename(columns={'item_sales_level': 'user_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d3_label, on="user_id", how="left")

    # 广告出现的次数
    d4_label = validate_label_range[['item_id', 'item_sales_level']]
    d4_label = d4_label.groupby(['item_id']).agg('count').reset_index()
    d4_label.rename(columns={'item_sales_level': 'item_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d4_label, on="item_id", how="left")

    # 该商家的特定广告出现的次数
    d5_label = validate_label_range[['item_id', 'shop_id', 'item_sales_level']]
    d5_label = d5_label.groupby(['item_id', 'shop_id']).agg('count').reset_index()
    d5_label.rename(columns={'item_sales_level': 'shop_item_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d5_label, on=["item_id", 'shop_id'], how="left")

    # 特定用户和该广告出现的次数
    d6_label = validate_label_range[['user_id', 'item_id', 'item_sales_level']]
    d6_label = d6_label.groupby(['user_id', 'item_id']).agg('count').reset_index()
    d6_label.rename(columns={'item_sales_level': 'user_item_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d6_label, on=["user_id", 'item_id'], how="left")

    d6 = d6_label.groupby(['user_id'], as_index=False)['item_id'].agg({'user_diff_item_count': 'count'})
    fvalidate = pd.merge(fvalidate, d6, on=["user_id"], how="left")

    d7 = d6_label.groupby(['item_id'], as_index=False)['user_id'].agg({'item_diff_user_count': 'count'})
    fvalidate = pd.merge(fvalidate, d7, on=["item_id"], how="left")


    # 用户和该用户的职业出现的次数
    d7_label = validate_label_range[['user_id', 'user_occupation_id', 'item_sales_level']]
    d7_label = d7_label.groupby(['user_id', 'user_occupation_id']).agg('count').reset_index()
    d7_label.rename(columns={'item_sales_level': 'user_and_user_occupation_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d7_label, on=["user_id", 'user_occupation_id'], how="left")

    # 用户和商铺,广告出现的次数
    d8_label = validate_label_range[['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d8_label = d8_label.groupby(['shop_id', 'user_id', 'item_id']).agg('count').reset_index()
    d8_label.rename(columns={'item_sales_level': 'user_shop_item_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d8_label, on=["shop_id", 'user_id', 'item_id'], how="left")


    validate_label_range['day'] = validate_label_range['context_timestamp'].map(lambda x: int(x[6:8]))
    validate_label_range['hour'] = validate_label_range['context_timestamp'].map(lambda x: int(x[8:10]))

    # user在每一个小时(24个小时)的的count个数
    d10_label = validate_label_range[['user_id', 'day', 'hour', 'item_sales_level']]
    d10_label = d10_label.groupby(['user_id', 'day', 'hour']).agg('count').reset_index()
    d10_label.rename(columns={'item_sales_level': 'user_hour_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d10_label, on=["user_id", 'day', 'hour'], how="left")

    # shop在每一个小时(24个小时)的的count个数
    d11_label = validate_label_range[['shop_id', 'day', 'hour', 'item_sales_level']]
    d11_label = d11_label.groupby(['shop_id', 'day', 'hour']).agg('count').reset_index()
    d11_label.rename(columns={'item_sales_level': 'shop_hour_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d11_label, on=["shop_id", 'day', 'hour'], how="left")

    # item在每一个小时(24个小时)的的count个数
    d12_label = validate_label_range[['item_id', 'day', 'hour', 'item_sales_level']]
    d12_label = d12_label.groupby(['item_id', 'day', 'hour']).agg('count').reset_index()
    d12_label.rename(columns={'item_sales_level': 'item_hour_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d12_label, on=["item_id", 'day', 'hour'], how="left")


    # user和item的context_stamp的排名(升序)
    fvalidate['context_timestamp_user_item_rank_label'] = fvalidate.groupby(['user_id', 'item_id'])['context_timestamp'].rank(
        ascending=True)
    fvalidate.drop_duplicates(inplace=True)
    # user和item的context_stamp的排名(降序)
    fvalidate['context_timestamp_user_item_rank_label_desc'] = fvalidate.groupby(['user_id', 'item_id'])['context_timestamp'].rank(
        ascending=False)
    fvalidate.drop_duplicates(inplace=True)



    # user和shop的context_stamp的排名(升序)
    fvalidate['context_timestamp_user_shop_rank_label'] = fvalidate.groupby(['user_id', 'shop_id'])['context_timestamp'].rank(
        ascending=True)
    fvalidate.drop_duplicates(inplace=True)
    # user和shop的context_stamp的排名(降序)
    fvalidate['context_timestamp_user_shop_rank_label_desc'] = fvalidate.groupby(['user_id', 'shop_id'])['context_timestamp'].rank(
        ascending=False)
    fvalidate.drop_duplicates(inplace=True)



    # 对同一个用户的时间进行排序(升序)
    fvalidate['context_timestamp_rank_label'] = fvalidate.groupby(['user_id'])['context_timestamp'].rank(ascending=True)
    fvalidate.drop_duplicates(inplace=True)
    # 对同一个用户的时间进行排序(降序)
    fvalidate['context_timestamp_rank_desc_label'] = fvalidate.groupby(['user_id'])['context_timestamp'].rank(ascending=False)
    fvalidate.drop_duplicates(inplace=True)


    # 对同一个shop的时间进行排序(升序)
    fvalidate['context_timestamp_shop_rank_label'] = fvalidate.groupby(['shop_id'])['context_timestamp'].rank(ascending=True)
    fvalidate.drop_duplicates(inplace=True)
    # 对同一个shop的时间进行排序(降序)
    fvalidate['context_timestamp_shop_rank_desc_label'] = fvalidate.groupby(['shop_id'])['context_timestamp'].rank(ascending=False)
    fvalidate.drop_duplicates(inplace=True)


    # item的context_stamp的排名(升序)
    fvalidate['context_timestamp_item_rank_label'] = fvalidate.groupby(['item_id'])['context_timestamp'].rank(ascending=True)
    fvalidate.drop_duplicates(inplace=True)
    # item的context_stamp的排名(降序)
    fvalidate['context_timestamp_item_rank_label_desc'] = fvalidate.groupby(['item_id'])['context_timestamp'].rank(ascending=False)
    fvalidate.drop_duplicates(inplace=True)
    #
    fvalidate['aa_user_item_shop_rank_label'] = fvalidate.groupby(['user_id', 'item_id', 'shop_id'])['context_timestamp'].rank(ascending=True)
    fvalidate.drop_duplicates(inplace=True)

    fvalidate['aa_user_item_shop_rank_desc_label'] = fvalidate.groupby(['user_id', 'item_id', 'shop_id'])['context_timestamp'].rank(ascending=False)
    fvalidate.drop_duplicates(inplace=True)

    # 广告商品页面展示标号
    d14_label = validate_label_range[['context_page_id', 'item_id', 'item_price_level']]
    d14_label = d14_label.groupby(['context_page_id', 'item_id']).agg('count').reset_index()
    d14_label.rename(columns={'item_price_level': 'context_page_item_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d14_label, on=["context_page_id", 'item_id'], how="left")

    # user和shop店铺评价的均值
    d15_label = validate_label_range[['user_id', 'shop_id', 'shop_review_num_level']]
    d15_label = d15_label.groupby(['user_id', 'shop_id']).agg('mean').reset_index()
    d15_label.rename(columns={'shop_review_num_level': 'user_shop_mean_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d15_label, on=["user_id", 'shop_id'], how="left")

    # user和item在每一个小时(24个小时)的的count个数
    d17_label = validate_label_range[['user_id', 'item_id', 'day', 'hour', 'item_sales_level']]
    d17_label = d17_label.groupby(['user_id', 'item_id', 'day', 'hour']).agg('count').reset_index()
    d17_label.rename(columns={'item_sales_level': 'user_item_hour_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d17_label, on=['user_id', "item_id", 'day', 'hour'], how="left")

    # user和shop在每一个小时(24个小时)的的count个数
    d18_label = validate_label_range[['user_id', 'shop_id', 'hour', 'day', 'item_sales_level']]
    d18_label = d18_label.groupby(['user_id', 'shop_id', 'hour', 'day']).agg('count').reset_index()
    d18_label.rename(columns={'item_sales_level': 'user_shop_hour_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d18_label, on=['user_id', "shop_id", 'hour', 'day'], how="left")

    # user_id,item_city_id的count
    d20_label = validate_label_range[['user_id', 'item_city_id', 'item_sales_level']]
    d20_label = d20_label.groupby(['user_id', 'item_city_id']).agg('count').reset_index()
    d20_label.rename(columns={'item_sales_level': 'user_item_city_id_count_label'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d20_label, on=["item_city_id", 'user_id'], how="left")

    # 用户上下一次点击广告的时间间隔
    d24_label = validate_label_range[['item_id', 'user_id', 'context_timestamp']]
    d24_label = d24_label.groupby(['user_id', 'item_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index()
    d24_label.rename(columns={'context_timestamp': 'dates'}, inplace=True)
    fvalidate = pd.merge(fvalidate, d24_label, on=["user_id", "item_id"], how="left")

    fvalidate['context_timestamp_and_dates'] = fvalidate.context_timestamp.astype('str') + '-' + fvalidate.dates
    fvalidate['user_before_day_click_item_gap'] = fvalidate.context_timestamp_and_dates.apply(get_day_gap_before)
    fvalidate['user_after_day_click_item_gap'] = fvalidate.context_timestamp_and_dates.apply(get_day_gap_after)

    # 用户是否是第一次点击特定广告
    fvalidate["is_first_get_coupon"] = fvalidate.context_timestamp_and_dates.apply(is_first_get_coupon)
    # 用户是否是最后一次点击特定广告
    fvalidate["is_last_get_coupon"] = fvalidate.context_timestamp_and_dates.apply(is_last_get_coupon)

    d25_label = validate_label_range.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count_label': 'count'})
    fvalidate = pd.merge(fvalidate, d25_label, on=["category", 'user_id'], how="left")

    # process property
    fvalidate['property_count'] = fvalidate['item_property_list'].apply(get_property_info)
    fvalidate['property'] = fvalidate['property_count'].apply(lambda x: x.split(';')[0])
    fvalidate['property_max_count'] = fvalidate['property_count'].apply(lambda x: x.split(';')[1])
    del fvalidate['property_count']

    # user with category rank
    fvalidate['user_and_category_rank'] = fvalidate.groupby(['user_id', 'category'])['context_timestamp'].rank(ascending=True)
    fvalidate['user_and_category_rank_desc'] = fvalidate.groupby(['user_id', 'category'])['context_timestamp'].rank(ascending=False)

    # gender_id with category
    test_range = validate_label_range[validate_label_range.user_gender_id != -1]
    d26_label = test_range.groupby(['user_gender_id', 'category'], as_index=False)['item_sales_level'].agg({'gender_category_count': 'count'})
    fvalidate = pd.merge(fvalidate, d26_label, on=["category", 'user_gender_id'], how="left")

    d27_label = validate_label_range.groupby(['user_age_level', 'category'], as_index=False)['item_sales_level'].agg({'user_age_category_count': 'count'})
    fvalidate = pd.merge(fvalidate, d27_label, on=["category", 'user_age_level'], how="left")

    d28_label = validate_label_range.groupby(['user_occupation_id', 'category'], as_index=False)['item_sales_level'].agg({'user_occupation_id_category_count': 'count'})
    fvalidate = pd.merge(fvalidate, d28_label, on=["category", 'user_occupation_id'], how="left")



    return fvalidate


def extract_test_feature(test_feature_range,test_label_range):
    """
        特征区间提取特征
    """
    # 商铺出现的次数
    d1 = test_feature_range[['shop_id', 'item_sales_level']]
    d1 = d1.groupby(['shop_id']).agg('count').reset_index()
    d1.rename(columns={'item_sales_level': 'shop_count'}, inplace=True)
    ftest = pd.merge(test_label_range, d1, on="shop_id", how="left")

    # 用户和商铺出现的次数
    d2 = test_feature_range[['shop_id', 'user_id', 'item_sales_level']]
    d2 = d2.groupby(['shop_id', 'user_id']).agg('count').reset_index()
    d2.rename(columns={'item_sales_level': 'user_shop_count'}, inplace=True)
    ftest = pd.merge(ftest, d2, on=["shop_id", 'user_id'], how="left")

    # 用户出现次数
    d3 = test_feature_range[['user_id', 'item_sales_level']]
    d3 = d3.groupby(['user_id']).agg('count').reset_index()
    d3.rename(columns={'item_sales_level': 'user_count'}, inplace=True)
    ftest = pd.merge(ftest, d3, on="user_id", how="left")

    # 用户出现次数(交易的次数)
    d4 = test_feature_range[test_feature_range['is_trade'] == 1][['user_id', 'item_sales_level']]
    d4 = d4.groupby(['user_id']).agg('count').reset_index()
    d4.rename(columns={'item_sales_level': 'user_count_istrade'}, inplace=True)
    ftest = pd.merge(ftest, d4, on="user_id", how="left")

    # 空值替换
    ftest['user_count_istrade'].replace(np.nan,0, inplace=True)
    ftest['user_count'].replace(np.nan,0, inplace=True)

    ftest['user_istrade_rate'] = ftest['user_count_istrade'].astype('float') / ftest['user_count'].astype('float')
    #
    # # 贝叶斯平滑
    ftest['user_istrade_rate_smooth'] = Bayesian_smoooth(ftest['user_count_istrade'],ftest['user_count'], 'test1')
    del ftest['user_istrade_rate']

    # 广告出现的次数
    d5 = test_feature_range[['item_id', 'item_sales_level']]
    d5 = d5.groupby(['item_id']).agg('count').reset_index()
    d5.rename(columns={'item_sales_level': 'item_count'}, inplace=True)
    ftest = pd.merge(ftest, d5, on="item_id", how="left")

    # 该商家的特定广告出现的次数
    d6 = test_feature_range[['item_id', 'shop_id', 'item_sales_level']]
    d6 = d6.groupby(['item_id', 'shop_id']).agg('count').reset_index()
    d6.rename(columns={'item_sales_level': 'shop_item_count'}, inplace=True)
    ftest = pd.merge(ftest, d6, on=["item_id", 'shop_id'], how="left")

    # 特定用户和该广告出现的次数
    d7 = test_feature_range[['user_id', 'item_id', 'item_sales_level']]
    d7 = d7.groupby(['user_id', 'item_id']).agg('count').reset_index()
    d7.rename(columns={'item_sales_level': 'user_item_count'}, inplace=True)
    ftest = pd.merge(ftest, d7, on=["user_id", 'item_id'], how="left")

    # 特定用户和该广告出现的次数(交易成功)
    d8 = test_feature_range[test_feature_range.is_trade == 1][['user_id', 'item_id', 'item_sales_level']]
    d8 = d8.groupby(['user_id', 'item_id']).agg('count').reset_index()
    d8.rename(columns={'item_sales_level': 'user_item_count_istrade'}, inplace=True)
    ftest = pd.merge(ftest, d8, on=["user_id", 'item_id'], how="left")

    # 空值替换
    ftest['user_item_count'].replace(np.nan, 0, inplace=True)
    ftest['user_item_count_istrade'].replace(np.nan,0, inplace=True)

    # 交易率
    ftest['user_item_count_istrade_rate'] = ftest['user_item_count_istrade'].astype('float') / ftest['user_item_count'].astype('float')
    # # 贝叶斯平滑
    ftest['user_item_count_istrade_rate_smooth'] = Bayesian_smoooth(ftest['user_item_count_istrade'], ftest['user_item_count'],'test2')
    del ftest['user_item_count_istrade_rate']

    # 用户和该用户的职业出现的次数
    d9 = test_feature_range[['user_id', 'user_occupation_id', 'item_sales_level']]
    d9 = d9.groupby(['user_id', 'user_occupation_id']).agg('count').reset_index()
    d9.rename(columns={'item_sales_level': 'user_and_user_occupation_count'}, inplace=True)
    ftest = pd.merge(ftest, d9, on=["user_id", 'user_occupation_id'], how="left")


    # 该职业出现的次数
    d13 = test_feature_range[['user_occupation_id', 'item_price_level']]
    d13 = d13.groupby(['user_occupation_id']).agg('count').reset_index()
    d13.rename(columns={'item_price_level': 'user_occupation_id_count'}, inplace=True)
    ftest = pd.merge(ftest, d13, on="user_occupation_id", how="left")

    # 广告价格等级出现的次数
    d14 = test_feature_range[['item_price_level', 'user_id']]
    d14 = d14.groupby(['item_price_level']).agg('count').reset_index()
    d14.rename(columns={'user_id': 'item_price_level_count'}, inplace=True)
    ftest = pd.merge(ftest, d14, on="item_price_level", how="left")

    # 性别对于是否交易的影响
    d15 = test_feature_range[['user_gender_id', 'user_id']]
    d15 = d15.groupby(['user_gender_id']).agg('count').reset_index()
    d15.rename(columns={'user_id': 'user_gender_id_count'}, inplace=True)
    ftest = pd.merge(ftest, d15, on="user_gender_id", how="left")


    # 看当前日期是否是高峰期
    ftest['is_high_trade'] = ftest['context_timestamp'].apply(date2morning_afternoon)


    # 收藏级别出现的次数
    d17 = test_feature_range[['item_collected_level', 'item_price_level']]
    d17 = d17.groupby(['item_collected_level']).agg('count').reset_index()
    d17.rename(columns={'item_price_level': 'item_collected_count'}, inplace=True)
    ftest = pd.merge(ftest, d17, on="item_collected_level", how="left")

    # 广告商品页面展示标号
    d18 = test_feature_range[['context_page_id', 'item_id', 'item_price_level']]
    d18 = d18.groupby(['context_page_id', 'item_id']).agg('count').reset_index()
    d18.rename(columns={'item_price_level': 'context_page_item_count'}, inplace=True)
    ftest = pd.merge(ftest, d18, on=["context_page_id", 'item_id'], how="left")

    # 用户和商铺,广告出现的次数
    d20 = test_feature_range[['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d20 = d20.groupby(['shop_id', 'user_id', 'item_id']).agg('count').reset_index()
    d20.rename(columns={'item_sales_level': 'user_shop_item_count'}, inplace=True)
    ftest = pd.merge(ftest, d20, on=["shop_id", 'user_id', 'item_id'], how="left")


    test_feature_range['day'] = test_feature_range['context_timestamp'].map(lambda x: int(x[6:8]))
    test_feature_range['hour'] = test_feature_range['context_timestamp'].map(lambda x: int(x[8:10]))

    ftest['day'] = ftest['context_timestamp'].map(lambda x: int(x[6:8]))
    ftest['hour'] = ftest['context_timestamp'].map(lambda x: int(x[8:10]))

    # user和shop在每一个小时(24个小时)的的count个数
    d21 = test_feature_range[['user_id', 'shop_id', 'hour', 'item_sales_level']]
    d21 = d21.groupby(['user_id', 'shop_id', 'hour']).agg('count').reset_index()
    d21.rename(columns={'item_sales_level': 'user_shop_hour_count'}, inplace=True)
    ftest = pd.merge(ftest, d21, on=['user_id', "shop_id", 'hour'], how="left")

    # user在每一个小时(24个小时)的的count个数
    d22 = test_feature_range[['user_id','hour', 'item_sales_level']]
    d22 = d22.groupby(['user_id','hour']).agg('count').reset_index()
    d22.rename(columns={'item_sales_level': 'user_hour_count'}, inplace=True)
    ftest = pd.merge(ftest, d22, on=["user_id", 'hour'], how="left")

    # shop在每一个小时(24个小时)的的count个数
    d23 = test_feature_range[['shop_id','hour', 'item_sales_level']]
    d23 = d23.groupby(['shop_id','hour']).agg('count').reset_index()
    d23.rename(columns={'item_sales_level': 'shop_hour_count'}, inplace=True)
    ftest = pd.merge(ftest, d23, on=["shop_id",'hour'], how="left")

    # item在每一个小时(24个小时)的的count个数
    d24 = test_feature_range[['item_id','hour', 'item_sales_level']]
    d24 = d24.groupby(['item_id','hour']).agg('count').reset_index()
    d24.rename(columns={'item_sales_level': 'item_hour_count'}, inplace=True)
    ftest = pd.merge(ftest, d24, on=["item_id",'hour'], how="left")

    # user和item在每一个小时(24个小时)的的count个数
    d29 = test_feature_range[['user_id', 'item_id', 'day', 'hour', 'item_sales_level']]
    d29 = d29.groupby(['user_id', 'item_id', 'day', 'hour']).agg('count').reset_index()
    d29.rename(columns={'item_sales_level': 'user_item_hour_count'}, inplace=True)
    ftest = pd.merge(ftest, d29, on=['user_id', "item_id", 'day', 'hour'], how="left")

    # 用户和广告出现的次数
    d30 = test_feature_range[test_feature_range.is_trade == 1][['item_id', 'item_sales_level']]
    d30 = d30.groupby(['item_id']).agg('count').reset_index()
    d30.rename(columns={'item_sales_level': 'item_count_istrade'}, inplace=True)
    ftest = pd.merge(ftest, d30, on=["item_id"], how="left")

    # 空值替换
    ftest['item_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['item_count'].replace(np.nan, 0, inplace=True)

    ftest['item_istrade_rate'] = ftest.item_count_istrade.astype('float') / ftest.item_count.astype('float')
    #
    # # 贝叶斯平滑
    ftest['item_istrade_rate_smooth'] = Bayesian_smoooth(ftest['item_count_istrade'],ftest['item_count'], 'test3')
    del ftest['item_istrade_rate']

    # shop交易count的比率
    d33 = test_feature_range[test_feature_range.is_trade == 1][['shop_id', 'item_sales_level']]
    d33 = d33.groupby(['shop_id']).agg('count').reset_index()
    d33.rename(columns={'item_sales_level': 'shop_count_istrade'}, inplace=True)
    ftest = pd.merge(ftest, d33, on=["shop_id"], how="left")

    ftest['shop_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['shop_count'].replace(np.nan, 0, inplace=True)

    ftest['shop_istrade_rate'] = ftest.shop_count_istrade.astype('float') / ftest.shop_count.astype('float')

    # # 贝叶斯平滑
    ftest['shop_istrade_rate_smooth'] = Bayesian_smoooth(ftest['shop_count_istrade'], ftest['shop_count'], 'test4')
    del ftest['shop_istrade_rate']

    # item_city_id的count
    d34 = test_feature_range[['item_city_id', 'item_sales_level']]
    d34 = d34.groupby(['item_city_id']).agg('count').reset_index()
    d34.rename(columns={'item_sales_level': 'item_city_id_count'}, inplace=True)
    ftest = pd.merge(ftest, d34, on="item_city_id", how="left")

    # user_id,item_city_id的count
    d35 = test_feature_range[['user_id','item_city_id', 'item_sales_level']]
    d35 = d35.groupby(['user_id','item_city_id']).agg('count').reset_index()
    d35.rename(columns={'item_sales_level': 'user_item_city_id_count'}, inplace=True)
    ftest = pd.merge(ftest, d35, on=["item_city_id",'user_id'], how="left")

    # user和shop的次数
    d36 = test_feature_range[test_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_sales_level']]
    d36 = d36.groupby(['shop_id', 'user_id']).agg('count').reset_index()
    d36.rename(columns={'item_sales_level': 'user_shop_count_istrade'}, inplace=True)
    ftest = pd.merge(ftest, d36, on=["shop_id", 'user_id'], how="left")

    ftest['user_shop_count'].replace(np.nan, 0, inplace=True)
    ftest['user_shop_count_istrade'].replace(np.nan, 0, inplace=True)

    # ftest['user_shop_count_minus_user_shop_istrade'] = ftest.user_shop_count.astype('int') - ftest.user_shop_count_istrade.astype('int')
    ftest['user_shop_count_istrade_rate'] = ftest.user_shop_count_istrade.astype('float') / ftest.user_shop_count.astype('float')

    # # 贝叶斯平滑
    ftest['user_shop_count_istrade_rate_smooth'] = Bayesian_smoooth(ftest['user_shop_count_istrade'], ftest['user_shop_count'], 'test5')
    del ftest['user_shop_count_istrade_rate']

    # 用户和广告出现的次数item_count  # 空值替换
    ftest['item_count'].replace(np.nan,0, inplace=True)
    ftest['item_count_istrade'].replace(np.nan,0, inplace=True)

    # # 用户和广告出现的次数item_count
    # ftest['item_count_minus_item_istrade'] = ftest.item_count.astype('int') - ftest.item_count_istrade.astype('int')
    # ftest['user_item_count_minus_user_item_istrade'] = ftest['user_item_count'].astype('int') - ftest['user_item_count_istrade'].astype('int')
    # ftest['user_count_minus_user_count_istrade'] = ftest['user_count'].astype('int') - ftest['user_count_istrade'].astype('int')

    # user和item_brand_id的次数
    d37 = test_feature_range.groupby(['user_id', 'item_brand_id'], as_index=False)['item_sales_level'].agg({'user_item_brand_count': 'count'})
    ftest = pd.merge(ftest, d37, on=["item_brand_id", 'user_id'], how="left")

    # user with shop_star_level count
    d38 = test_feature_range.groupby(['user_id', 'shop_star_level'], as_index=False)['item_sales_level'].agg({'user_shop_star_level_count': 'count'})
    ftest = pd.merge(ftest, d38, on=["shop_star_level", 'user_id'], how="left")

    # user with item_sales_level count
    d39 = test_feature_range.groupby(['user_id', 'item_sales_level'], as_index=False)['item_sales_level'].agg({'user_item_sales_level_count': 'count'})
    ftest = pd.merge(ftest, d39, on=["item_sales_level", 'user_id'], how="left")

    # user with item_collected_level count
    d40 = test_feature_range.groupby(['user_id', 'item_collected_level'], as_index=False)['item_sales_level'].agg({'user_item_collected_level_count': 'count'})
    ftest = pd.merge(ftest, d40, on=["item_collected_level", 'user_id'], how="left")
    #
    # user with category count
    d41 = test_feature_range.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count': 'count'})
    ftest = pd.merge(ftest, d41, on=["category", 'user_id'], how="left")

    #
    d44 = test_feature_range.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count': 'count'})
    ftest = pd.merge(ftest, d44, on=["item_city_id", 'user_id', 'item_id'], how="left")

    ######  new
    d45 = test_feature_range[test_feature_range.is_trade == 1][['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d45 = d45.groupby(['shop_id', 'user_id', 'item_id'], as_index=False)['item_sales_level'].agg({'user_shop_item_count_istrade': 'count'})
    ftest = pd.merge(ftest, d45, on=['shop_id', 'user_id', 'item_id'], how="left")

    ftest['user_shop_item_count'].replace(np.nan, 0, inplace=True)
    ftest['user_shop_item_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['user_shop_item_count_istrade_rate'] = ftest.user_shop_item_count_istrade.astype('float') / ftest.user_shop_item_count.astype('float')
    # # 贝叶斯平滑
    ftest['user_shop_item_count_istrade_rate_smooth'] = Bayesian_smoooth(ftest['user_shop_item_count_istrade'],ftest['user_shop_item_count'],'test6')
    del ftest['user_shop_item_count_istrade_rate']

    ####
    d46 = test_feature_range[test_feature_range.is_trade == 1][['user_id', 'item_id', 'item_city_id', 'item_sales_level']]
    d46 = d46.groupby(['user_id', 'item_id', 'item_city_id'], as_index=False)['item_sales_level'].agg({'user_item_item_city_id_count_istrade': 'count'})
    ftest = pd.merge(ftest, d46, on=['user_id', 'item_id', 'item_city_id'], how="left")

    ftest['user_item_item_city_id_count'].replace(np.nan, 0, inplace=True)
    ftest['user_item_item_city_id_count_istrade'].replace(np.nan, 0, inplace=True)

    ftest['user_item_item_city_id_count_istrade_rate'] = ftest.user_item_item_city_id_count_istrade.astype('float') / ftest.user_item_item_city_id_count.astype('float')
    #  贝叶斯平滑
    ftest['user_item_item_city_id_count_istrade_rate_smooth'] = Bayesian_smoooth(ftest['user_item_item_city_id_count_istrade'], ftest['user_item_item_city_id_count'],
                                                                                     'test7')
    del ftest['user_item_item_city_id_count_istrade_rate']

    ### user category
    d47 = test_feature_range[test_feature_range.is_trade == 1][['user_id', 'category', 'item_sales_level']]
    d47 = d47.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count_istrade': 'count'})
    ftest = pd.merge(ftest, d47, on=['user_id', 'category'], how="left")

    ftest['user_category_count'].replace(np.nan, 0, inplace=True)
    ftest['user_category_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['user_category_count_istrade_rate'] = ftest.user_category_count_istrade.astype('float') / ftest.user_category_count.astype('float')
    #  贝叶斯平滑
    ftest['user_category_count_istrade_rate_smooth'] = Bayesian_smoooth(ftest['user_category_count_istrade'],ftest['user_category_count'],'test8')
    del ftest['user_category_count_istrade_rate']


    # user_gender_id
    d48 = test_feature_range[test_feature_range.is_trade == 1][['user_gender_id','item_sales_level']]
    d48 = d48.groupby(['user_gender_id'], as_index=False)['item_sales_level'].agg({'user_gender_id_count_istrade': 'count'})
    ftest = pd.merge(ftest, d48, on=['user_gender_id'], how="left")

    ftest['user_gender_id_count'].replace(np.nan, 0, inplace=True)
    ftest['user_gender_id_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['user_gender_id_count_istrade_rate'] = ftest.user_gender_id_count_istrade.astype('float') / ftest.user_gender_id_count.astype('float')

    # user_occupation_id
    d49 = test_feature_range[test_feature_range.is_trade == 1][['user_occupation_id', 'item_sales_level']]
    d49 = d49.groupby(['user_occupation_id'], as_index=False)['item_sales_level'].agg({'user_occupation_id_count_istrade': 'count'})
    ftest = pd.merge(ftest, d49, on=['user_occupation_id'], how="left")

    ftest['user_occupation_id_count'].replace(np.nan, 0, inplace=True)
    ftest['user_occupation_id_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['user_occupation_id_count_istrade_rate'] = ftest.user_occupation_id_count_istrade.astype('float') / ftest.user_occupation_id_count.astype('float')

    # user_age_level
    d50 = test_feature_range[['user_age_level', 'item_sales_level']]
    d50 = d50.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count': 'count'})
    ftest = pd.merge(ftest, d50, on=['user_age_level'], how="left")

    d51 = test_feature_range[test_feature_range.is_trade == 1][['user_age_level', 'item_sales_level']]
    d51 = d51.groupby(['user_age_level'], as_index=False)['item_sales_level'].agg({'user_age_level_count_istrade': 'count'})
    ftest = pd.merge(ftest, d51, on=['user_age_level'], how="left")

    ftest['user_age_level_count'].replace(np.nan, 0, inplace=True)
    ftest['user_age_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['user_age_level_count_istrade_rate'] = ftest.user_age_level_count_istrade.astype('float') / ftest.user_age_level_count.astype('float')

    # item_sales_level
    d52 = test_feature_range[['item_sales_level', 'context_timestamp']]
    d52 = d52.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count': 'count'})
    ftest = pd.merge(ftest, d52, on=['item_sales_level'], how="left")

    d53 = test_feature_range[test_feature_range.is_trade == 1][['item_sales_level', 'context_timestamp']]
    d53 = d53.groupby(['item_sales_level'], as_index=False)['context_timestamp'].agg({'item_sales_level_count_istrade': 'count'})
    ftest = pd.merge(ftest, d53, on=['item_sales_level'], how="left")

    ftest['item_sales_level_count'].replace(np.nan, 0, inplace=True)
    ftest['item_sales_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['item_sales_level_count_istrade_rate'] = ftest.item_sales_level_count_istrade.astype('float') / ftest.item_sales_level_count.astype('float')

    # item_collected_level
    d54 = test_feature_range[test_feature_range.is_trade == 1][['item_collected_level', 'item_sales_level']]
    d54 = d54.groupby(['item_collected_level'], as_index=False)['item_sales_level'].agg({'item_collected_level_count_istrade': 'count'})
    ftest = pd.merge(ftest, d54, on=['item_collected_level'], how="left")

    ftest['item_collected_count'].replace(np.nan, 0, inplace=True)
    ftest['item_collected_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['item_collected_level_count_istrade_rate'] = ftest.item_collected_level_count_istrade.astype('float') / ftest.item_collected_count.astype('float')

    # user_star_level
    d55 = test_feature_range[['user_star_level', 'item_sales_level']]
    d55 = d55.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count': 'count'})
    ftest = pd.merge(ftest, d55, on=['user_star_level'], how="left")

    d56 = test_feature_range[test_feature_range.is_trade == 1][['user_star_level', 'item_sales_level']]
    d56 = d56.groupby(['user_star_level'], as_index=False)['item_sales_level'].agg({'user_star_level_count_istrade': 'count'})
    ftest = pd.merge(ftest, d56, on=['user_star_level'], how="left")

    ftest['user_star_level_count'].replace(np.nan, 0, inplace=True)
    ftest['user_star_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['user_star_level_count_istrade_rate'] = ftest.user_star_level_count_istrade.astype('float') / ftest.user_star_level_count.astype('float')

    # item_pv_level
    d57 = test_feature_range[['item_pv_level', 'item_sales_level']]
    d57 = d57.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count': 'count'})
    ftest = pd.merge(ftest, d57, on=['item_pv_level'], how="left")

    d58 = test_feature_range[test_feature_range.is_trade == 1][['item_pv_level', 'item_sales_level']]
    d58 = d58.groupby(['item_pv_level'], as_index=False)['item_sales_level'].agg({'item_pv_level_count_istrade': 'count'})
    ftest = pd.merge(ftest, d58, on=['item_pv_level'], how="left")

    ftest['item_pv_level_count'].replace(np.nan, 0, inplace=True)
    ftest['item_pv_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['item_pv_level_count_istrade_rate'] = ftest.item_pv_level_count_istrade.astype('float') / ftest.item_pv_level_count.astype('float')

    #  贝叶斯平滑
    ftest['item_pv_level_count_istrade_rate_smooth'] = Bayesian_smoooth(ftest['item_pv_level_count_istrade'],ftest['item_pv_level_count'], 'test9')
    del ftest['item_pv_level_count_istrade_rate']

    # shop_review_num_level
    d59 = test_feature_range[['shop_review_num_level', 'item_sales_level']]
    d59 = d59.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count': 'count'})
    ftest = pd.merge(ftest, d59, on=['shop_review_num_level'], how="left")

    d60 = test_feature_range[test_feature_range.is_trade == 1][['shop_review_num_level', 'item_sales_level']]
    d60 = d60.groupby(['shop_review_num_level'], as_index=False)['item_sales_level'].agg({'shop_review_num_level_count_istrade': 'count'})
    ftest = pd.merge(ftest, d60, on=['shop_review_num_level'], how="left")

    ftest['shop_review_num_level_count'].replace(np.nan, 0, inplace=True)
    ftest['shop_review_num_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['shop_review_num_level_count_istrade_rate'] = ftest.shop_review_num_level_count_istrade.astype('float') / ftest.shop_review_num_level_count.astype('float')

    #  贝叶斯平滑
    ftest['shop_review_num_level_count_istrade_rate_smooth'] = Bayesian_smoooth(ftest['shop_review_num_level_count_istrade'], ftest['shop_review_num_level_count'], 'test10')
    del ftest['shop_review_num_level_count_istrade_rate']


    # item_price_level
    d62 = test_feature_range[test_feature_range.is_trade == 1][['item_price_level', 'item_sales_level']]
    d62 = d62.groupby(['item_price_level'], as_index=False)['item_sales_level'].agg({'item_price_level_count_istrade': 'count'})
    ftest = pd.merge(ftest, d62, on=['item_price_level'], how="left")

    ftest['item_price_level_count'].replace(np.nan, 0, inplace=True)
    ftest['item_price_level_count_istrade'].replace(np.nan, 0, inplace=True)
    ftest['item_price_level_count_istrade_rate'] = ftest.item_price_level_count_istrade.astype('float') / ftest.item_price_level_count.astype('float')
    #  贝叶斯平滑
    ftest['item_price_level_count_istrade_rate_smooth'] = Bayesian_smoooth(ftest['item_price_level_count_istrade'], ftest['item_price_level_count'], 'test11')
    del ftest['item_price_level_count_istrade_rate']

    # # shop_star_level
    # d63 = test_feature_range[['shop_star_level', 'item_sales_level']]
    # d63 = d63.groupby(['shop_star_level'], as_index=False)['item_sales_level'].agg({'shop_star_level_count': 'count'})
    # ftest = pd.merge(ftest, d63, on=['shop_star_level'], how="left")
    #
    # d64 = test_feature_range[test_feature_range.is_trade == 1][['shop_star_level', 'item_sales_level']]
    # d64 = d64.groupby(['shop_star_level'], as_index=False)['item_sales_level'].agg({'shop_star_level_count_istrade': 'count'})
    # ftest = pd.merge(ftest, d64, on=['shop_star_level'], how="left")
    #
    # ftest['shop_star_level_count'].replace(np.nan, 0, inplace=True)
    # ftest['shop_star_level_count_istrade'].replace(np.nan, 0, inplace=True)
    # ftest['shop_star_level_count_istrade_rate'] = ftest.shop_star_level_count_istrade.astype('float') / ftest.shop_star_level_count.astype('float')
    # #  贝叶斯平滑
    # ftest['shop_star_level_count_istrade_rate_smooth'] = Bayesian_smoooth(ftest['shop_star_level_count_istrade'], ftest['shop_star_level_count'], 'test12')
    # del ftest['shop_star_level_count_istrade_rate']


    print("测试集特征区间结束!")

    """
        标签区间提取特征
    """
    ftest = get_test_label(ftest, test_label_range)
    print("测试集标签区间结束!")
    return ftest


def get_test_label(ftest, test_label_range):

    # 商铺出现的次数
    d1_label = test_label_range[['shop_id', 'item_sales_level']]
    d1_label = d1_label.groupby(['shop_id']).agg('count').reset_index()
    d1_label.rename(columns={'item_sales_level': 'shop_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d1_label, on="shop_id", how="left")

    # 用户和商铺出现的次数
    d2_label = test_label_range[['shop_id', 'user_id', 'item_sales_level']]
    d2_label = d2_label.groupby(['shop_id', 'user_id']).agg('count').reset_index()
    d2_label.rename(columns={'item_sales_level': 'user_shop_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d2_label, on=["shop_id", 'user_id'], how="left")

    d2 = d2_label.groupby(['user_id'], as_index=False)['shop_id'].agg({'user_diff_shop_count': 'count'})
    ftest = pd.merge(ftest, d2, on=["user_id"], how="left")

    d3 = d2_label.groupby(['shop_id'], as_index=False)['user_id'].agg({'shop_diff_user_count': 'count'})
    ftest = pd.merge(ftest, d3, on=["shop_id"], how="left")

    # 用户出现次数
    d3_label = test_label_range[['user_id', 'item_sales_level']]
    d3_label = d3_label.groupby(['user_id']).agg('count').reset_index()
    d3_label.rename(columns={'item_sales_level': 'user_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d3_label, on="user_id", how="left")

    # 广告出现的次数
    d4_label = test_label_range[['item_id', 'item_sales_level']]
    d4_label = d4_label.groupby(['item_id']).agg('count').reset_index()
    d4_label.rename(columns={'item_sales_level': 'item_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d4_label, on="item_id", how="left")

    # 该商家的特定广告出现的次数
    d5_label = test_label_range[['item_id', 'shop_id', 'item_sales_level']]
    d5_label = d5_label.groupby(['item_id', 'shop_id']).agg('count').reset_index()
    d5_label.rename(columns={'item_sales_level': 'shop_item_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d5_label, on=["item_id", 'shop_id'], how="left")

    # 特定用户和该广告出现的次数
    d6_label = test_label_range[['user_id', 'item_id', 'item_sales_level']]
    d6_label = d6_label.groupby(['user_id', 'item_id']).agg('count').reset_index()
    d6_label.rename(columns={'item_sales_level': 'user_item_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d6_label, on=["user_id", 'item_id'], how="left")

    d6 = d6_label.groupby(['user_id'], as_index=False)['item_id'].agg({'user_diff_item_count': 'count'})
    ftest = pd.merge(ftest, d6, on=["user_id"], how="left")

    d7 = d6_label.groupby(['item_id'], as_index=False)['user_id'].agg({'item_diff_user_count': 'count'})
    ftest = pd.merge(ftest, d7, on=["item_id"], how="left")

    # 用户和该用户的职业出现的次数
    d7_label = test_label_range[['user_id', 'user_occupation_id', 'item_sales_level']]
    d7_label = d7_label.groupby(['user_id', 'user_occupation_id']).agg('count').reset_index()
    d7_label.rename(columns={'item_sales_level': 'user_and_user_occupation_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d7_label, on=["user_id", 'user_occupation_id'], how="left")

    # 用户和商铺,广告出现的次数
    d8_label = test_label_range[['shop_id', 'user_id', 'item_id', 'item_sales_level']]
    d8_label = d8_label.groupby(['shop_id', 'user_id', 'item_id']).agg('count').reset_index()
    d8_label.rename(columns={'item_sales_level': 'user_shop_item_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d8_label, on=["shop_id", 'user_id', 'item_id'], how="left")


    test_label_range['day'] = test_label_range['context_timestamp'].map(lambda x: int(x[6:8]))
    test_label_range['hour'] = test_label_range['context_timestamp'].map(lambda x: int(x[8:10]))

    # user在每一个小时(24个小时)的的count个数
    d10_label = test_label_range[['user_id', 'day', 'hour', 'item_sales_level']]
    d10_label = d10_label.groupby(['user_id', 'day', 'hour']).agg('count').reset_index()
    d10_label.rename(columns={'item_sales_level': 'user_hour_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d10_label, on=["user_id", 'day', 'hour'], how="left")

    # shop在每一个小时(24个小时)的的count个数
    d11_label = test_label_range[['shop_id', 'day', 'hour', 'item_sales_level']]
    d11_label = d11_label.groupby(['shop_id', 'day', 'hour']).agg('count').reset_index()
    d11_label.rename(columns={'item_sales_level': 'shop_hour_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d11_label, on=["shop_id", 'day', 'hour'], how="left")

    # item在每一个小时(24个小时)的的count个数
    d12_label = test_label_range[['item_id', 'day', 'hour', 'item_sales_level']]
    d12_label = d12_label.groupby(['item_id', 'day', 'hour']).agg('count').reset_index()
    d12_label.rename(columns={'item_sales_level': 'item_hour_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d12_label, on=["item_id", 'day', 'hour'], how="left")

    # user和item的context_stamp的排名(升序)
    ftest['context_timestamp_user_item_rank_label'] = ftest.groupby(['user_id', 'item_id'])['context_timestamp'].rank(
        ascending=True)
    ftest.drop_duplicates(inplace=True)
    # user和item的context_stamp的排名(降序)
    ftest['context_timestamp_user_item_rank_label_desc'] = ftest.groupby(['user_id', 'item_id'])['context_timestamp'].rank(
        ascending=False)
    ftest.drop_duplicates(inplace=True)


    # user和shop的context_stamp的排名(升序)
    ftest['context_timestamp_user_shop_rank_label'] = ftest.groupby(['user_id', 'shop_id'])['context_timestamp'].rank(
        ascending=True)
    ftest.drop_duplicates(inplace=True)
    # user和shop的context_stamp的排名(降序)
    ftest['context_timestamp_user_shop_rank_label_desc'] = ftest.groupby(['user_id', 'shop_id'])['context_timestamp'].rank(
        ascending=False)
    ftest.drop_duplicates(inplace=True)


    # 对同一个用户的时间进行排序(升序)
    ftest['context_timestamp_rank_label'] = ftest.groupby(['user_id'])['context_timestamp'].rank(ascending=True)
    ftest.drop_duplicates(inplace=True)
    # 对同一个用户的时间进行排序(降序)
    ftest['context_timestamp_rank_desc_label'] = ftest.groupby(['user_id'])['context_timestamp'].rank(ascending=False)
    ftest.drop_duplicates(inplace=True)


    # 对同一个shop的时间进行排序(升序)
    ftest['context_timestamp_shop_rank_label'] = ftest.groupby(['shop_id'])['context_timestamp'].rank(
        ascending=True)
    ftest.drop_duplicates(inplace=True)
    # 对同一个shop的时间进行排序(降序)
    ftest['context_timestamp_shop_rank_desc_label'] = ftest.groupby(['shop_id'])['context_timestamp'].rank(
        ascending=False)
    ftest.drop_duplicates(inplace=True)

    # item的context_stamp的排名(升序)
    ftest['context_timestamp_item_rank_label'] = ftest.groupby(['item_id'])['context_timestamp'].rank(ascending=True)
    ftest.drop_duplicates(inplace=True)
    # item的context_stamp的排名(降序)
    ftest['context_timestamp_item_rank_label_desc'] = ftest.groupby(['item_id'])['context_timestamp'].rank(ascending=False)
    ftest.drop_duplicates(inplace=True)

    ftest['aa_user_item_shop_rank_label'] = ftest.groupby(['user_id', 'item_id', 'shop_id'])['context_timestamp'].rank(ascending=True)
    ftest.drop_duplicates(inplace=True)

    ftest['aa_user_item_shop_rank_desc_label'] = ftest.groupby(['user_id', 'item_id', 'shop_id'])['context_timestamp'].rank(ascending=False)
    ftest.drop_duplicates(inplace=True)

    # 广告商品页面展示标号
    d14_label = test_label_range[['context_page_id', 'item_id', 'item_price_level']]
    d14_label = d14_label.groupby(['context_page_id', 'item_id']).agg('count').reset_index()
    d14_label.rename(columns={'item_price_level': 'context_page_item_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d14_label, on=["context_page_id", 'item_id'], how="left")

    # user和shop店铺评价的均值
    d15_label = test_label_range[['user_id', 'shop_id', 'shop_review_num_level']]
    d15_label = d15_label.groupby(['user_id', 'shop_id']).agg('mean').reset_index()
    d15_label.rename(columns={'shop_review_num_level': 'user_shop_mean_label'}, inplace=True)
    ftest = pd.merge(ftest, d15_label, on=["user_id", 'shop_id'], how="left")

    # user和item在每一个小时(24个小时)的的count个数
    d17_label = test_label_range[['user_id', 'item_id', 'day', 'hour', 'item_sales_level']]
    d17_label = d17_label.groupby(['user_id', 'item_id', 'day', 'hour']).agg('count').reset_index()
    d17_label.rename(columns={'item_sales_level': 'user_item_hour_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d17_label, on=['user_id', "item_id", 'day', 'hour'], how="left")

    # user和shop在每一个小时(24个小时)的的count个数
    d18_label = test_label_range[['user_id', 'shop_id', 'hour', 'day', 'item_sales_level']]
    d18_label = d18_label.groupby(['user_id', 'shop_id', 'hour', 'day']).agg('count').reset_index()
    d18_label.rename(columns={'item_sales_level': 'user_shop_hour_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d18_label, on=['user_id', "shop_id", 'hour', 'day'], how="left")

    # user_id,item_city_id的count
    d20_label = test_label_range[['user_id', 'item_city_id', 'item_sales_level']]
    d20_label = d20_label.groupby(['user_id', 'item_city_id']).agg('count').reset_index()
    d20_label.rename(columns={'item_sales_level': 'user_item_city_id_count_label'}, inplace=True)
    ftest = pd.merge(ftest, d20_label, on=["item_city_id", 'user_id'], how="left")

    # 用户上下一次点击广告的时间间隔
    d24_label = test_label_range[['item_id', 'user_id', 'context_timestamp']]
    d24_label = d24_label.groupby(['user_id', 'item_id'])['context_timestamp'].agg(lambda x: ':'.join(x)).reset_index()
    d24_label.rename(columns={'context_timestamp': 'dates'}, inplace=True)
    ftest = pd.merge(ftest, d24_label, on=["user_id", "item_id"], how="left")

    ftest['context_timestamp_and_dates'] = ftest.context_timestamp.astype('str') + '-' + ftest.dates
    ftest['user_before_day_click_item_gap'] = ftest.context_timestamp_and_dates.apply(get_day_gap_before)
    ftest['user_after_day_click_item_gap'] = ftest.context_timestamp_and_dates.apply(get_day_gap_after)

    # 用户是否是第一次点击特定广告
    ftest["is_first_get_coupon"] = ftest.context_timestamp_and_dates.apply(is_first_get_coupon)
    # 用户是否是最后一次点击特定广告
    ftest["is_last_get_coupon"] = ftest.context_timestamp_and_dates.apply(is_last_get_coupon)

    d25_label = test_label_range.groupby(['user_id', 'category'], as_index=False)['item_sales_level'].agg({'user_category_count_label': 'count'})
    ftest = pd.merge(ftest, d25_label, on=["category", 'user_id'], how="left")

    # process property
    ftest['property_count'] = ftest['item_property_list'].apply(get_property_info)
    ftest['property'] = ftest['property_count'].apply(lambda x: x.split(';')[0])
    ftest['property_max_count'] = ftest['property_count'].apply(lambda x: x.split(';')[1])
    del ftest['property_count']

    # user with category rank
    ftest['user_and_category_rank'] = ftest.groupby(['user_id','category'])['context_timestamp'].rank(ascending=True)
    ftest['user_and_category_rank_desc'] = ftest.groupby(['user_id', 'category'])['context_timestamp'].rank(ascending=False)

    # gender_id with category
    test_range = test_label_range[test_label_range.user_gender_id != -1]
    d26_label = test_range.groupby(['user_gender_id', 'category'], as_index=False)['item_sales_level'].agg({'gender_category_count': 'count'})
    ftest = pd.merge(ftest, d26_label, on=["category", 'user_gender_id'], how="left")

    d27_label = test_label_range.groupby(['user_age_level', 'category'], as_index=False)['item_sales_level'].agg({'user_age_category_count': 'count'})
    ftest = pd.merge(ftest, d27_label, on=["category", 'user_age_level'], how="left")

    d28_label = test_label_range.groupby(['user_occupation_id', 'category'], as_index=False)['item_sales_level'].agg({'user_occupation_id_category_count': 'count'})
    ftest = pd.merge(ftest, d28_label, on=["category", 'user_occupation_id'], how="left")







    return ftest



def main():
    train_feature_range1, train_label_range1, \
    train_feature_range2, train_label_range2, \
    train_feature_range3, train_label_range3, \
    train_feature_range4, train_label_range4, \
    validate_feature_range, validate_label_range,\
    test_feature_range,test_label_range = load_data_one()



    ################################################## multi_process start ##############################################
    process_list = []
    pool = ProcessPoolExecutor(max_workers=50)

    tr1 = pool.submit(extract_train_feature,(train_feature_range1),(train_label_range1),('one'))
    tr2 = pool.submit(extract_train_feature,(train_feature_range2), (train_label_range2),('two'))
    tr3 = pool.submit(extract_train_feature,(train_feature_range3), (train_label_range3),('three'))
    tr4 = pool.submit(extract_train_feature,(train_feature_range4), (train_label_range4),('four'))

    val = pool.submit(extract_validate_feature,(validate_feature_range),(validate_label_range))
    test = pool.submit(extract_test_feature,(test_feature_range), (test_label_range))

    process_list.append(tr1)
    process_list.append(tr2)
    process_list.append(tr3)
    process_list.append(tr4)

    process_list.append(val)
    process_list.append(test)

    wait(process_list,timeout=None,return_when='ALL_COMPLETED')

    train1 = tr1.result()
    train2 = tr2.result()
    train3 = tr3.result()
    train4 = tr4.result()

    validate = val.result()
    test = test.result()
    ################################################## multi_process end ##############################################



    ftrain = pd.concat([train1, train2, train3, train4], axis=0)

    ftrain.to_csv('data/ftrain.csv', index=None)
    validate.to_csv('data/fvalidate.csv', index=None)
    test.to_csv('data/ftest.csv', index=None)



if __name__ == '__main__':
    main()