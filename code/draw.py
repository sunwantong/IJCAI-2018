import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def draw():
    dframe = pd.read_csv('../round2_data/round2_train.csv', sep=",")
    dframe_test = pd.read_csv('../round2_data/round2_test.csv', sep=",")
    #

    dframe_bak = dframe.copy()
    dframe_test_bak = dframe_test.copy()
    # df = dframe
    # df['context_timestamp'] = df['context_timestamp'].astype('str')
    # df['day'] = df['context_timestamp'].map(lambda x: x[6:8])
    # df = df[['day', 'instance_id']]
    # df.groupby(['day']).count().plot.bar()
    # plt.savefig('icon/day_count.png')
    # plt.show()

    df = dframe.append(dframe_test)
    df['context_timestamp'] = df['context_timestamp'].astype('str')
    df['day'] = df['context_timestamp'].map(lambda x: x[6:8])
    df['hour'] = df['context_timestamp'].map(lambda x: int(x[8:10]))
    df_bak = df.copy()
    # #
    # #
    df = df[df.hour < 12][['day', 'instance_id']]
    df_mor = df.groupby(['day'],as_index=False)['instance_id'].agg({'day_mor_count': 'count'})

    df_bak = df_bak[df_bak.hour >= 12][['day', 'instance_id']]
    df_night = df_bak.groupby(['day'], as_index=False)['instance_id'].agg({'day_ni_count': 'count'})

    df = pd.merge(df_mor,df_night,on=['day'],how='left')
    # df_all_count = df
    df = df.set_index('day')
    df = df.reindex(['31','01','02','03','04','05','06','07'])

    df.plot.line()
    plt.savefig('../icon/all_line.png')
    plt.show()

    # df = dframe_bak.append(dframe_test_bak)
    # df = df[df.is_trade == 1]
    # df['context_timestamp'] = df['context_timestamp'].astype('str')
    # df['day'] = df['context_timestamp'].map(lambda x: x[6:8])
    # df['hour'] = df['context_timestamp'].map(lambda x: int(x[8:10]))
    # df_bak = df.copy()
    # # #
    # # #
    # df = df[df.hour < 12][['day', 'instance_id']]
    # df_mor = df.groupby(['day'], as_index=False)['instance_id'].agg({'day_mor_count_istrade': 'count'})
    #
    # df_bak = df_bak[df_bak.hour > 12][['day', 'instance_id']]
    # df_night = df_bak.groupby(['day'], as_index=False)['instance_id'].agg({'day_ni_count_istrade': 'count'})
    #
    # df_istrade = pd.merge(df_mor, df_night, on=['day'], how='left')
    # # df = df.set_index('day')
    # #
    # # df.plot.bar()
    # # plt.savefig('icon/mor_day_count_istrade.png')
    # # plt.show()
    #
    # df_all = pd.merge(df_all_count, df_istrade, on=['day'], how='left')
    # df_all['day_mor_count_istrade_rate'] = df_all['day_mor_count_istrade'] / df_all['day_mor_count']
    # df_all['day_ni_count_istrade_rate'] = df_all['day_ni_count_istrade'] / df_all['day_ni_count']
    # df_all = df_all[['day','day_mor_count_istrade_rate','day_ni_count_istrade_rate']]
    # df_all = df_all.set_index('day')
    # df_all.plot.bar()
    # plt.savefig('icon/mor_day_count_istrade_rate.png')
    # plt.show()









    # df = df[df.day == '04'][['hour','instance_id']]
    # df.groupby(['hour']).count().plot.bar()
    # plt.savefig('icon/day_count_04.png')
    # plt.show()
    #
    # df = df[df.day == '07'][['hour', 'instance_id']]
    # df.groupby(['hour']).count().plot.bar()
    # plt.savefig('icon/day_count_07.png')
    # plt.show()


def draw_divide_feature_size():
    dframe = pd.read_csv('round2_data/round2_train.csv', sep=",")
    dframe_test = pd.read_csv('round2_data/round2_test_a.csv', sep=",")

    df = dframe
    df_test = dframe_test

    df['context_timestamp'] = df['context_timestamp'].astype('str')
    df_test['context_timestamp'] = df_test['context_timestamp'].astype('str')

    test_user_id = df_test['user_id']
    test_user_id_list = list(test_user_id)

    res = list(set(df['user_id']).intersection(set(df_test['user_id'])))
    print(len(res))



    data = df[df['user_id'].isin(test_user_id_list)]

    print(len(data))

    print(data['context_timestamp'].min())
    print(data['context_timestamp'].max())
    print(data[data.context_timestamp == '20180831000006'][['user_id']])

    test_data = df_test[df_test['user_id'].isin(['7027612589588888174', '7027612589588888174'])]
    print(test_data)


    # data['day'] = data['context_timestamp'].map(lambda x: x[6:8])
    # data = data[['day','user_id']]
    # data.groupby(['day']).count().plot.bar()
    # plt.show()
    # df['context_timestamp'] = df['context_timestamp'].astype('str')
    # df['day'] = df['context_timestamp'].map(lambda x: x[6:8])
    # df['hour'] = df['context_timestamp'].map(lambda x: int(x[8:10]))
    # df_bak = df.copy()

def main():
    draw()
    # draw_divide_feature_size()

if __name__ == '__main__':
    main()