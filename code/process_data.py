import pandas as pd
import datetime
import time

"""
 

"""
def parse_time(dframe):
    def convert_time(my_time):
        timeArray = time.localtime(my_time)
        otherStyleTime = time.strftime("%Y%m%d%H%M%S", timeArray)
        return otherStyleTime

    dframe['context_timestamp'] = dframe['context_timestamp'].map(convert_time)
    return dframe



def filter_feature():
    df_train = pd.read_csv('../data/ftrain.csv', sep=",")
    na_count = df_train.isnull().sum().sort_values(ascending=False)
    # print(na_count)
    # na_count = df_train[df_train == '-1'].sum().sort_values(ascending=False)
    na_rate = na_count / len(df_train)
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'ratio'])
    print(na_data.head(20))

def main():
    # dframe = pd.read_csv('round2_data/round2_train.txt', sep=" ")
    # dframe_test = pd.read_csv('../round2_data/round2_test_b.txt', sep=" ")
    #
    # dframe = parse_time(dframe)
    # dframe_test = parse_time(dframe_test)
    # #
    # dframe.to_csv("round2_data/round2_train.csv", index=None)
    dframe_test.to_csv("../round2_data/round2_test_b.csv", index=None)
    #
    # print(dframe['context_timestamp'].max(), dframe['context_timestamp'].min())
    print(dframe_test['context_timestamp'].max(), dframe_test['context_timestamp'].min())

    # filter_feature()

if __name__ == '__main__':
    main()