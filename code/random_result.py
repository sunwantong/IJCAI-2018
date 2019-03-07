import pandas as pd
import numpy as np


"""
   df['scrore']=df['instance_id'].map(lambda x:random.uniform(0,0.3))
"""

def set_value(s):
    random = np.random.RandomState(0)  # RandomState生成随机数种子
    a = random.uniform(0, 0.31)  # 随机数范围
    x = round(a,9)
    return x

def load_csv():
    random = np.random.RandomState(0)  # RandomState生成随机数种子
    dframe_test = pd.read_csv('round2_data/round2_test_a.submit', sep=" ")
    dframe_test['predicted_score'] = 1
    dframe_test = dframe_test[['instance_id','user_id']]

    values = []
    for i in range(len(dframe_test)):
        a = random.uniform(0, 0.31)  # 随机数范围
        x = round(a, 9)
        values.append(x)

    a = pd.DataFrame(values,columns=['predicted_score'])
    dframe_test = pd.concat([dframe_test['instance_id'],a],axis=1)
    print(dframe_test[['instance_id','predicted_score']])#
    dframe_test.to_csv("submit/result_random.submit", index=None, sep=' ', line_terminator='\r')

    print(len(dframe_test))


def main():
    load_csv()

if __name__ == '__main__':
    main()
