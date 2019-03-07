import pandas as pd
import numpy as np



"""
10645.353774053106
"""
def load_csv():
    # result = pd.read_csv('../round2_data/round2_test_b.csv',sep=",")
    # print(len(result))
    result = pd.read_csv('../submit/result_B(5.13).txt', sep=" ")
    result['predicted_score'] = result['predicted_score'] / 1.216315
    result.to_csv("../submit/result_B(chu_1.216315_5.13).txt", index=None, sep=' ', line_terminator='\r')
    print(result[['predicted_score']].sum())
    print(len(result))

def model_fuse():
    res = pd.read_csv('../submit/result_B(chu_1.216315_5.13).txt', sep=" ")
    res2 = pd.read_csv('../submit/result_B(1.3892_510).txt', sep=" ")

    r = res["predicted_score"]
    r_s = res2["predicted_score"]

    res.rename(columns={'predicted_score': 'pred'}, inplace=True)
    res["predicted_score"] = list(map(lambda x, y: (x * 0.6 + y * 0.4), r,r_s))
    result = res[['instance_id','predicted_score']]
    result.to_csv("../submit/result_fuse(5.13).txt", index=None, sep=' ', line_terminator='\r')

    print(result['predicted_score'].sum())

def main():
    # load_csv()
    model_fuse()

if __name__ == '__main__':
    main()











































































































































