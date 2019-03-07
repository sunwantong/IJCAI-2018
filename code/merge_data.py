import pandas as pd




def merge_data():
    test_a = pd.read_csv('../round2_data/round2_test_a.csv',sep=",")
    test_b = pd.read_csv('../round2_data/round2_test_b.csv', sep=",")

    test = pd.concat([test_b, test_a], axis=0)
    test.to_csv('../round2_data/round2_test.csv', index=None)


def merge_a_b():
    r_full = pd.read_csv('../submit/result_A_B_full(5.13).txt', sep=" ")
    r_a = pd.read_csv('../submit/result_1dot5days(5.3).txt', sep=" ")
    r_a.rename(columns={'predicted_score': 'pred'}, inplace=True)
    res = pd.merge(r_full, r_a, on=['instance_id'], how='left')

    res = res[res['pred'].isnull()]

    res = res[['instance_id', 'predicted_score']]
    print(res['predicted_score'].sum())
    res.to_csv("../submit/result_B(5.13).txt", index=None, sep=' ', line_terminator='\r')

def main():
    # merge_data()
    merge_a_b()


if __name__ == '__main__':
    main()