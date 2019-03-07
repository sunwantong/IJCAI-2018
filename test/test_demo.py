import pandas as pd

"""

368.63602056505005  have context_page_id

371.50857337030004  haven't context_page_id
context_page_id is used? 

"""

if __name__ == '__main__':
    dframe1 = pd.read_csv('../txt/result.txt', sep=" ")
    # dframe3 = pd.read_csv('../txt/result_stacking.txt', sep=" ")

    a = dframe1['predicted_score'].sum()
    # c = dframe3['predicted_score'].sum()
    print(a)