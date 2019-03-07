import pandas as pd




def cos(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)

def main():
    result = pd.read_csv('submit/result_1dot5days(1.075).txt', sep=" ")
    result_chengji = pd.read_csv('submit/result_1dot5days(5.3).txt', sep=" ")

    a = result['predicted_score'].astype('float')
    b = result_chengji['predicted_score'].astype('float')
    cal_value = cos(a,b)
    print(cal_value)

if __name__ == '__main__':
    main()