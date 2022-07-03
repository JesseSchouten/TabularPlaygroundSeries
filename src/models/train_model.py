from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import pickle

def read_data():
    dtypes = {
        'f_00':'float32',
        'f_01': 'float32',
        'f_02':'float32',
        'f_03': 'float32',
        'f_04':'float32',
        'f_05': 'float32',
        'f_06':'float32',
        'f_07': 'int8',
        'f_08':'int8',
        'f_09': 'int8',
        'f_10':'int8',
        'f_11': 'int8',
        'f_12':'int8',
        'f_13': 'int8',
        'f_14':'float32',
        'f_15': 'float32',
        'f_16':'float32',
        'f_17': 'float32',
        'f_18':'float32',
        'f_19': 'float32',
        'f_20':'float32',
        'f_21': 'float32',
        'f_22':'float32',
        'f_23': 'float32',
        'f_24':'float32',
        'f_25': 'float32',
        'f_26':'float32',
        'f_27': 'float32',
        'f_28':'float32',
        'f_29': 'float32',
         }
    df = pd.read_csv('data/raw/data.csv')
    return df

def get_float_cols(df):
    float_cols = []
    for c,t in zip([col for col in df.columns], [col for col in df.dtypes]):
        if 'float' in str(t):
            float_cols.append(c)
    return float_cols

def main():
    df = read_data()
    float_col_list = get_float_cols(df)
    X = df[float_col_list]
    gmm = GaussianMixture(n_components=9,covariance_type='diag', random_state=0)
    gmm.fit(X)
    gmm_name = 'my_baseline_model'
    with open('src/models/'+ gmm_name + '.pkl', 'wb') as file:
        pickle.dump(gmm, file)

if __name__ == '__main__':
    main()