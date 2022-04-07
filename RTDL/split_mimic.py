import pandas as pd
from sklearn.model_selection import train_test_split

mimic = pd.read_csv('data/metaMIMIC.csv', delimiter = ',')
mimic_target_columns = ['diabetes_diagnosed', 'hypertensive_diagnosed', 'ischematic_diagnosed',
                          'heart_diagnosed', 'overweight_diagnosed', 'anemia_diagnosed', 'respiratory_diagnosed',
                          'hypotension_diagnosed', 'lipoid_diagnosed', 'atrial_diagnosed', 'purpura_diagnosed', 'alcohol_diagnosed']
y_full = mimic[mimic_target_columns]
mimic.drop(columns = ['subject_id'], inplace = True)
mimic.drop(columns = mimic_target_columns, inplace = True)
X_full = mimic.astype('float')
categorical_columns = ['gender']
numerical_columns = list(X_full.columns[X_full.columns != 'gender'])
X_full.loc[X_full['gender'] == 1, 'gender'] = 'male'
X_full.loc[X_full['gender'] == 0, 'gender'] = 'female'


X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1875, random_state=1) # 0.1875 x 0.8 = 0.15

X_train.to_csv('data/mimic_train_X.csv', index = False)
X_val.to_csv('data/mimic_val_X.csv', index = False)
X_test.to_csv('data/mimic_test_X.csv', index = False)
y_train.to_csv('data/mimic_train_y.csv', index = False)
y_val.to_csv('data/mimic_val_y.csv', index = False)
y_test.to_csv('data/mimic_test_y.csv', index = False)


# mimic_target_columns = ['diabetes_diagnosed', 'hypertensive_diagnosed', 'ischematic_diagnosed',
#                   'heart_diagnosed', 'overweight_diagnosed', 'anemia_diagnosed', 'respiratory_diagnosed',
#                   'hypotension_diagnosed', 'lipoid_diagnosed', 'atrial_diagnosed', 'purpura_diagnosed', 'alcohol_diagnosed']
# y_full = mimic[mimic_target_columns]
# mimic.drop(columns = ['subject_id'], inplace = True)
# mimic.drop(columns = mimic_target_columns, inplace = True)
# X_full = mimic.astype('float')
# # print(X_full[['age', 'gender']])
# categorical_columns = ['gender']
# numerical_columns = list(X_full.columns[X_full.columns != 'gender'])
# X_full.loc[X_full['gender'] == 1, 'gender'] = 'male'
# X_full.loc[X_full['gender'] == 0, 'gender'] = 'female'