import pandas as pd
import statsmodels.api as sm
import numpy as np

df = pd.read_csv("dfc.csv")

df['target'] = ((df['review_emotion'] == 0) & (df['review_theme'] == 'тарифы и условия')).astype(int)

categorical_features = [
    'company', 'review_source', 'business_line', 'product', 
    'solution_flg', 'gender_cd', 'education_level_cd', 
    'marital_status_cd', 'citizenship_country', 'segment_name', 'age_segment'
]
numerical_features = [
    'children_cnt', 'new_flg', 'influencer_flg', 
    'subscription_important_flg', 'is_profitable'
]

for col in categorical_features:
    df[col] = df[col].fillna('Unknown')

for col in numerical_features:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

df['is_profitable'] = df['is_profitable'].astype(int)
    
X_categorical = pd.get_dummies(df[categorical_features], drop_first=True, dtype=int)

X = pd.concat([df[numerical_features], X_categorical], axis=1)
y = df['target']

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] == 1)]
X = X.drop(to_drop, axis=1)
print(f"Удалено {len(to_drop)} идеально коллинеарных признаков: {to_drop}")


X_const = sm.add_constant(X)

logit_model = sm.Logit(y, X_const.astype(float))
try:
    result = logit_model.fit()
except np.linalg.LinAlgError:
    print("Стандартная модель не сошлась, пробую регуляризацию...")
    result = logit_model.fit_regularized(method='l1')


print("--- Logistic Regression Summary ---")
print(result.summary())
