import pandas as pd
import statsmodels.api as sm
import numpy as np

# Загрузка данных
df = pd.read_csv("dfc.csv")

# 1. Создание целевой переменной
df['target'] = ((df['review_emotion'] == 0) & (df['review_theme'] == 'тарифы и условия')).astype(int)

# 2. Определение признаков
categorical_features = [
    'company', 'review_source', 'business_line', 'product', 
    'solution_flg', 'gender_cd', 'education_level_cd', 
    'marital_status_cd', 'citizenship_country', 'segment_name', 'age_segment'
]
numerical_features = [
    'children_cnt', 'new_flg', 'influencer_flg', 
    'subscription_important_flg', 'is_profitable'
]

# 3. Обработка пропусков
for col in categorical_features:
    df[col] = df[col].fillna('Unknown')

for col in numerical_features:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Преобразование is_profitable в int
df['is_profitable'] = df['is_profitable'].astype(int)
    
# 4. One-Hot Encoding
X_categorical = pd.get_dummies(df[categorical_features], drop_first=True, dtype=int)

# 5. Сборка итогового набора данных
X = pd.concat([df[numerical_features], X_categorical], axis=1)
y = df['target']

# 6. Удаление идеально коллинеарных признаков
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] == 1)]
X = X.drop(to_drop, axis=1)
print(f"Удалено {len(to_drop)} идеально коллинеарных признаков: {to_drop}")


# Построение и обучение модели
X_const = sm.add_constant(X)

# Попробуем fit_regularized, чтобы избежать проблем с разделением
logit_model = sm.Logit(y, X_const.astype(float))
try:
    # Сначала стандартный fit
    result = logit_model.fit()
except np.linalg.LinAlgError:
    print("Стандартная модель не сошлась, пробую регуляризацию...")
    # Если не сошлось, используем регуляризацию
    result = logit_model.fit_regularized(method='l1')


# Вывод результатов
print("--- Logistic Regression Summary ---")
print(result.summary())
