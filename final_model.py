import pandas as pd
import statsmodels.api as sm
import numpy as np

# Загрузка данных
df = pd.read_csv("dfc.csv")

# 1. Создание целевой переменной
df['target'] = ((df['review_emotion'] == 0) & (df['review_theme'] == 'тарифы и условия')).astype(int)

# 2. Определение признаков
categorical_features = [
    'product', 'solution_flg', 'gender_cd', 'age_segment' # age_segment для разбивки
]
final_numerical_features = [
    'new_flg', 'influencer_flg'
]

# 3. Обработка пропусков
for col in categorical_features:
    df[col] = df[col].fillna('Unknown')

for col in final_numerical_features:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# 4. One-Hot Encoding и выбор итоговых признаков
df_dummies = pd.get_dummies(df[categorical_features], drop_first=True, dtype=int)

# Собираем итоговый набор признаков
final_features_list = [
    'new_flg', 'influencer_flg',
    'product_Автокредиты', 'product_Банкоматы', 'product_Бизнес-продукты',
    'product_Дебетовые карты', 'product_Инвестиции и брокерские счета',
    'product_Ипотека и ипотечное рефинансирование', 'product_Не определено',
    'product_Подписки', 'product_Потребительские кредиты',
    'product_Премиальные продукты', 'product_Проблемная задолженность',
    'solution_flg_проблема решена', 'gender_cd_M'
]

X_final_dummies = pd.get_dummies(df[['product', 'solution_flg', 'gender_cd']], drop_first=True, dtype=int)
X_final = pd.concat([df[final_numerical_features], X_final_dummies], axis=1)

# Убедимся, что все колонки из списка есть в датафрейме, иначе добавим их с нулями
for col in final_features_list:
    if col not in X_final.columns:
        if col in df_dummies.columns: # Проверяем, есть ли колонка в исходных dummies
             X_final[col] = df_dummies[col]
        else:
             X_final[col] = 0 # Если ее не было даже в dummies (из-за фильтрации), то это 0


X_final = X_final[final_features_list]
y = df['target']


# --- Функция для обучения и вывода результатов ---
def run_logit(X_data, y_data):
    if len(y_data.unique()) < 2:
        return None, None, f"В подгруппе только один класс, модель не может быть обучена."

    X_const = sm.add_constant(X_data)
    logit_model = sm.Logit(y_data, X_const.astype(float))
    
    try:
        result = logit_model.fit(disp=0)
        p_value = result.pvalues.get('influencer_flg', 'N/A')
        coeff = result.params.get('influencer_flg', 'N/A')
        return coeff, p_value, None
    except Exception as e:
        return None, None, str(e)


# --- Проверка на устойчивость по age_segment ---
print("--- Проверка модели на устойчивость по сегментам возраста ---")
age_segments = df['age_segment'].unique()

for segment in age_segments:
    print(f"\nАнализ для сегмента: '{segment}'")
    segment_indices = df[df['age_segment'] == segment].index
    X_segment = X_final.loc[segment_indices]
    y_segment = y.loc[segment_indices]
    
    coeff, p_value, error = run_logit(X_segment, y_segment)
    
    if error:
        print(f"  Ошибка: {error}")
        continue

    print(f"  Коэффициент для 'influencer_flg': {coeff:.4f}")
    print(f"  P-value для 'influencer_flg': {p_value:.4f}")

    if p_value < 0.05:
        print("  Вывод: Коэффициент значим.")
    else:
        print("  Вывод: Коэффициент не значим.")

# --- Обучение итоговой модели на всех данных ---
print("\n\n--- Результаты итоговой модели на всех данных ---")
X_const = sm.add_constant(X_final)
final_logit_model = sm.Logit(y, X_const.astype(float))
final_result = final_logit_model.fit()
print(final_result.summary())