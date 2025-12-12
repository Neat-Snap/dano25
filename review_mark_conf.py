import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

df = pd.read_csv('cleaned.csv')
df['review_mark_numeric'] = pd.to_numeric(df['review_mark_numeric'], errors='coerce')
df = df.dropna(subset=['review_mark_numeric', 'review_emotion'])

y_true = (df['review_emotion'] == 0).astype(int)

thresholds = [1, 2, 3, 4, 5]
precisions = []
recalls = []
f1s = []
accuracies = []

for t in thresholds:
    y_pred = (df['review_mark_numeric'] <= t).astype(int)
    
    precisions.append(precision_score(y_true, y_pred, zero_division=0))
    recalls.append(recall_score(y_true, y_pred))
    f1s.append(f1_score(y_true, y_pred))

plt.figure(figsize=(10, 6))

plt.plot(thresholds, precisions, marker='s', label='Precision (Точность)', linestyle='--', color='blue', alpha=0.7)
plt.plot(thresholds, recalls, marker='^', label='Recall (Полнота)', linestyle='--', color='green', alpha=0.7)
plt.plot(thresholds, f1s, marker='o', label='F1-Score (Баланс)', linewidth=3, color='red')

plt.title('Как работает F1-Score: Поиск баланса', fontsize=14)
plt.xlabel('Порог оценки (Всё, что <= X, считаем негативом)', fontsize=12)
plt.ylabel('Значение метрики', fontsize=12)
plt.xticks(thresholds)
plt.grid(True, alpha=0.5)
plt.legend()

plt.annotate('Много пропускаем\n(Низкий Recall)', xy=(1, 0.45), xytext=(1.2, 0.3), arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.annotate('Много лишнего\n(Низкая Precision)', xy=(5, 0.4), xytext=(3.5, 0.5), arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.annotate('Идеальный баланс', xy=(3, 1.0), xytext=(3, 0.8), arrowprops=dict(facecolor='red', arrowstyle='->'))

plt.tight_layout()
plt.savefig('f1_explanation.png')
plt.show()