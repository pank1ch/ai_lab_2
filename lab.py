import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

import sys
sys.stdout.reconfigure(encoding="utf-8")

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

transactions = df.applymap(lambda x: str(x).strip() if pd.notna(x) else "").values.tolist()
transactions = [[item for item in row if item != ""] for row in transactions]

print("Транзакций:", len(transactions))

# длина транзакций
transaction_lengths = [len(t) for t in transactions]

plt.hist(transaction_lengths, bins=range(1, max(transaction_lengths)+2))
plt.xlabel("Длина транзакции")
plt.ylabel("Частота")
plt.title("Распределение длин транзакций")
plt.show()

# === 3. Получаем список уникальных товаров ===
unique_items = sorted({item for trx in transactions for item in trx})
print("Уникальных товаров:", len(unique_items))
# print('Список уникальных товаров:')
# for item in unique_items:
#     print(item)


# one-hot encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
data = pd.DataFrame(te_array, columns=te.columns_)

# FPGrowth 
fpg = fpgrowth(data, min_support=0.02, use_colnames=True)
print("Частые наборы (FPG):")
pd.set_option('display.max_rows', None) 
print(fpg)


rules_fpg = association_rules(fpg, metric="confidence", min_threshold=0.25)

rules_fpg_filtered = rules_fpg[[
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift"
]]

print("Правила (FPG):")
print(rules_fpg_filtered)


# APRIORI 
apriori_df = apriori(data, min_support=0.02, use_colnames=True)
print("Частые наборы (Apriori):")
pd.set_option('display.max_rows', None)
print(apriori_df)

apriori_rules = association_rules(apriori_df, metric="confidence", min_threshold=0.25)


apriori_rules_filtered = apriori_rules[[
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift"
]]

print("Правила (Apriori):")
print(apriori_rules_filtered)


#  мин support для наборов длины 1, 2, 3 

print("\n=== Определение минимальной поддержки для наборов разной длины ===")

support_values = [x / 100 for x in range(40, 0, -1)]  # 0.20 -> 0.01

def find_min_support_for_k(k):
    for s in support_values:
        freq = apriori(data, min_support=s, use_colnames=True)
        if freq['itemsets'].apply(lambda x: len(x) == k).any():
            print(f"Минимальная поддержка для наборов длины {k}: {s}")
            return s
    print(f"Наборы длины {k} не найдены ни при одной поддержке.")
    return None

find_min_support_for_k(1)
find_min_support_for_k(2)
find_min_support_for_k(3)



# Топ-10 самых популярных товаров
fpg['itemsets_str'] = fpg['itemsets'].apply(lambda x: ', '.join(list(x)))
top_products = fpg.sort_values(by='support', ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x='support', y='itemsets_str', data=top_products)
plt.title("Топ-10 самых популярных товаров")
plt.tight_layout()
plt.show()

rules_fpg['antecedents_str'] = rules_fpg['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_fpg['consequents_str'] = rules_fpg['consequents'].apply(lambda x: ', '.join(list(x)))

# граф достоверности
plt.bar(range(len(rules_fpg)), rules_fpg['confidence'])
plt.xticks(range(len(rules_fpg)), 
           [f"{a} → {c}" for a, c in zip(rules_fpg['antecedents_str'], rules_fpg['consequents_str'])],
           rotation=90)
plt.ylabel("Достоверность (confidence)")
plt.title("Граф достоверности правил (FPGrowth)")
plt.tight_layout()
plt.show()

# визуализация ассоциативных правил 
G = nx.DiGraph()

for _, row in rules_fpg.iterrows():
    a = row['antecedents_str']
    c = row['consequents_str']
    conf = row['confidence']
    G.add_edge(a, c, weight=round(conf, 3))

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=1.3)

nx.draw(G, pos, with_labels=True, node_size=2200, node_color="lightgreen", font_size=8)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Граф ассоциативных правил (FPGrowth)")
plt.tight_layout()
plt.show()


#Heatmap lift-метрики

heatmap_data = rules_fpg_filtered.pivot_table(
    index=rules_fpg_filtered["antecedents"].apply(lambda x: ', '.join(list(x))),
    columns=rules_fpg_filtered["consequents"].apply(lambda x: ', '.join(list(x))),
    values="lift",
    fill_value=0
)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=False, cmap="YlGnBu")
plt.title("Тепловая карта lift-метрики ассоциативных правил (FPGrowth)")
plt.xlabel("Consequent")
plt.ylabel("Antecedent")
plt.tight_layout()
plt.show()