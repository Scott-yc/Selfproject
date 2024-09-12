# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, LabelEncoder
#
# # 加载数据
# df = pd.read_csv('83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv')
#
# # 2.1 数据探索
# # 基本统计描述
# print(df.describe())
#
# # 数据类型概览
# print(df.dtypes)
#
# # 计算每列的缺失值数量
# print(df.isnull().sum())
# sns.heatmap(df.isnull(), cbar=False)
# plt.show()
#
# # 箱线图检查异常值
# for column in df.select_dtypes(include=['float', 'int']).columns:
#     plt.figure()
#     sns.boxplot(x=df[column])
#     plt.title(column)
#     plt.show()
#
# # 2.2 数据清洗
# # 使用零值数量识别并删除异常值
# zero_threshold = 1
# # 计算每一行的零值数量
# zero_counts = (df == 0).sum(axis=1)
# # 只保留零值数量小于阈值的行
# df = df[zero_counts < zero_threshold]
#
# # 删除年份等无关特征
# df.drop(['Latitude', 'Longitude', 'Year_start', 'Year_end', 'Temperature', 'Vial', 'Replicate'], axis=1, inplace=True)
#
# # 检查洗完结果
# print(df.describe())
#
# # 2.3 特征工程
# # 对'Species'和'Population'进行One-Hot编码
# df = pd.get_dummies(df, columns=['Species', 'Population'])
#
# # 标签编码
# le = LabelEncoder()
# df['Sex'] = le.fit_transform(df['Sex'])
#
# # 选择连续变量列
# continuous_columns = ['Thorax_length', 'l2', 'l3p', 'l3d', 'lpd', 'l3', 'w1', 'w2', 'w3', 'wing_loading']
#
# # 检查连续变量列的数据类型，并将其转换为数值类型
# df[continuous_columns] = df[continuous_columns].apply(pd.to_numeric, errors='coerce')
#
# # 删除包含NaN值的行
# df = df.dropna(subset=continuous_columns)
#
# # 标准化连续变量
# scaler = StandardScaler()
# df[continuous_columns] = scaler.fit_transform(df[continuous_columns])
#
# # 设置显示的最大行数
# pd.set_option('display.max_rows', None)
#
# # 显示处理后的数据集
# print(df.info())
# print(df.head())
# print(df)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, f1_score

# 加载数据
df = pd.read_csv('83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv')

# 数据探索
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())

sns.heatmap(df.isnull(), cbar=False)
plt.show()

for column in df.select_dtypes(include=['float', 'int']).columns:
    plt.figure()
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.show()

# 数据清洗
zero_threshold = 1
zero_counts = (df == 0).sum(axis=1)
df = df[zero_counts < zero_threshold]
df.drop(['Latitude', 'Longitude', 'Year_start', 'Year_end', 'Temperature', 'Vial', 'Replicate'], axis=1, inplace=True)
print(df.describe())

# 特征工程
le_species = LabelEncoder()
df['Species'] = le_species.fit_transform(df['Species'])
le_population = LabelEncoder()
df['Population'] = le_population.fit_transform(df['Population'])
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
print(dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_))))

continuous_columns = ['Thorax_length', 'l2', 'l3p', 'l3d', 'lpd', 'l3', 'w1', 'w2', 'w3', 'wing_loading']
df[continuous_columns] = df[continuous_columns].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=continuous_columns)


# 标准化前的数据分布
plt.figure(figsize=(14, 8))
colors = sns.color_palette("hsv", len(continuous_columns))
for i, column in enumerate(continuous_columns):
    sns.histplot(df[column], kde=True, color=colors[i], label=column, alpha=0.5)
plt.title('Distribution of Continuous Features Before Standardization')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()



scaler = StandardScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# 标准化后的数据分布
plt.figure(figsize=(14, 8))
for i, column in enumerate(continuous_columns):
    sns.histplot(df[column], kde=True, color=colors[i], label=column, alpha=0.5)
plt.title('Distribution of Continuous Features After Standardization')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()


print(df.shape)
print(df.info())
print(df.head())
print(df)

# 数据集分割
X = df.drop('Sex', axis=1)
y = df['Sex']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)

# GridSearchCV
param_grid = [
    {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']},
    {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']},
    {'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['saga'], 'l1_ratio': [0.1, 0.5, 0.7]}
]
log_reg = LogisticRegression(max_iter=10000)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_}")
# transfer CV to DataFrame
results = pd.DataFrame(grid_search.cv_results_)

solvers = results['param_solver'].unique()
# heat map
for solver in solvers:
    # filter
    solver_data = results[results['param_solver'] == solver]
    if 'param_penalty' in solver_data:
        pivot_table = solver_data.pivot_table(values='mean_test_score', index='param_C', columns='param_penalty')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f")
        plt.title(f'Grid Search Scores with solver={solver}')
        plt.xlabel('Penalty')
        plt.ylabel('C value')
        plt.show()

# best parameters to do model training
best_log_reg = grid_search.best_estimator_
best_log_reg.fit(X_train, y_train)

# calculate probability
y_prob = best_log_reg.predict_proba(X_test)[:, 1]

# define threshold
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']  # different color

# evaluate performance
f1_scores = []
plt.figure(figsize=(10, 8))
for i, threshold in enumerate(thresholds):
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    f1_scores.append(f1)
    print(f"Threshold: {threshold}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("___________________________")

    # print Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.plot(recall, precision, marker='.', color=colors[i], label=f'Threshold: {threshold}')

plt.title('Precision-Recall Curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# paint F1-Score
plt.figure(figsize=(10, 8))
for i, threshold in enumerate(thresholds):
    plt.plot(thresholds, f1_scores, marker='o', color=colors[i], label=f'Threshold: {threshold}')

plt.title('F1-Score vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.legend()
plt.show()
# output F1-Score highhest threshold
best_threshold = thresholds[f1_scores.index(max(f1_scores))]
print(f"Best Threshold based on F1-Score: {best_threshold}")
# paint ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, marker='.', color='b')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
# paint whole Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(recall, precision, marker='.', color='r')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


