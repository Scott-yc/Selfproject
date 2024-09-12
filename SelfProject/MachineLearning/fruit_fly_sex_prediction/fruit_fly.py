
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, f1_score

# # 加载数据
# df = pd.read_csv('83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv')

# # 数据探索
# print(df.describe())
# print(df.dtypes)
# print(df.isnull().sum())

# sns.heatmap(df.isnull(), cbar=False)
# plt.show()

# for column in df.select_dtypes(include=['float', 'int']).columns:
#     plt.figure()
#     sns.boxplot(x=df[column])
#     plt.title(column)
#     plt.show()

# # 数据清洗
# zero_threshold = 1
# zero_counts = (df == 0).sum(axis=1)
# df = df[zero_counts < zero_threshold]
# df.drop(['Latitude', 'Longitude', 'Year_start', 'Year_end', 'Temperature', 'Vial', 'Replicate'], axis=1, inplace=True)
# print(df.describe())

# # 特征工程
# le_species = LabelEncoder()
# df['Species'] = le_species.fit_transform(df['Species'])
# le_population = LabelEncoder()
# df['Population'] = le_population.fit_transform(df['Population'])
# le_sex = LabelEncoder()
# df['Sex'] = le_sex.fit_transform(df['Sex'])
# print(dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_))))

# continuous_columns = ['Thorax_length', 'l2', 'l3p', 'l3d', 'lpd', 'l3', 'w1', 'w2', 'w3', 'wing_loading']
# df[continuous_columns] = df[continuous_columns].apply(pd.to_numeric, errors='coerce')
# df = df.dropna(subset=continuous_columns)


# # 标准化前的数据分布
# plt.figure(figsize=(14, 8))
# colors = sns.color_palette("hsv", len(continuous_columns))
# for i, column in enumerate(continuous_columns):
#     sns.histplot(df[column], kde=True, color=colors[i], label=column, alpha=0.5)
# plt.title('Distribution of Continuous Features Before Standardization')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()



# scaler = StandardScaler()
# df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# # 标准化后的数据分布
# plt.figure(figsize=(14, 8))
# for i, column in enumerate(continuous_columns):
#     sns.histplot(df[column], kde=True, color=colors[i], label=column, alpha=0.5)
# plt.title('Distribution of Continuous Features After Standardization')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()


# print(df.shape)
# print(df.info())
# print(df.head())
# print(df)

# # 数据集分割
# X = df.drop('Sex', axis=1)
# y = df['Sex']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("Training set size:", X_train.shape, y_train.shape)
# print("Test set size:", X_test.shape, y_test.shape)

# # GridSearchCV
# param_grid = [
#     {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']},
#     {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']},
#     {'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['saga'], 'l1_ratio': [0.1, 0.5, 0.7]}
# ]
# log_reg = LogisticRegression(max_iter=10000)
# grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# print(f"Best parameters found: {grid_search.best_params_}")
# print(f"Best cross-validation accuracy: {grid_search.best_score_}")
# # transfer CV to DataFrame
# results = pd.DataFrame(grid_search.cv_results_)

# solvers = results['param_solver'].unique()
# # heat map
# for solver in solvers:
#     # filter
#     solver_data = results[results['param_solver'] == solver]
#     if 'param_penalty' in solver_data:
#         pivot_table = solver_data.pivot_table(values='mean_test_score', index='param_C', columns='param_penalty')
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f")
#         plt.title(f'Grid Search Scores with solver={solver}')
#         plt.xlabel('Penalty')
#         plt.ylabel('C value')
#         plt.show()

# # best parameters to do model training
# best_log_reg = grid_search.best_estimator_
# best_log_reg.fit(X_train, y_train)

# # calculate probability
# y_prob = best_log_reg.predict_proba(X_test)[:, 1]

# # define threshold
# thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']  # different color

# # evaluate performance
# f1_scores = []
# plt.figure(figsize=(10, 8))
# for i, threshold in enumerate(thresholds):
#     y_pred = (y_prob >= threshold).astype(int)
#     f1 = f1_score(y_test, y_pred, zero_division=1)
#     f1_scores.append(f1)
#     print(f"Threshold: {threshold}")
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred, zero_division=1))
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
#     print("___________________________")

#     # print Precision-Recall curve
#     precision, recall, _ = precision_recall_curve(y_test, y_pred)
#     plt.plot(recall, precision, marker='.', color=colors[i], label=f'Threshold: {threshold}')

# plt.title('Precision-Recall Curves')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend()
# plt.show()

# # paint F1-Score
# plt.figure(figsize=(10, 8))
# for i, threshold in enumerate(thresholds):
#     plt.plot(thresholds, f1_scores, marker='o', color=colors[i], label=f'Threshold: {threshold}')

# plt.title('F1-Score vs Threshold')
# plt.xlabel('Threshold')
# plt.ylabel('F1-Score')
# plt.legend()
# plt.show()
# # output F1-Score highhest threshold
# best_threshold = thresholds[f1_scores.index(max(f1_scores))]
# print(f"Best Threshold based on F1-Score: {best_threshold}")
# # paint ROC curve
# fpr, tpr, _ = roc_curve(y_test, y_prob)
# plt.figure()
# plt.plot(fpr, tpr, marker='.', color='b')
# plt.title('ROC Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.show()
# # paint whole Precision-Recall curve
# precision, recall, _ = precision_recall_curve(y_test, y_prob)
# plt.figure()
# plt.plot(recall, precision, marker='.', color='r')
# plt.title('Precision-Recall Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.show()






# 导入相关库
import pandas as pd  # 用于数据操作和分析的库
import seaborn as sns  # 用于数据可视化的库
import matplotlib.pyplot as plt  # 用于绘制图表的库
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 用于数据标准化和分类标签编码的工具
from sklearn.model_selection import train_test_split, GridSearchCV  # 用于数据集拆分和网格搜索的工具
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, f1_score  # 用于模型评估的度量工具

# 加载数据
df = pd.read_csv('83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv')  # 读取CSV文件到DataFrame中

# 数据探索
print(df.describe())  # 打印数值型数据的统计信息，如均值、方差、最大值、最小值等
print(df.dtypes)  # 打印每列的数据类型
print(df.isnull().sum())  # 打印每列中缺失值的数量

# 可视化数据缺失情况，使用热力图展示缺失值的位置
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# 对数值型列绘制箱线图，查看数据的分布以及是否有离群值
for column in df.select_dtypes(include=['float', 'int']).columns:
    plt.figure()
    sns.boxplot(x=df[column])  # 绘制箱线图
    plt.title(column)  # 设置图表标题为当前列名
    plt.show()

# 数据清洗
zero_threshold = 1  # 定义零值的阈值
zero_counts = (df == 0).sum(axis=1)  # 统计每行中零值的数量
df = df[zero_counts < zero_threshold]  # 删除包含过多零值的行
df.drop(['Latitude', 'Longitude', 'Year_start', 'Year_end', 'Temperature', 'Vial', 'Replicate'], axis=1, inplace=True)  # 删除无关列
print(df.describe())  # 再次打印数据清洗后的统计信息

# 特征工程
le_species = LabelEncoder()  # 初始化LabelEncoder对象
df['Species'] = le_species.fit_transform(df['Species'])  # 对Species列进行标签编码
le_population = LabelEncoder()  # 初始化LabelEncoder对象
df['Population'] = le_population.fit_transform(df['Population'])  # 对Population列进行标签编码
le_sex = LabelEncoder()  # 初始化LabelEncoder对象
df['Sex'] = le_sex.fit_transform(df['Sex'])  # 对Sex列进行标签编码
print(dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_))))  # 打印性别编码的映射关系

# 确保数值型列是正确的数值类型，如果转换失败则设置为NaN
continuous_columns = ['Thorax_length', 'l2', 'l3p', 'l3d', 'lpd', 'l3', 'w1', 'w2', 'w3', 'wing_loading']
df[continuous_columns] = df[continuous_columns].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=continuous_columns)  # 删除包含NaN值的行

# 标准化前的数据分布
plt.figure(figsize=(14, 8))  # 设置图表大小
colors = sns.color_palette("hsv", len(continuous_columns))  # 设置颜色
for i, column in enumerate(continuous_columns):
    sns.histplot(df[column], kde=True, color=colors[i], label=column, alpha=0.5)  # 绘制直方图
plt.title('标准化前连续特征的分布')  # 设置标题
plt.xlabel('数值')  # 设置X轴标签
plt.ylabel('频率')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.show()

# 数据标准化
scaler = StandardScaler()  # 初始化标准化器
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])  # 对连续型特征进行标准化

# 标准化后的数据分布
plt.figure(figsize=(14, 8))  # 设置图表大小
for i, column in enumerate(continuous_columns):
    sns.histplot(df[column], kde=True, color=colors[i], label=column, alpha=0.5)  # 绘制标准化后的直方图
plt.title('标准化后连续特征的分布')  # 设置标题
plt.xlabel('标准化数值')  # 设置X轴标签
plt.ylabel('频率')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.show()

# 打印数据集的维度、信息和前几行数据
print(df.shape)  # 打印数据的形状（行数和列数）
print(df.info())  # 打印数据的概览信息
print(df.head())  # 打印前几行数据
print(df)  # 打印完整的数据集

# 数据集分割
X = df.drop('Sex', axis=1)  # 特征矩阵，去掉目标列'Sex'
y = df['Sex']  # 目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 将数据集划分为训练集和测试集
print("训练集大小:", X_train.shape, y_train.shape)  # 打印训练集大小
print("测试集大小:", X_test.shape, y_test.shape)  # 打印测试集大小

# GridSearchCV
param_grid = [  # 定义超参数的搜索空间
    {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']},  # L1正则化的参数
    {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']},  # L2正则化的参数
    {'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['saga'], 'l1_ratio': [0.1, 0.5, 0.7]}  # ElasticNet正则化的参数
]
log_reg = LogisticRegression(max_iter=10000)  # 初始化逻辑回归模型
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')  # 使用网格搜索进行超参数优化
grid_search.fit(X_train, y_train)  # 在训练集上训练模型

print(f"找到的最佳参数: {grid_search.best_params_}")  # 打印找到的最佳超参数
print(f"最佳交叉验证准确率: {grid_search.best_score_}")  # 打印最佳交叉验证得分

# 将网格搜索结果转换为DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# 获取所有solver参数的唯一值
solvers = results['param_solver'].unique()
# 绘制热力图展示网格搜索结果
for solver in solvers:
    solver_data = results[results['param_solver'] == solver]  # 筛选对应solver的结果
    if 'param_penalty' in solver_data:
        pivot_table = solver_data.pivot_table(values='mean_test_score', index='param_C', columns='param_penalty')  # 创建透视表
        plt.figure(figsize=(10, 8))  # 设置图表大小
        sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f")  # 绘制热力图
        plt.title(f'使用solver={solver}的网格搜索得分')  # 设置图表标题
        plt.xlabel('Penalty')  # 设置X轴标签
        plt.ylabel('C值')  # 设置Y轴标签
        plt.show()

# 使用最佳超参数训练模型
best_log_reg = grid_search.best_estimator_  # 获取最佳估计器
best_log_reg.fit(X_train, y_train)  # 在训练集上训练模型

# 计算测试集上的预测概率
y_prob = best_log_reg.predict_proba(X_test)[:, 1]  # 获取类别1的预测概率

# 定义不同的决策阈值
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 不同的阈值
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']  # 设置不同阈值对应的颜色

# 评估模型性能
f1_scores = []  # 用于存储F1得分
plt.figure(figsize=(10, 8))  # 设置图表大小
for i, threshold in enumerate(thresholds):
    y_pred = (y_prob >= threshold).astype(int)  # 根据当前阈值生成预测结果
    f1 = f1_score(y_test, y_pred, zero_division=1)  # 计算F1得分
    f1_scores.append(f1)  # 将F1得分添加到列表
    print(f"阈值: {threshold}")
    print("分类报告:")
    print(classification_report(y_test, y_pred, zero_division=1))  # 打印分类报告
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))  # 打印混淆矩阵
    print("___________________________")

    # 绘制Precision-Recall曲线
    precision, recall, _ = precision_recall_curve(y_test, y_pred)  # 计算Precision-Recall值
    plt.plot(recall, precision, marker='.', color=colors[i], label=f'阈值: {threshold}')  # 绘制曲线

plt.title('Precision-Recall 曲线')  # 设置图表标题
plt.xlabel('召回率')  # 设置X轴标签
plt.ylabel('精度')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.show()

# 绘制F1得分与阈值的关系
plt.figure(figsize=(10, 8))  # 设置图表大小
for i, threshold in enumerate(thresholds):
    plt.plot(thresholds, f1_scores, marker='o', color=colors[i], label=f'阈值: {threshold}')  # 绘制F1得分曲线

plt.title('F1得分与阈值的关系')  # 设置图表标题
plt.xlabel('阈值')  # 设置X轴标签
plt.ylabel('F1得分')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.show()

# 输出基于F1得分的最佳阈值
best_threshold = thresholds[f1_scores.index(max(f1_scores))]
print(f"基于F1得分的最佳阈值: {best_threshold}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)  # 计算假阳性率和真阳性率
plt.figure()
plt.plot(fpr, tpr, marker='.', color='b')  # 绘制ROC曲线
plt.title('ROC 曲线')  # 设置图表标题
plt.xlabel('假阳性率')  # 设置X轴标签
plt.ylabel('真阳性率')  # 设置Y轴标签
plt.show()

# 绘制Precision-Recall曲线（基于概率）
precision, recall, _ = precision_recall_curve(y_test, y_prob)  # 计算Precision-Recall曲线
plt.figure()
plt.plot(recall, precision, marker='.', color='r')  # 绘制Precision-Recall曲线
plt.title('Precision-Recall 曲线')  # 设置图表标题
plt.xlabel('召回率')  # 设置X轴标签
plt.ylabel('精度')  # 设置Y轴标签
plt.show()


