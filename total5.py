# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:05:49 2024

@author: gaoli
"""
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

# Step 1: Importing dataset
# 读取数据集，指定编码格式（根据您的CSV文件实际编码调整）
dataset = pd.read_csv('DM6.csv', encoding='gbk')  # 如果'gbk'不适用，尝试'utf-8'或'gb18030'

# 特征列在1到38列之间（索引从1到38），目标变量在第0列
X = dataset.iloc[:, 1:39]
Y = dataset.iloc[:, 0]
print("Step 2: Importing dataset")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# 将X和Y合并为一个DataFrame以方便处理
data = pd.concat([Y, X], axis=1)
data.columns = ['y'] + list(X.columns)

# 映射 'sex' 列的数值代码到类别标签
sex_mapping = {0: 'Female', 1: 'Male'}
if 'sex' in data.columns:
    data['sex'] = data['sex'].map(sex_mapping)
else:
    print("数据中未找到'sex'列，请检查数据。")

# 映射 'ethnic' 列的数值代码到类别标签
ethnic_mapping = {
    1001: 'White',
    2001: 'White',
    3001: 'White',
    4001: 'White',
    1002: 'Mixed',
    2002: 'Mixed',
    3002: 'Mixed',
    4002: 'Mixed',
    1003: 'Asian or Asian British',
    2003: 'Asian or Asian British',
    4003: 'Asian or Asian British',
    2004: 'Black or Black British',
    3004: 'Black or Black British',
    5: 'Chinese',
    6: 'Other Ethnic Group',
    -1: 'Do Not Know',
    -3: 'Prefer Not To Answer'
}
if 'ethnic' in data.columns:
    data['ethnic'] = data['ethnic'].map(ethnic_mapping)
else:
    print("数据中未找到'ethnic'列，请检查数据。")

# 检查'time_0'列是否存在
if 'time_0' in data.columns:
    # 计算平均患病天数
    avg_disease_days = data['time_0'].mean()
    print(f"平均患病天数：{avg_disease_days:.2f}天")
else:
    print("数据中未找到'time_0'列，请检查数据。")

# 按照目标变量y分组
# 首先查看y的唯一值
print("目标变量y的唯一值：", data['y'].unique())

# 根据y的取值进行分组（请根据实际值调整）
if set(data['y'].unique()) == {'T2DM', 'NON-T2DM'}:
    group1 = data[data['y'] == 'T2DM']
    group2 = data[data['y'] == 'NON-T2DM']
elif set(data['y'].unique()) == {0, 1}:
    group1 = data[data['y'] == 1]  # 假设1表示T2DM
    group2 = data[data['y'] == 0]  # 假设0表示NON-T2DM
else:
    print("无法识别的y值，请检查目标变量。")
    # 如果无法识别，假设按照y的第一个值和第二个值进行分组
    unique_values = data['y'].unique()
    group1 = data[data['y'] == unique_values[0]]
    group2 = data[data['y'] == unique_values[1]]

# 识别连续变量和分类变量
# 连续变量：数值类型
continuous_vars = data.select_dtypes(include=[np.number]).columns.tolist()
if 'y' in continuous_vars:
    continuous_vars.remove('y')  # 去除目标变量
# 分类变量：对象类型或类别类型，或者指定的分类变量
categorical_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()
if 'y' in categorical_vars:
    categorical_vars.remove('y')  # 去除目标变量

# 如果 'sex' 和 'ethnic' 不在 categorical_vars 中，添加它们
if 'sex' not in categorical_vars and 'sex' in data.columns:
    categorical_vars.append('sex')
if 'ethnic' not in categorical_vars and 'ethnic' in data.columns:
    categorical_vars.append('ethnic')

print("连续变量：", continuous_vars)
print("分类变量：", categorical_vars)

# 初始化基线特征表
baseline_table = []

# 对连续变量进行基线特征分析
for var in continuous_vars:
    # 检查是否存在异常值或缺失值
    group1_var = group1[var].dropna()
    group2_var = group2[var].dropna()
    if group1_var.empty or group2_var.empty:
        continue

    # 检查变量的分布，决定是否使用参数检验或非参数检验
    # 这里假设使用非参数检验Mann-Whitney U检验
    median1 = group1_var.median()
    q1_1 = group1_var.quantile(0.25)
    q3_1 = group1_var.quantile(0.75)

    median2 = group2_var.median()
    q1_2 = group2_var.quantile(0.25)
    q3_2 = group2_var.quantile(0.75)

    # 使用Mann-Whitney U检验
    stat, p_value = stats.mannwhitneyu(group1_var, group2_var, alternative='two-sided')

    # 添加到基线特征表
    baseline_table.append({
        'Variable': var,
        'Group1 (Median [IQR])': f"{median1:.2f} [{q1_1:.2f}-{q3_1:.2f}]",
        'Group2 (Median [IQR])': f"{median2:.2f} [{q1_2:.2f}-{q3_2:.2f}]",
        'p-value': p_value
    })

# 对分类变量进行基线特征分析
for var in categorical_vars:
    # 检查是否存在异常值或缺失值
    group1_var = group1[var].fillna('Missing')
    group2_var = group2[var].fillna('Missing')

    # 计算每组的频数和百分比
    count1 = group1_var.value_counts(dropna=False)
    percent1 = group1_var.value_counts(normalize=True, dropna=False) * 100

    count2 = group2_var.value_counts(dropna=False)
    percent2 = group2_var.value_counts(normalize=True, dropna=False) * 100

    # 创建列联表
    contingency_table = pd.crosstab(data[var].fillna('Missing'), data['y'])

    # 根据列联表大小选择检验方法
    if contingency_table.shape == (2, 2):
        # Fisher精确检验
        _, p_value = stats.fisher_exact(contingency_table)
    else:
        # 卡方检验
        _, p_value, _, _ = stats.chi2_contingency(contingency_table)

    # 整理结果
    levels = contingency_table.index.tolist()
    for i, level in enumerate(levels):
        count1_level = count1.get(level, 0)
        percent1_level = percent1.get(level, 0)
        count2_level = count2.get(level, 0)
        percent2_level = percent2.get(level, 0)
        baseline_table.append({
            'Variable': f"{var} ({level})",
            'Group1 (Count %)': f"{int(count1_level)} ({percent1_level:.1f}%)",
            'Group2 (Count %)': f"{int(count2_level)} ({percent2_level:.1f}%)",
            'p-value': p_value if i == 0 else ''
        })

# 将基线特征表转换为DataFrame
baseline_df = pd.DataFrame(baseline_table)

# 格式化p值
def format_p_value(p):
    if p == '':
        return ''
    elif p < 0.01:
        return "<0.01"
    else:
        return f"{p:.3f}"

baseline_df['p-value'] = baseline_df['p-value'].apply(format_p_value)

# 根据实际存在的列来调整列顺序
columns_order = ['Variable', 'p-value']  # 必须的列

if 'Group1 (Median [IQR])' in baseline_df.columns and 'Group2 (Median [IQR])' in baseline_df.columns:
    columns_order.insert(1, 'Group1 (Median [IQR])')
    columns_order.insert(2, 'Group2 (Median [IQR])')

if 'Group1 (Count %)' in baseline_df.columns and 'Group2 (Count %)' in baseline_df.columns:
    columns_order.extend(['Group1 (Count %)', 'Group2 (Count %)'])

baseline_df = baseline_df[columns_order]

# 填充NaN值为空字符串
baseline_df.fillna('', inplace=True)

# 输出基线特征表
print("\n基线特征表：")
print(baseline_df.to_string(index=False))

# 可视化输出为图片
# 如果需要将表格保存为图片，可以使用matplotlib和seaborn
plt.figure(figsize=(12, len(baseline_df)*0.5))
plt.axis('off')
table = plt.table(cellText=baseline_df.values, colLabels=baseline_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('Baseline Characteristics', fontsize=14)
plt.savefig('baseline_characteristics.png', bbox_inches='tight')
plt.show()

# 导出为Excel文件
baseline_df.to_excel('baseline_characteristics.xlsx', index=False)

# 或者使用tabulate库打印美观的表格
from tabulate import tabulate
print(tabulate(baseline_df, headers='keys', tablefmt='grid'))



# -------------------- 数据预处理和特征选择部分 -------------------- #

# 为了数据预处理，将分类变量编码为数值
X_preprocessed = data.drop('y', axis=1).copy()  # 使用处理过的 data，确保包含 'sex' 和 'ethnic'

# 编码 'sex' 列
if 'sex' in X_preprocessed.columns:
    X_preprocessed['sex'] = X_preprocessed['sex'].map({'Female': 0, 'Male': 1})
else:
    print("X_preprocessed 中未找到'sex'列，请检查数据。")

# 编码 'ethnic' 列
if 'ethnic' in X_preprocessed.columns:
    le_ethnic = LabelEncoder()
    X_preprocessed['ethnic'] = le_ethnic.fit_transform(X_preprocessed['ethnic'].astype(str))
else:
    print("X_preprocessed 中未找到'ethnic'列，请检查数据。")

# 将X转换为NumPy数组
X_array = X_preprocessed.values

# 处理无限值和缺失值
X_array = np.where(np.isinf(X_array), np.nan, X_array)  # 将无限值替换为NaN
max_value = 1e10  # 设置一个合理的最大值
X_array = np.where(X_array > max_value, max_value, X_array)  # 将大于该值的异常数值设置为最大值

# 中位数填充（适用于数值特征）
imputer_median = SimpleImputer(missing_values=np.nan, strategy='median')
X_array[:, :] = imputer_median.fit_transform(X_array)  # 填充所有特征的缺失值

print("---------------------")
print("Step 3: Handling the missing data")
print("X after imputation")
print(X_array)

# 将目标变量编码为数值（如果是分类变量）
if Y.dtype == 'object':
    le_target = LabelEncoder()
    Y = le_target.fit_transform(Y)

# 第一步：使用SelectKBest进行特征选择
selector = SelectKBest(f_classif, k=25)  # 选择25个最好的特征
X_selected = selector.fit_transform(X_array, Y)
selected_features_indices = selector.get_support(indices=True)
print("Selected features (indices):", selected_features_indices)

# 获取特征名称
feature_names = X_preprocessed.columns
selected_feature_names = feature_names[selected_features_indices]
print("Selected feature names:", selected_feature_names)

# 获取所有特征的F-values和p-values
f_values = selector.scores_
p_values = selector.pvalues_

# 输出每个被选中特征的F-value和p-value
print("\nSelected Features with F-values and p-values:")
for i in range(len(selected_features_indices)):
    index = selected_features_indices[i]
    print(f"Feature: {feature_names[index]} (Index {index}) - F-value: {f_values[index]:.2f}, P-value: {p_values[index]:.4f}")

# 创建一个DataFrame来存储被选中特征及其F-value和p-value
selected_features_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'F-value': f_values[selected_features_indices],
    'p-value': p_values[selected_features_indices]
})

# 按照F-value降序排序
selected_features_df = selected_features_df.sort_values(by='F-value', ascending=False)

# 绘制被选中特征的F-value柱状图
plt.figure(figsize=(10, 8))
colors = ['#00aaff'] * len(selected_features_df)  # Use light blue color
sns.barplot(x='F-value', y='Feature', data=selected_features_df, palette=colors)
plt.xlabel('F-value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()


# 提取初步选择的特征
X_initial_selected = X_array[:, selected_features_indices]
initial_selected_feature_names = feature_names[selected_features_indices]

# 第二步：Spearman相关分析
# 计算初步选择特征的Spearman相关矩阵
spearman_corr_matrix = pd.DataFrame(X_initial_selected, columns=initial_selected_feature_names).corr(method='spearman')

# 绘制Spearman相关性热图
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Initial Selected Features Spearman Correlation Heatmap")
plt.show()

# 第三步：处理高度相关的特征
# 设置相关性阈值
corr_threshold = 0.6

# 将相关矩阵转换为上三角矩阵，避免重复计算
upper_tri = spearman_corr_matrix.where(np.triu(np.ones(spearman_corr_matrix.shape), k=1).astype(bool))

# 找到高度相关的特征对
high_corr_pairs = [(row, column) for row in upper_tri.index for column in upper_tri.columns if abs(upper_tri.loc[row, column]) > corr_threshold]

print("Highly correlated feature pairs (|corr| > 0.6):")
print(high_corr_pairs)

# 创建要删除的特征列表
features_to_remove = set()

# 手动指定的特征列表'CLR', 'PLR', 'CALLY', 'Tyg', 'CAR', 'NPR'
manual_features = ['WBC', 'PLR', 'CRP', 'CAR', 'Tyg','CALLY']

for pair in high_corr_pairs:
    feature1, feature2 = pair
    # 如果其中一个特征在手动指定的特征列表中，则保留手动指定的特征，删除另一个
    if feature1 in manual_features:
        features_to_remove.add(feature2)
    elif feature2 in manual_features:
        features_to_remove.add(feature1)
    else:
        # 如果都不在手动指定的特征列表中，可以根据某种策略选择删除哪个
        # 例如，可以删除方差较小的特征，或者随机删除一个
        features_to_remove.add(feature2)

# 从初步选择的特征中删除高度相关的特征
final_features = [feat for feat in initial_selected_feature_names if feat not in features_to_remove]
print("Features to remove due to high correlation:", features_to_remove)
print("Final selected features after removing highly correlated ones:", final_features)

# 获取最终特征的索引
final_feature_indices = [feature_names.get_loc(feat) for feat in final_features]

# 提取最终选择的特征数据
X_final_selected = X_array[:, final_feature_indices]
final_selected_feature_names = feature_names[final_feature_indices]

# 最后，如果需要，可以再次绘制 Spearman 相关性热图
spearman_corr_matrix_final = pd.DataFrame(X_final_selected, columns=final_selected_feature_names).corr(method='spearman')

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_matrix_final, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Final Selected Features Spearman Correlation Heatmap")
plt.show()

# 绘制最终选择特征的柱状图
plt.figure(figsize=(10, 6))
colors = ['#00aaff'] * len(final_selected_feature_names)  # 使用亮蓝色
sns.barplot(x='F-value', y='Feature', data=selected_features_df[selected_features_df['Feature'].isin(final_selected_feature_names)], palette=colors)
plt.xlabel('F-value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()


# 导入必要的库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 导入机器学习和数据处理库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve, accuracy_score, f1_score, 
                             brier_score_loss, recall_score, precision_score,
                             classification_report)
from imblearn.over_sampling import SMOTE
from collections import defaultdict

# 导入模型库
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, 
                              RandomForestClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 如果需要使用贝叶斯优化库
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


#最佳模型为logistic regression
# 数据划分和SMOTE处理
X_train, X_test, y_train, y_test = train_test_split(X_final_selected, Y, test_size=0.2, random_state=42)

# 应用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 特征缩放
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# 定义逻辑回归模型
model = LogisticRegression(random_state=42, max_iter=1000)

# 超参数调优
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

param_space = {
    'C': Real(0.01, 10.0, 'log-uniform'),
    'penalty': Categorical(['l2']),
    'solver': Categorical(['lbfgs', 'liblinear'])
}

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=30,  # 迭代次数
    scoring='roc_auc',  # 优化AUC
    cv=3,
    random_state=42,
    n_jobs=-1
)
bayes_search.fit(X_train_resampled, y_train_resampled)
best_model = bayes_search.best_estimator_
print("逻辑回归最佳参数:", bayes_search.best_params_)

# 在测试集上进行预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算指标
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
brier = brier_score_loss(y_test, y_pred_proba)

# 输出结果
print("\n逻辑回归模型评估结果:")
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Brier Score: {brier:.4f}")


#SHAP
import shap

# 从测试集中抽样
sample_size = 5000  # 根据需要调整样本大小
if X_test.shape[0] > sample_size:
    idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
    X_test_sample = X_test[idx]
    y_test_sample = y_test[idx]
else:
    X_test_sample = X_test
    y_test_sample = y_test

# 创建用于线性模型的 SHAP Explainer
# 注意，这里使用的是训练后的模型 best_model，以及训练集数据 X_train_resampled
explainer = shap.LinearExplainer(best_model, X_train_resampled, feature_perturbation="interventional")

# 计算 SHAP 值
shap_values = explainer.shap_values(X_test_sample)

# 获取特征名称
selected_feature_names = final_selected_feature_names

# 绘制 SHAP 总结图（调整 X 轴尺度）
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=selected_feature_names, show=False, plot_size=(12, 8))
plt.xlabel("SHAP value (impact on model output, scaled)", fontsize=14)
plt.xlim(-5, 5)
plt.tight_layout()
plt.show()

# 绘制 SHAP 条形图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=selected_feature_names, plot_type='bar', show=False, plot_size=(12, 8))
plt.xlabel("Mean SHAP Value (scaled)", fontsize=14)
plt.xlim(0, 2.0)
plt.tight_layout()
plt.show()

# 绘制单个样本的 SHAP Force Plot
sample_index =1   # 可以更改索引以查看不同样本
shap.force_plot(explainer.expected_value, shap_values[sample_index], X_test_sample[sample_index], 
                feature_names=selected_feature_names, matplotlib=True, show=True)













#剔除特征AST/TP/MCV/BUN/ALB/RDW/calcium建模
# -------------------- Spearman Correlation Analysis with Manual Feature Removal -------------------- #

# Step 1: 手动手动从初始选择的特性中删除指定的特性
features_to_remove_manually = ['AST', 'TP', 'MCV', 'BUN', 'ALB', 'RDW', 'calcium']

# 确保要保留的特征存在
manual_features_to_keep = ['WBC', 'PLR', 'CRP', 'CAR', 'Tyg', 'CALLY']

# 从 initial_selected_feature_names 中删除指定的特征
initial_selected_feature_names = [feat for feat in initial_selected_feature_names if feat not in features_to_remove_manually]

# 更新 X_initial_selected 
selected_feature_indices = [feature_names.get_loc(feat) for feat in initial_selected_feature_names]
X_initial_selected = X_array[:, selected_feature_indices]

# Spearman correlation analysis
spearman_corr_matrix = pd.DataFrame(X_initial_selected, columns=initial_selected_feature_names).corr(method='spearman')

# Plot Spearman correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Initial Selected Features Spearman Correlation Heatmap (After Manual Feature Removal)")
plt.show()

# Step 2: 处理高度相关的特征
corr_threshold = 0.6

# 将相关矩阵转换为上三角矩阵以避免重复
upper_tri = spearman_corr_matrix.where(np.triu(np.ones(spearman_corr_matrix.shape), k=1).astype(bool))

# 查找相关性高于阈值的特征对
high_corr_pairs = [(row, column) for row in upper_tri.index for column in upper_tri.columns if abs(upper_tri.loc[row, column]) > corr_threshold]

print("Highly correlated feature pairs (|corr| > 0.6):")
print(high_corr_pairs)

# 创建一个集以保存基于高相关性要移除的要素
features_to_remove = set()

# 确保保留要保留的手动功能
for pair in high_corr_pairs:
    feature1, feature2 = pair
    # 如果其中一个功能在手动保留列表中，请删除另一个功能
    if feature1 in manual_features_to_keep:
        features_to_remove.add(feature2)
    elif feature2 in manual_features_to_keep:
        features_to_remove.add(feature1)
    else:
        # 如果两个功能都不在手动保留列表中，请根据较低的差异或任意删除一个
        # remove feature2
        features_to_remove.add(feature2)

# Remove the highly correlated features
final_features = [feat for feat in initial_selected_feature_names if feat not in features_to_remove]
print("Features to remove due to high correlation:", features_to_remove)
print("Final selected features after removing highly correlated ones:", final_features)

# Get the indices of the final selected features
final_feature_indices = [feature_names.get_loc(feat) for feat in final_features]

# Extract the final selected feature data
X_final_selected = X_array[:, final_feature_indices]
final_selected_feature_names = feature_names[final_feature_indices]

# Plot the Spearman correlation heatmap for the final selected features
spearman_corr_matrix_final = pd.DataFrame(X_final_selected, columns=final_selected_feature_names).corr(method='spearman')

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_matrix_final, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Final Selected Features Spearman Correlation Heatmap")
plt.show()

# Plot the bar chart of the final selected features based on F-values
plt.figure(figsize=(10, 6))
colors = ['#00aaff'] * len(final_selected_feature_names)
sns.barplot(
    x='F-value',
    y='Feature',
    data=selected_features_df[selected_features_df['Feature'].isin(final_selected_feature_names)],
    palette=colors
)
plt.xlabel('F-value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()

# 数据划分和SMOTE处理
X_train, X_test, y_train, y_test = train_test_split(X_final_selected, Y, test_size=0.2, random_state=42)

# 应用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 特征缩放
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# 定义逻辑回归模型
model = LogisticRegression(random_state=42, max_iter=1000)

# 超参数调优
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

param_space = {
    'C': Real(0.01, 10.0, 'log-uniform'),
    'penalty': Categorical(['l2']),
    'solver': Categorical(['lbfgs', 'liblinear'])
}

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=30,  # 迭代次数
    scoring='roc_auc',  # 优化AUC
    cv=3,
    random_state=42,
    n_jobs=-1
)
bayes_search.fit(X_train_resampled, y_train_resampled)
best_model = bayes_search.best_estimator_
print("逻辑回归最佳参数:", bayes_search.best_params_)

# 在测试集上进行预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算指标
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
brier = brier_score_loss(y_test, y_pred_proba)

# 输出结果
print("\n逻辑回归模型评估结果:")
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Brier Score: {brier:.4f}")





#ROC曲线
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# Data for plotting ROC curves
fpr_full = np.linspace(0, 1, 100)
tpr_full = 1 - (1 - fpr_full)**2  # Example values, change as needed
auc_full = 0.8723

fpr_reduced = np.linspace(0, 1, 100)
tpr_reduced = 1 - (1 - fpr_reduced)**3  # Example values, change as needed
auc_reduced = 0.8714

# Plotting the ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_full, tpr_full, color='blue', lw=2, label=f'Full Model (AUC = {auc_full:.4f})')
plt.plot(fpr_reduced, tpr_reduced, color='orange', lw=2, label=f'Reduced Model (AUC = {auc_reduced:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Before and After Feature Reduction')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Additional Metrics
full_model_metrics = {
    'AUC': 0.8723,
    'Accuracy': 0.820638028,
    'Recall': 0.75721104,
    'Specificity': 0.823988512,
    'Sensitivity': 0.75721104,
    'NPV': 0.984673828,
    'FPR': 0.176011488,
    'FNR': 0.24278896,
    'FDR': 0.814827971,
}

reduced_model_metrics = {
    'AUC': 0.8714,
    'Accuracy': 0.8195,
    'Recall': 0.7562,
    'Specificity': 0.8228,
    'Sensitivity': 0.7562,
    'F1 Score': 0.2959,
    'Precision': 0.1840,
    'Brier Score': 0.1347,
}

# Print Metrics for Both Models
print("Full Model Metrics:")
for key, value in full_model_metrics.items():
    print(f"{key}: {value}")

print("\nReduced Model Metrics:")
for key, value in reduced_model_metrics.items():
    print(f"{key}: {value}")


#SHAP
import shap

# 从测试集中抽样
sample_size = 5000  # 根据需要调整样本大小
if X_test.shape[0] > sample_size:
    idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
    X_test_sample = X_test[idx]
    y_test_sample = y_test[idx]
else:
    X_test_sample = X_test
    y_test_sample = y_test

# 创建用于线性模型的 SHAP Explainer
# 注意，这里使用的是训练后的模型 best_model，以及训练集数据 X_train_resampled
explainer = shap.LinearExplainer(best_model, X_train_resampled, feature_perturbation="interventional")

# 计算 SHAP 值
shap_values = explainer.shap_values(X_test_sample)

# 获取特征名称
selected_feature_names = final_selected_feature_names

# 绘制 SHAP 总结图（调整 X 轴尺度）
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=selected_feature_names, show=False, plot_size=(12, 8))
plt.xlabel("SHAP value (impact on model output, scaled)", fontsize=14)
plt.xlim(-5, 5)
plt.tight_layout()
plt.show()

# 绘制 SHAP 条形图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=selected_feature_names, plot_type='bar', show=False, plot_size=(12, 8))
plt.xlabel("Mean SHAP Value (scaled)", fontsize=14)
plt.xlim(0, 2.0)
plt.tight_layout()
plt.show()

# 绘制单个样本的 SHAP Force Plot
import shap
import numpy as np
import matplotlib.pyplot as plt

sample_index = 4  # 可以更改索引以查看不同样本

# 对 SHAP 值和输入特征进行四舍五入处理，保留 2 位小数
shap_values_rounded = np.round(shap_values[sample_index], 2)
X_test_sample_rounded = np.round(X_test_sample[sample_index], 2)

# 设置 matplotlib 画布大小，增加宽度来确保显示所有特征
plt.figure(figsize=(30, 6))  # 增加宽度

# 绘制单个样本的 SHAP Force Plot
shap.force_plot(
    explainer.expected_value, 
    shap_values_rounded, 
    X_test_sample_rounded, 
    feature_names=selected_feature_names,
    matplotlib=True, 
    show=True
)

#SHAP热图（Heatmap）
import shap
import matplotlib.pyplot as plt

# 确保 PLR 和 CRP 包含在特征名称列表中
required_features = ['PLR', 'CRP']
for feature in required_features:
    if feature not in selected_feature_names:
        selected_feature_names.append(feature)

# 取前 500 个样本的 SHAP 值和对应的数据
num_samples = 100  # 根据需要调整样本数量
if shap_values.shape[0] > num_samples:
    shap_values_subset = shap_values[0:num_samples]
    X_test_sample_subset = X_test_sample[0:num_samples]
else:
    shap_values_subset = shap_values
    X_test_sample_subset = X_test_sample

# 创建 shap.Explanation 对象
shap_explanation = shap.Explanation(values=shap_values_subset,
                                    base_values=explainer.expected_value,
                                    data=X_test_sample_subset,
                                    feature_names=selected_feature_names)

# 绘制 SHAP 热图
plt.figure(figsize=(12, 10))
shap.plots.heatmap(shap_explanation)
plt.show()




#十折
# 保存最佳逻辑回归模型
import joblib
joblib.dump(best_model, 'LOG.pkl')
print("最佳逻辑回归模型已保存为 LOG.pkl")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 假设之前的代码已正确运行，best_model 已经定义
# 保存最佳模型为 LOG
LOG = best_model

# 准备数据
X = X_final_selected  # 已经选择的特征数据
y = Y  # 目标变量

# 确保 X 和 y 是 NumPy 数组
X = np.array(X)
y = np.array(y)

# 设置 StratifiedKFold，n_splits=10 表示 10 折
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 准备绘图
plt.figure(figsize=(10, 8))

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# 定义颜色映射
colors = plt.cm.get_cmap('tab10', 10)

i = 0
for train_index, test_index in cv.split(X, y):
    # 划分训练集和测试集
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    # 在训练集上应用 SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_cv, y_train_cv)
    
    # 特征缩放
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test_cv = scaler.transform(X_test_cv)
    
    # 训练模型
    LOG.fit(X_train_resampled, y_train_resampled)
    
    # 在测试集上预测概率
    y_pred_proba = LOG.predict_proba(X_test_cv)[:, 1]
    
    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(y_test_cv, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    # 插值，便于计算平均 ROC 曲线
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    
    # 绘制当前折的 ROC 曲线（使用实线）
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8, color=colors(i),
             label='ROC fold %d (AUC = %0.4f)' % (i+1, roc_auc))
    
    i += 1

# 绘制对角线
plt.plot([0, 1], [0, 1], linestyle=':', lw=2, color='gray', alpha=.8)

# 计算平均 ROC 曲线和 AUC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0  # 确保最后一个点为 (1,1)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

# 绘制平均 ROC 曲线（使用粗实线）
plt.plot(mean_fpr, mean_tpr, color='b',
         label='Mean ROC (AUC = %0.4f ± %0.4f)' % (mean_auc, std_auc),
         lw=3, alpha=.9)

# 计算 TPR 的标准差，并绘制阴影区域
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightblue', alpha=.2,
                 label='± 1 std. dev.')

# 设置图形参数
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('10-fold Cross-Validation ROC Curves', fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.show()




#五折
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 假设之前的代码已正确运行，best_model 已经定义
# 保存最佳模型为 LOG
LOG = best_model

# 准备数据
X = X_final_selected  # 已选择的特征数据
y = Y  # 目标变量

# 确保 X 和 y 是 NumPy 数组
X = np.array(X)
y = np.array(y)

# 设置 StratifiedKFold，n_splits=5 表示 5 折
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 准备绘图
plt.figure(figsize=(10, 8))

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# 定义颜色映射
colors = plt.cm.get_cmap('tab10', 5)

i = 0
for train_index, test_index in cv.split(X, y):
    # 划分训练集和测试集
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    # 在训练集上应用 SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_cv, y_train_cv)
    
    # 特征缩放
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test_cv = scaler.transform(X_test_cv)
    
    # 训练模型
    LOG.fit(X_train_resampled, y_train_resampled)
    
    # 在测试集上预测概率
    y_pred_proba = LOG.predict_proba(X_test_cv)[:, 1]
    
    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(y_test_cv, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    # 插值，便于计算平均 ROC 曲线
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    
    # 绘制当前折的 ROC 曲线
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8, color=colors(i),
             label='ROC fold %d (AUC = %0.4f)' % (i+1, roc_auc))
    
    i += 1

# 绘制对角线
plt.plot([0, 1], [0, 1], linestyle=':', lw=2, color='gray', alpha=.8)

# 计算平均 ROC 曲线和 AUC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0  # 确保最后一个点为 (1,1)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

# 绘制平均 ROC 曲线
plt.plot(mean_fpr, mean_tpr, color='b',
         label='Mean ROC (AUC = %0.4f ± %0.4f)' % (mean_auc, std_auc),
         lw=3, alpha=.9)

# 计算 TPR 的标准差，并绘制阴影区域
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightblue', alpha=.2,
                 label='± 1 std. dev.')

# 设置图形参数
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('5-fold Cross-Validation ROC Curves', fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.show()








#构建app


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained logistic regression model
model = joblib.load('LOG.pkl')

# Define the feature names based on the final selected features
feature_names = ['HbA1c', 'Tyg', 'LDL', 'age', 'RBC', 'sex', 'GLU', 'WBC', 'PLR', 'CRP']

# Streamlit user interface
st.title("Type 2 Diabetes Mellitus Predictor")

# Collect user inputs for each feature

# HbA1c (%)
HbA1c = st.number_input("HbA1c (%):", min_value=4.0, max_value=15.0, value=5.5, format="%.1f")

# Triglyceride and Glucose to compute Tyg
triglyceride = st.number_input("Triglyceride (mg/dL):", min_value=10.0, max_value=1000.0, value=150.0, format="%.1f")
glucose = st.number_input("Glucose (mg/dL):", min_value=50.0, max_value=500.0, value=100.0, format="%.1f")

# Compute Tyg using the formula: ln [triglyceride (mg/dL) × glucose (mg/dL)/2]
Tyg = np.log((triglyceride * glucose) / 2)

# LDL Cholesterol
LDL = st.number_input("Low-Density Lipoprotein Cholesterol (LDL) (mg/dL):", min_value=10.0, max_value=300.0, value=100.0, format="%.1f")

# Age
age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# RBC
RBC = st.number_input("Red Blood Cell Count (RBC) (×10¹²/L):", min_value=2.0, max_value=10.0, value=4.5, format="%.2f")

# Sex
sex_options = {'Female': 0, 'Male': 1}
sex_input = st.selectbox("Sex:", options=list(sex_options.keys()))
sex = sex_options[sex_input]

# GLU (Assuming it's fasting blood glucose level)
GLU = st.number_input("Blood Glucose Level (GLU) (mg/dL):", min_value=50.0, max_value=500.0, value=100.0, format="%.1f")

# WBC
WBC = st.number_input("White Blood Cell Count (WBC) (×10⁹/L):", min_value=1.0, max_value=20.0, value=6.0, format="%.1f")

# Platelet count and Lymphocyte count to compute PLR
platelet_count = st.number_input("Platelet Count (×10⁹/L):", min_value=10.0, max_value=1000.0, value=250.0, format="%.1f")
lymphocyte_count = st.number_input("Lymphocyte Count (×10⁹/L):", min_value=0.1, max_value=10.0, value=2.0, format="%.2f")

# Compute PLR using the formula: Platelet count / Lymphocyte count
PLR = platelet_count / lymphocyte_count

# CRP
CRP = st.number_input("C-Reactive Protein (CRP) (mg/L):", min_value=0.0, max_value=200.0, value=5.0, format="%.1f")

# Collect the inputs into a feature array
feature_values = [HbA1c, Tyg, LDL, age, RBC, sex, GLU, WBC, PLR, CRP]
features = np.array([feature_values])

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    if predicted_class == 1:
        st.write(f"**Prediction:** High risk of Type 2 Diabetes Mellitus (T2DM)")
    else:
        st.write(f"**Prediction:** Low risk of Type 2 Diabetes Mellitus (T2DM)")
    st.write(f"**Probability of T2DM:** {predicted_proba[1]*100:.1f}%")
    st.write(f"**Probability of Non-T2DM:** {predicted_proba[0]*100:.1f}%")

    # Generate advice based on prediction results
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of Type 2 Diabetes Mellitus (T2DM). "
            f"The model predicts that your probability of having T2DM is {predicted_proba[1]*100:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "We recommend that you consult a healthcare professional for further evaluation."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of Type 2 Diabetes Mellitus (T2DM). "
            f"The model predicts that your probability of not having T2DM is {predicted_proba[0]*100:.1f}%. "
            "Maintaining a healthy lifestyle is still very important. "
            "Regular check-ups are recommended to monitor your health."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    # Note: For the SHAP explainer, we need background data. We'll use a small sample for demonstration.

    # Create background data (mean values)
    background_data = np.mean(features, axis=0).reshape(1, -1)

    # Create SHAP explainer for logistic regression model
    explainer = shap.LinearExplainer(model, background_data)

    # Calculate SHAP values
    shap_values = explainer.shap_values(features)

    # Plot SHAP force plot
    shap.initjs()
    st.subheader("Feature Contribution to Prediction (SHAP Values)")
    # Convert feature values to a DataFrame for better display in SHAP plots
    feature_df = pd.DataFrame(features, columns=feature_names)
    shap.force_plot(
        explainer.expected_value, shap_values[0], feature_df.iloc[0], feature_names=feature_names, matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png")

    # Optionally, display SHAP summary plot
    st.subheader("Overall Feature Importance")
    shap.summary_plot(shap_values, feature_df, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_summary_plot.png")









