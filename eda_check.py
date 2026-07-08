"""EDA 验证脚本：确认 396 份 HRA 数据的仪表盘指标计算合理（不读隐私列）。"""
import pandas as pd, numpy as np

SRC = '/sandbox/workspace/uploads/Data.xls'
THRESH = 30  # HRA 通用判读：|偏离|>=30 为需关注

SYS_MAP = {
    '呼吸系统': ['气管附近','支气管区域','左肺上叶区域','左肺下叶区域','右肺上叶区域','右肺中叶区域','右肺下叶区域','胸部左侧区域','胸部右侧区域'],
    '消化系统': ['食道上段','食道下段','胃区域','十二指肠区域','小肠区域','盲肠和阑尾区域','升结肠区域','结肠肝区','结肠脾区','降结肠区域','乙状结肠区域','直肠区域','肝左叶及胆管区域','肝右页','胆囊区域','胰腺区域'],
    '心血管系统': ['心脏区域','左颈动脉','右颈动脉','心肺循环','心肌','左心室','右心室'],
    '骨骼系统': ['C1','C2','C3','C4','C5','C6','C7','Th1','Th2','Th3','Th4','Th5','Th6','Th7','Th8','Th9','Th10','Th11','Th12','L1','L2','L3','左膝区域（腿部血管）','右膝区域（腿部血管）'],
    '内分泌系统': ['甲状腺区域','甲状腺左叶区域','甲状腺右叶区域','垂体区域','下丘脑区域','丘脑','左侧肾上腺髓质','右侧肾上腺髓质','胸腺','脾脏区域'],
    '泌尿生殖系统': ['左肾及输尿管区域','右肾及输尿管区域','膀胱区域','前列腺区域','左睾丸区域','右睾丸区域'],
    '免疫系统': ['左侧颈部区域','右侧颈部区域','右侧膈神经区域'],
    '神经感官系统': ['左眼和泪腺区域','右眼和泪腺区域','左上颌窦区域','右上颌窦区域','右侧鼻前庭和固有鼻腔区域','左侧鼻前庭和固有鼻腔区域','左唾液腺','右唾液腺','左耳区域','右耳区域'],
}

df = pd.read_excel(SRC)
# 年龄/BMI（不读姓名/会员号）
bd = pd.to_datetime(df['出生日期'].astype(str).str.strip(), errors='coerce')
cd = pd.to_datetime(df['检查日期'].astype(str).str.strip(), errors='coerce')
df['age'] = (cd - bd).dt.days / 365.25
df['bmi'] = pd.to_numeric(df['体重 (kg)'], errors='coerce') / (pd.to_numeric(df['身高 (cm)'], errors='coerce') / 100) ** 2

all_regions = [c for cols in SYS_MAP.values() for c in cols if c in df.columns]
absdf = df[all_regions].apply(pd.to_numeric, errors='coerce').abs()
print('区域列数:', len(all_regions), '| 样本:', len(df), '| 有效年龄:', int(df['age'].notna().sum()))

# 1 综合健康指数
avg_abs = absdf.mean(axis=1)
df['health_index'] = (100 - avg_abs * (100 / 60)).clip(0, 100)
def band(x):
    return '健康' if x >= 80 else '良好' if x >= 60 else '亚健康' if x >= 40 else '警戒'
df['band'] = df['health_index'].apply(band)
print('\n[1] 群体健康指数 mean=%.1f' % df['health_index'].mean())
print('    档位分布:', df['band'].value_counts().to_dict())

# 2 系统异常率
print('\n[2] 各系统异常率(|偏离|>=%d占比):' % THRESH)
sys_rate = {s: absdf[[c for c in cols if c in absdf.columns]].ge(THRESH).mean().mean() * 100 for s, cols in SYS_MAP.items()}
for k, v in sorted(sys_rate.items(), key=lambda x: -x[1]):
    print('    %s: %.1f%%' % (k, v))

# 3 年龄分层
df['age_grp'] = pd.cut(df['age'], [0, 25, 35, 45, 55, 200], labels=['<25', '25-34', '35-44', '45-54', '55+'])
print('\n[3] 年龄组 平均健康指数:')
print(df.groupby('age_grp', observed=True)['health_index'].mean().round(1).to_dict())

# 4 性别
print('\n[4] 性别 平均健康指数:')
print(df.groupby(df['性别'].astype(str).str.strip())['health_index'].mean().round(1).to_dict())

# 5 区域 TOP10
reg_mean = absdf.mean().sort_values(ascending=False)
print('\n[5] 区域平均绝对偏离 TOP10:')
for c, v in reg_mean.head(10).items():
    print('    %s: %.1f' % (c, v))

# 6 系统联动
sys_flag = pd.DataFrame({s: absdf[[c for c in cols if c in absdf.columns]].ge(THRESH).mean(axis=1) for s, cols in SYS_MAP.items()})
print('\n[6] 系统异常共现相关(>0.3):')
corr = sys_flag.corr()
for i in range(len(corr)):
    for j in range(i + 1, len(corr)):
        if corr.iloc[i, j] > 0.3:
            print('    %s ~ %s: %.2f' % (corr.index[i], corr.columns[j], corr.iloc[i, j]))
print('\nBMI mean=%.1f' % df['bmi'].mean())
