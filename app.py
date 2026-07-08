"""
HRA 群体健康洞察仪表盘  v1.0
=================================================================
数据：396 份真实 HRA（惠斯安普）健康风险评估检测记录（已脱敏，仅做群体统计）
用途：帮养老机构 / 体检中心 / 健康企业「一眼看清人群整体健康趋势与干预重点」
对应专业方向：数字健康管理 · 健康大数据
作者：袁爱（东北大学秦皇岛分校 健康服务与管理专业）
-----------------------------------------------------------------
指标说明
- 身体区域功能偏离值：HRA 对 85 个身体区域逐项检测，输出功能偏离度（正=功能偏亢，负=功能偏低），
  绝对值越大代表该区域越偏离正常。本仪表盘取 |偏离| 衡量异常程度。
- 异常参考线：|偏离| >= 30 记为「需关注」（HRA 通用判读惯例，机构可自定义阈值）。
- 综合健康指数（派生指标）：100 − 平均区域绝对偏离 ×(100/60)，用于群体横向对比，非临床诊断。
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

THRESH = 30  # 异常参考线

# 85 个身体区域 → 8 大系统归类
SYS_MAP = {
    '呼吸系统': ['气管附近', '支气管区域', '左肺上叶区域', '左肺下叶区域', '右肺上叶区域',
               '右肺中叶区域', '右肺下叶区域', '胸部左侧区域', '胸部右侧区域'],
    '消化系统': ['食道上段', '食道下段', '胃区域', '十二指肠区域', '小肠区域', '盲肠和阑尾区域',
               '升结肠区域', '结肠肝区', '结肠脾区', '降结肠区域', '乙状结肠区域', '直肠区域',
               '肝左叶及胆管区域', '肝右页', '胆囊区域', '胰腺区域'],
    '心血管系统': ['心脏区域', '左颈动脉', '右颈动脉', '心肺循环', '心肌', '左心室', '右心室'],
    '骨骼系统': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Th1', 'Th2', 'Th3', 'Th4', 'Th5',
               'Th6', 'Th7', 'Th8', 'Th9', 'Th10', 'Th11', 'Th12', 'L1', 'L2', 'L3',
               '左膝区域（腿部血管）', '右膝区域（腿部血管）'],
    '内分泌系统': ['甲状腺区域', '甲状腺左叶区域', '甲状腺右叶区域', '垂体区域', '下丘脑区域',
                '丘脑', '左侧肾上腺髓质', '右侧肾上腺髓质', '胸腺', '脾脏区域'],
    '泌尿生殖系统': ['左肾及输尿管区域', '右肾及输尿管区域', '膀胱区域', '前列腺区域',
                  '左睾丸区域', '右睾丸区域'],
    '免疫系统': ['左侧颈部区域', '右侧颈部区域', '右侧膈神经区域'],
    '神经感官系统': ['左眼和泪腺区域', '右眼和泪腺区域', '左上颌窦区域', '右上颌窦区域',
                  '右侧鼻前庭和固有鼻腔区域', '左侧鼻前庭和固有鼻腔区域', '左唾液腺',
                  '右唾液腺', '左耳区域', '右耳区域'],
}

FONT = dict(family='Microsoft YaHei, SimHei, PingFang SC, Noto Sans CJK SC, sans-serif')
BAND_ORDER = ['健康', '良好', '亚健康', '警戒']
BAND_COLOR = {'健康': '#2E7D32', '良好': '#7CB342', '亚健康': '#FB8C00', '警戒': '#E53935'}


@st.cache_data
def load_default():
    return pd.read_excel('data.xlsx')


def compute(df):
    # 年龄 / BMI（不加载姓名、会员号等隐私列）
    bd = pd.to_datetime(df['出生日期'].astype(str).str.strip(), errors='coerce')
    cd = pd.to_datetime(df['检查日期'].astype(str).str.strip(), errors='coerce')
    df['age'] = (cd - bd).dt.days / 365.25
    h = pd.to_numeric(df['身高 (cm)'], errors='coerce') / 100
    df['bmi'] = pd.to_numeric(df['体重 (kg)'], errors='coerce') / (h ** 2)
    # 区域偏离绝对值矩阵
    regions = [c for cols in SYS_MAP.values() for c in cols if c in df.columns]
    absdf = df[regions].apply(pd.to_numeric, errors='coerce').abs()
    avg_abs = absdf.mean(axis=1)
    df['health_index'] = (100 - avg_abs * (100 / 60)).clip(0, 100)
    df['band'] = df['health_index'].apply(
        lambda x: '健康' if x >= 80 else '良好' if x >= 60 else '亚健康' if x >= 40 else '警戒')
    df['sex'] = df['性别'].astype(str).str.strip()
    df['age_grp'] = pd.cut(df['age'], [0, 25, 35, 45, 55, 200],
                           labels=['<25', '25-34', '35-44', '45-54', '55+'])
    return df, absdf, regions


# ---------- 数据源：默认内置 396 份，支持上传自定义数据 ----------
with st.sidebar:
    st.header('📥 上传你的 HRA 数据')
    st.caption('支持 Excel(.xlsx/.xls)，需与示例同格式（含 85 个身体区域列 + 人口学列）。')
    uploaded = st.file_uploader('选择文件', type=['xlsx', 'xls'])
    if uploaded is not None:
        raw = pd.read_excel(uploaded)
        st.success(f'✅ 已加载上传数据：{raw.shape[0]} 行')
    else:
        raw = load_default()
        st.info('当前展示：内置 396 份示例数据。上传文件即可分析你自己的群体。')
    st.caption('⚠️ 公开 demo，请勿上传含敏感个人身份信息的数据。')

df, absdf, regions = compute(raw)
if not regions:
    st.error('⚠️ 上传的文件未找到 HRA 身体区域列，请确认格式与示例一致（含「左肺上叶区域」「胃区域」等 85 个区域列）。可在仓库内下载 data.xlsx 对照格式。')
    st.stop()

# ---------- 派生指标 ----------
N = len(df)
HI_MEAN = df['health_index'].mean()
sys_rate = {s: absdf[[c for c in cols if c in absdf.columns]].ge(THRESH).mean().mean() * 100
            for s, cols in SYS_MAP.items()}
TOP_SYS = max(sys_rate, key=sys_rate.get)
RISK_SHARE = df['band'].isin(['亚健康', '警戒']).mean() * 100

# ============================ 页面 ============================
st.set_page_config(page_title='HRA 群体健康洞察仪表盘', page_icon='📊', layout='wide')
st.title('📊 HRA 群体健康洞察仪表盘')
st.caption('HRA 群体健康洞察 · 数字健康管理 / 健康大数据方向作品 · 作者：袁爱')

# 侧边栏说明
with st.sidebar:
    st.header('ℹ️ 关于本仪表盘')
    st.markdown(
        '**数据**：396 份真实 HRA 健康风险评估记录（已脱敏，仅群体统计，不含个人姓名/编号）。\n\n'
        '**核心指标**：85 个身体区域功能偏离值 → 按 8 大系统归类。\n\n'
        '**异常参考线**：`|区域偏离| ≥ 30` 记为「需关注」（HRA 通用判读惯例，可在代码中调整 `THRESH`）。\n\n'
        '**综合健康指数**（派生）：`100 − 平均区域绝对偏离 ×(100/60)`，用于群体横向对比，**非临床诊断**。\n\n'
        '**适用场景**：养老机构人群管理、体检中心企业报告、健康企业员工关怀。')
    st.divider()
    st.caption('东北大学秦皇岛分校 · 健康服务与管理（120410T）')

# ---------- 数据来源 banner ----------
src_label = '你上传的数据' if uploaded is not None else '内置 396 份示例数据'
st.caption(f'📌 当前分析样本：**{N} 人** · 数据来源：{src_label}')

# ---------- 顶部 KPI ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric('样本量', f'{N} 人')
c2.metric('平均健康指数', f'{HI_MEAN:.1f} / 100')
c3.metric('最高危系统', TOP_SYS)
c4.metric('亚健康+警戒占比', f'{RISK_SHARE:.1f}%')

st.divider()

# ---------- 1. 群体健康指数分布 ----------
st.header('1️⃣ 群体健康指数分布')
col_a, col_b = st.columns([3, 2])
with col_a:
    cnt = df['band'].value_counts().reindex(BAND_ORDER).fillna(0)
    fig = go.Figure(data=[go.Pie(
        labels=cnt.index, values=cnt.values,
        marker=dict(colors=[BAND_COLOR[b] for b in cnt.index]),
        textinfo='percent+label', hole=0.45)])
    fig.update_layout(font=FONT, height=360, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
with col_b:
    st.info(
        '**💡 专业解读**\n\n'
        f'本群体平均健康指数 **{HI_MEAN:.1f}**，处于「良好」偏下、逼近「亚健康」临界。\n\n'
        f'约 **{RISK_SHARE:.1f}%** 的人群落在亚健康及以下（{int(cnt["亚健康"])} 人亚健康 + '
        f'{int(cnt["警戒"])} 人警戒），提示存在**较大的群体性健康干预空间**——这正是机构配置干预资源的依据。')

st.divider()

# ---------- 2. 各系统异常率排行 ----------
st.header('2️⃣ 各系统异常率排行（干预优先级）')
sr = pd.Series(sys_rate).sort_values()
fig = go.Figure(go.Bar(
    x=sr.values, y=sr.index, orientation='h',
    marker=dict(color=sr.values, colorscale='YlOrRd'),
    text=[f'{v:.1f}%' for v in sr.values], textposition='outside'))
fig.update_layout(font=FONT, height=380, margin=dict(t=20, b=20),
                  xaxis_title='异常率（|区域偏离|≥30 的区域占比）')
st.plotly_chart(fig, use_container_width=True)
st.info(
    '**💡 专业解读**\n\n'
    f'**{TOP_SYS}（{sys_rate[TOP_SYS]:.1f}%）、呼吸系统（{sys_rate["呼吸系统"]:.1f}%）、'
    f'骨骼系统（{sys_rate["骨骼系统"]:.1f}%）**为三大高发异常系统。\n\n'
    '三者分别与「饮食结构 / 久坐 / 姿势不良」高度相关，应作为机构干预资源投放的**优先方向**；'
    '免疫系统异常率最低（8.2%），说明群体免疫储备相对尚可。')

st.divider()

# ---------- 3. 年龄分层对比 ----------
st.header('3️⃣ 年龄分层对比')
ag = df.groupby('age_grp', observed=True)['health_index'].mean()
agn = df.groupby('age_grp', observed=True).size()
fig = go.Figure()
fig.add_trace(go.Scatter(x=ag.index, y=ag.values, mode='lines+markers+text',
                         text=[f'{v:.0f}' for v in ag.values], textposition='top center',
                         line=dict(color='#1565C0', width=3), marker=dict(size=10)))
fig.update_layout(font=FONT, height=340, margin=dict(t=20, b=20),
                  yaxis_title='平均健康指数', xaxis_title='年龄组')
st.plotly_chart(fig, use_container_width=True)
st.info(
    '**💡 专业解读**\n\n'
    f'25–34 岁群体指数最高（{ag.get("25-34",0):.1f}），但 **45–54 岁骤降至 {ag.get("45-54",0):.1f}**，'
    '呈典型「中年塌陷」特征。\n\n'
    '建议将 **45 岁以上人群列为重点筛查与早期干预对象**，在机能明显下滑前介入，性价比最高。')

st.divider()

# ---------- 4. 性别差异 ----------
st.header('4️⃣ 性别差异（各系统异常率）')
rows = []
for sex in ['男', '女']:
    sub = absdf[df['sex'] == sex]
    for s, cols in SYS_MAP.items():
        cols2 = [c for c in cols if c in sub.columns]
        if cols2:
            rows.append({'性别': sex, '系统': s, '异常率': sub[cols2].ge(THRESH).mean().mean() * 100})
gs = pd.DataFrame(rows)
fig = px.bar(gs, x='系统', y='异常率', color='性别', barmode='group',
             color_discrete_map={'男': '#1565C0', '女': '#E91E63'})
fig.update_layout(font=FONT, height=380, margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)
st.info(
    '**💡 专业解读**\n\n'
    f'女性平均健康指数（{df[df["sex"]=="女"]["health_index"].mean():.1f}）**低于男性**'
    f'（{df[df["sex"]=="男"]["health_index"].mean():.1f}），且内分泌、骨骼系统更易受累。\n\n'
    '建议为女性群体配置**针对性的内分泌与骨骼健康管理方案**（如骨密度监测、激素平衡调理）。')

st.divider()

# ---------- 5. 区域偏离 TOP 10 ----------
st.header('5️⃣ 身体区域偏离 TOP 10（最弱器官定位）')
reg_mean = absdf.mean().sort_values(ascending=False).head(10).iloc[::-1]
fig = go.Figure(go.Bar(
    x=reg_mean.values, y=reg_mean.index, orientation='h',
    marker=dict(color=reg_mean.values, colorscale='Reds'),
    text=[f'{v:.1f}' for v in reg_mean.values], textposition='outside'))
fig.update_layout(font=FONT, height=420, margin=dict(t=20, b=20),
                  xaxis_title='平均 |区域偏离|')
st.plotly_chart(fig, use_container_width=True)
st.info(
    '**💡 专业解读**\n\n'
    f'**{reg_mean.index[-1]}、{reg_mean.index[-2]}** 及心肺循环区域偏离最显著，提示'
    '呼吸系统与心肺循环功能整体偏弱。\n\n'
    '这可与**有氧训练、呼吸训练**等运动干预直接对接——正好呼应「社会体育指导员」专长，'
    '把数据洞察落到可执行的健康活动上。')

st.divider()

# ---------- 6. 系统联动热力图 ----------
st.header('6️⃣ 系统异常联动（健康问题「打包」规律）')
sys_flag = pd.DataFrame({s: absdf[[c for c in cols if c in absdf.columns]].ge(THRESH).mean(axis=1)
                         for s, cols in SYS_MAP.items()})
corr = sys_flag.corr()
fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                aspect='auto')
fig.update_layout(font=FONT, height=480, margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)
st.info(
    '**💡 专业解读**\n\n'
    '各系统异常**高度共现**（相关 0.5–0.89），说明群体问题并非单点，而是「**整体机能偏低**」的系统性表现。\n\n'
    '干预应采用「**整体调理 + 重点突破**」策略，而非头痛医头；同时可据此设计跨系统的综合干预课程，'
    '提升投入产出比。')

st.divider()
st.caption('📌 本仪表盘为群体健康洞察工具，所有结论基于群体统计，不构成任何个体的临床诊断。'
           '综合健康指数为派生相对指标，异常参考线可依据机构标准调整。')
