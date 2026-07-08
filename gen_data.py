"""生成部署用脱敏数据：删除姓名/会员号等隐私列，仅保留群体统计所需字段。"""
import pandas as pd
src = '/sandbox/workspace/uploads/Data.xls'
out = '/sandbox/workspace/outputs/hra_dashboard/data.xlsx'
df = pd.read_excel(src)
before = df.shape[1]
df = df.drop(columns=[c for c in ['姓名', '会员号'] if c in df.columns])
df.to_excel(out, index=False)
print(f'已脱敏保存: {out} | 列 {before} -> {df.shape[1]} | 行 {df.shape[0]} | 已删除隐私列: 姓名/会员号')
