"""冒烟测试：mock streamlit，真实执行 app.py 全部计算与 plotly 构建逻辑，抓运行期错误。"""
import sys, os
from unittest.mock import MagicMock
os.chdir(os.path.dirname(os.path.abspath(__file__)))
st = MagicMock()
st.cache_data = lambda f: f  # 装饰器变 identity
st.columns = lambda n: tuple(MagicMock() for _ in range(n if isinstance(n, int) else len(n)))
sys.modules['streamlit'] = st
import importlib.util
spec = importlib.util.spec_from_file_location('hra_app', 'app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
print('SMOKE OK -> 样本=%d 平均指数=%.1f 最高危系统=%s 亚健康+警戒=%.1f%%'
      % (app.N, app.HI_MEAN, app.TOP_SYS, app.RISK_SHARE))
