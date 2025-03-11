"""
纳斯达克晴雨表算法完整实现
版本：3.1
"""

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from datetime import datetime, timedelta
import seaborn as sns
import json

# ================== 配置模块 ==================
class Config:
    ALPHA_VANTAGE_API = "YOUR_API_KEY"
    GOLD_SYMBOL = "GC=F"
    NDX_SYMBOL = "^NDX"
    RISK_CONFIG = {
        "max_drawdown": 0.15,
        "vix_thresholds": [15, 25],
        "volatility_window": 21
    }

# ================== 数据获取模块 ==================
class DataFetcher:
    @staticmethod
    def fetch_realtime_data(symbol):
        """从Alpha Vantage获取实时数据"""
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={Config.ALPHA_VANTAGE_API}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()['Global Quote']
            return {
                'price': float(data['05. price']),
                'change_pct': float(data['10. change percent'].rstrip('%'))
            }
        except Exception as e:
            print(f"数据获取失败: {e}")
            return None

    @staticmethod
    def generate_vol_surface():
        """生成模拟波动率曲面"""
        strikes = np.linspace(17000, 19000, 5)
        maturities = [1, 7, 30]
        iv_matrix = np.array([
            [24.5, 23.1, 22.3],
            [22.1, 21.0, 20.5],
            [20.0, 19.5, 19.0],
            [22.3, 21.2, 20.8],
            [24.8, 23.5, 22.9]
        ])
        return strikes, maturities, iv_matrix

# ================== 分析计算模块 ==================    
class Analyzer:
    def __init__(self):
        self.history = pd.DataFrame(columns=['timestamp', 'gold_pct', 'vix', 'score'])
        
    def calculate_gold_factor(self, gold_pct):
        """计算黄金动量因子"""
        if gold_pct > 0.5:
            return -1.5
        elif gold_pct < -0.5:
            return +1.5
        else:
            return 0.0
            
    def analyze_volatility(self, iv_matrix):
        """分析波动率微笑偏度"""
        left_skew = iv_matrix[0,0] - iv_matrix[2,0] > 3
        return -2.0 if left_skew else 1.0
        
    def dynamic_weighting(self, vix):
        """动态权重调整"""
        if vix < 15:
            return (0.55, 0.30)
        elif vix < 25:
            return (0.40, 0.45)
        else:
            return (0.25, 0.50)
            
    def calculate_score(self, gold_factor, vol_factor, vix):
        """计算综合得分"""
        w_gold, w_vol = self.dynamic_weighting(vix)
        return gold_factor * w_gold + vol_factor * w_vol

# ================== 风险管理模块 ==================        
class RiskManager:
    def __init__(self):
        self.risk_events = []
        
    def check_risk(self, score, vix):
        """执行风险检查"""
        risk_level = 0
        if score <= -2.0 and vix > 30:
            self.trigger_alert("极端风险事件", "同时触发晴雨表负分和VIX高位")
            risk_level = 3
        elif score <= -1.5:
            self.trigger_alert("高风险状态", "晴雨表得分低于-1.5")
            risk_level = 2
        return risk_level
            
    def trigger_alert(self, title, message):
        """触发风险警报"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "level": "CRITICAL",
            "title": title,
            "message": message
        }
        self.risk_events.append(event)
        print(f"[RISK ALERT] {title}: {message}")

# ================== 可视化模块 ==================        
class Visualizer:
    @staticmethod
    def plot_3d_volsurface(strikes, maturities, iv_matrix):
        """绘制3D波动率曲面"""
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(strikes, maturities)
        Z = iv_matrix.T  # 转置保证维度对齐
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Days to Maturity')
        ax.set_zlabel('Implied Volatility (%)')
        plt.title("NDX波动率曲面")
        plt.colorbar(surf)
        
    @staticmethod
    def plot_heatmap(scores):
        """绘制风险热力图"""
        plt.figure(figsize=(12,6))
        sns.heatmap(pd.DataFrame(scores), 
                   cmap='RdYlGn', 
                   center=0,
                   annot=True, 
                   fmt=".1f",
                   linewidths=0.5)
        plt.title("市场情绪热力图")
        plt.ylabel("时间窗口")
        plt.xlabel("因子组合")

# ================== 主程序 ==================        
if __name__ == "__main__":
    # 初始化模块
    fetcher = DataFetcher()
    analyzer = Analyzer()
    risk_mgr = RiskManager()
    visualizer = Visualizer()
    
    # 获取实时数据
    gold_data = fetcher.fetch_realtime_data(Config.GOLD_SYMBOL)
    ndx_data = fetcher.fetch_realtime_data(Config.NDX_SYMBOL)
    strikes, maturities, iv_matrix = fetcher.generate_vol_surface()
    
    if gold_data and ndx_data:
        # 计算因子
        gold_factor = analyzer.calculate_gold_factor(gold_data['change_pct'])
        vol_factor = analyzer.analyze_volatility(iv_matrix)
        
        # 假设当前VIX值
        current_vix = 22.3  # 实际应获取VIX数据
        
        # 合成得分
        composite_score = analyzer.calculate_score(gold_factor, vol_factor, current_vix)
        
        # 记录历史数据
        new_row = {
            'timestamp': datetime.now(),
            'gold_pct': gold_data['change_pct'],
            'vix': current_vix,
            'score': composite_score
        }
        analyzer.history = analyzer.history.append(new_row, ignore_index=True)
        
        # 风险检查
        risk_level = risk_mgr.check_risk(composite_score, current_vix)
        
        # 输出结果
        print(f"\n【市场晴雨表 {datetime.now().strftime('%Y-%m-%d %H:%M')}】")
        print(f"黄金变动: {gold_data['change_pct']:.2f}% → 因子: {gold_factor:.1f}")
        print(f"波动率偏度: {'左偏' if vol_factor < 0 else '正常'} → 因子: {vol_factor:.1f}")
        print(f"VIX指数: {current_vix:.1f}")
        print(f"综合得分: {composite_score:.2f} → 风险等级: {risk_level}")
        
        # 可视化
        visualizer.plot_3d_volsurface(strikes, maturities, iv_matrix)
        visualizer.plot_heatmap([analyzer.history['score'].values[-24:]])  # 展示最近24个数据点
        
        plt.tight_layout()
        plt.show()
    else:
        print("无法获取完整数据，请检查网络连接和API配置")