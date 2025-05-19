import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from datetime import datetime, timedelta
import yfinance as yf
import talib
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization, Attention
from tensorflow.keras.models import Model
import json

# Set Chinese font for plotting
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== Configuration Module ==================
class Config:
    ALPHA_VANTAGE_API = "PGG1V2L0G9GLCOPR"
    GOLD_SYMBOL = "GLD"
    NDX_SYMBOL = "QQQ"
    VIX_SYMBOL = "^VIX"
    RISK_CONFIG = {
        "max_drawdown": 0.15,
        "vix_thresholds": [15, 25],
        "volatility_window": 21,
        "margin_sensitivity": 0.8
    }

# ================== Data Fetching Module ==================
class DataFetcher:
    @staticmethod
    def fetch_realtime_data(symbol):
        """Fetch real-time data from Alpha Vantage"""
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={Config.ALPHA_VANTAGE_API}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()['Global Quote']
            return {
                'price': float(data['05. price']),
                'change_pct': float(data['10. change percent'].rstrip('%'))
            }
        except Exception as e:
            print(f"Data fetch failed: {e}")
            return None

    @staticmethod
    def fetch_historical_data(symbol, days=60):
        """Fetch historical data using yfinance"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d")
        return data['Close']

    @staticmethod
    def generate_vol_surface(symbol="QQQ", num_strikes=20, max_maturities=5):
        """Generate volatility surface using real option data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            expiry_dates = sorted(ticker.options, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
            selected_expiries = expiry_dates[:max_maturities]
            
            all_strikes = []
            iv_data = []
            
            for expiry in selected_expiries:
                chain = ticker.option_chain(expiry)
                calls = chain.calls[['strike', 'impliedVolatility']]
                puts = chain.puts[['strike', 'impliedVolatility']]
                combined = pd.concat([calls, puts])
                
                valid_data = combined[
                    (combined['impliedVolatility'] > 0.05) & 
                    (combined['impliedVolatility'] < 1.5)
                ]
                
                all_strikes.extend(valid_data['strike'].values)
                iv_data.append(valid_data)
            
            min_strike = np.min(all_strikes)
            max_strike = np.max(all_strikes)
            strikes = np.linspace(min_strike * 0.95, max_strike * 1.05, num_strikes)
            
            iv_matrix = []
            for expiry, df in zip(selected_expiries, iv_data):
                interp_iv = np.interp(
                    strikes,
                    df['strike'].sort_values().values,
                    df['impliedVolatility'].sort_values().values,
                    left=np.nan, right=np.nan
                )
                iv_matrix.append(interp_iv)
            
            iv_matrix = np.array(iv_matrix).T
            today = datetime.now()
            maturities_days = [
                (datetime.strptime(exp, "%Y-%m-%d") - today).days
                for exp in selected_expiries
            ]
            
            return strikes, maturities_days, iv_matrix
        except Exception as e:
            print(f"Volatility surface generation failed: {e}")
            return np.array([]), [], np.array([])

# ================== Analysis Module ==================
class Analyzer:
    def __init__(self):
        self.history = pd.DataFrame(columns=['timestamp', 'gold_pct', 'vix', 'score', 'sentiment'])
        self.lstm_model = self.build_lstm_model()

    def build_lstm_model(self):
        """Build LSTM model with attention mechanism"""
        inputs = Input(shape=(60, 5))
        x = LSTM(64, return_sequences=True)(inputs)
        x = Attention()([x, x])
        x = LayerNormalization()(x)
        x = LSTM(32)(x)
        outputs = Dense(1, activation='tanh')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def calculate_gold_factor(self, price_series):
        """Calculate gold momentum factor with MACD"""
        returns = np.log(price_series).diff().rolling(20).mean() * 100
        macd_line, signal_line, _ = talib.MACD(price_series)
        trend_strength = macd_line[-1] - signal_line[-1]
        
        if returns[-1] > 0.5 and trend_strength > 0:
            return -1.5  # Strong bearish
        elif returns[-1] < -0.5 and trend_strength < 0:
            return 1.5   # Strong bullish
        else:
            return np.sign(returns[-1]) if not np.isnan(returns[-1]) else 0.0

    def analyze_volatility(self, iv_matrix, vix):
        """Analyze volatility smile skewness with dynamic thresholds"""
        if iv_matrix.size == 0:
            return 0.0
        atm_idx = len(iv_matrix) // 2
        skewness = (iv_matrix[0, 0] - iv_matrix[-1, 0]) / iv_matrix[atm_idx, 0] * 100
        threshold = 5 if vix < 15 else 3 if vix < 25 else 2
        if skewness > threshold:
            return -2.0
        elif skewness < -threshold:
            return 1.0
        else:
            return 0.0

    def calculate_sentiment(self):
        """Placeholder for market sentiment analysis (e.g., from news or social media)"""
        # Simulated sentiment score based on random walk (replace with actual API)
        return np.random.uniform(-1, 1)

    def dynamic_weighting(self, vix):
        """Dynamic weighting based on VIX"""
        if vix < 15:
            return (0.50, 0.25, 0.15, 0.10)  # Gold, Vol, ML, Sentiment
        elif vix < 25:
            return (0.35, 0.40, 0.15, 0.10)
        else:
            return (0.20, 0.45, 0.25, 0.10)

    def calculate_score(self, gold_factor, vol_factor, ml_factor, sentiment, vix):
        """Calculate composite score"""
        w_gold, w_vol, w_ml, w_sent = self.dynamic_weighting(vix)
        return (gold_factor * w_gold + vol_factor * w_vol + 
                ml_factor * w_ml + sentiment * w_sent)

# ================== Risk Management Module ==================
class RiskManager:
    def __init__(self):
        self.risk_events = []

    def check_risk(self, score, vix, macro_indicators):
        """Perform risk checks with macro indicators"""
        risk_score = 0
        if score <= -2.0 and vix > 30:
            risk_score = 9.0  # Level 3
            self.trigger_alert("Extreme Risk", "Barometer negative and high VIX")
        elif score <= -1.5:
            risk_score = 7.0  # Level 2
            self.trigger_alert("High Risk", "Barometer score below -1.5")
        elif macro_indicators['gdp_growth'] < 0 or macro_indicators['inflation'] > 4:
            risk_score = 5.0  # Level 1
            self.trigger_alert("Macro Risk", "Adverse GDP or inflation")
        return risk_score

    def trigger_alert(self, title, message):
        """Trigger risk alert"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "level": "CRITICAL",
            "title": title,
            "message": message
        }
        self.risk_events.append(event)
        print(f"[RISK ALERT] {title}: {message}")

# ================== Visualization Module ==================
class Visualizer:
    @staticmethod
    def plot_3d_volsurface(strikes, maturities, iv_matrix):
        """Plot 3D volatility surface with smoothing"""
        if iv_matrix.size == 0:
            return
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(strikes, maturities)
        Z = iv_matrix.T
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
        ax.set_xlabel('执行价')
        ax.set_ylabel('到期天数')
        ax.set_zlabel('隐含波动率 (%)')
        plt.title("NDX波动率曲面")
        plt.colorbar(surf)
        plt.savefig('vol_surface.png')

    @staticmethod
    def plot_heatmap(scores):
        """Plot risk heatmap"""
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            pd.DataFrame(scores),
            cmap='RdYlGn',
            center=0,
            annot=True,
            fmt=".1f",
            linewidths=0.5
        )
        plt.title("市场情绪热力图")
        plt.ylabel("时间窗口")
        plt.xlabel("因子组合")
        plt.savefig('emotion_heatmap.png')

    @staticmethod
    def plot_sentiment_trend(sentiment_data):
        """Plot sentiment trend over time"""
        plt.figure(figsize=(10, 4))
        plt.plot(sentiment_data, label='市场情绪')
        plt.title('市场情绪趋势')
        plt.xlabel('时间')
        plt.ylabel('情绪得分')
        plt.legend()
        plt.grid(True)
        plt.savefig('sentiment_trend.png')

# ================== Main Program ==================
if __name__ == "__main__":
    # Initialize modules
    fetcher = DataFetcher()
    analyzer = Analyzer()
    risk_mgr = RiskManager()
    visualizer = Visualizer()

    # Fetch real-time and historical data
    gold_data = fetcher.fetch_realtime_data(Config.GOLD_SYMBOL)
    ndx_data = fetcher.fetch_realtime_data(Config.NDX_SYMBOL)
    vix_data = fetcher.fetch_realtime_data(Config.VIX_SYMBOL)
    strikes, maturities, iv_matrix = fetcher.generate_vol_surface()
    gold_prices = fetcher.fetch_historical_data(Config.GOLD_SYMBOL, days=60)

    if gold_data and ndx_data and vix_data:
        # Calculate factors
        gold_factor = analyzer.calculate_gold_factor(gold_prices)
        vol_factor = analyzer.analyze_volatility(iv_matrix, vix_data['price'])
        sentiment = analyzer.calculate_sentiment()
        
        # Simulate ML factor (LSTM prediction)
        # Placeholder: replace with actual LSTM input preparation
        ml_factor = np.random.uniform(-1, 1)  # Simulated

        # Calculate composite score
        composite_score = analyzer.calculate_score(
            gold_factor, vol_factor, ml_factor, sentiment, vix_data['price']
        )

        # Record history
        new_row = {
            'timestamp': datetime.now(),
            'gold_pct': gold_data['change_pct'],
            'vix': vix_data['price'],
            'score': composite_score,
            'sentiment': sentiment
        }
        analyzer.history = pd.concat([analyzer.history, pd.DataFrame([new_row])], ignore_index=True)

        # Risk check with macro indicators (simulated)
        macro_indicators = {'gdp_growth': 2.5, 'inflation': 3.0}  # Placeholder
        risk_score = risk_mgr.check_risk(composite_score, vix_data['price'], macro_indicators)

        # Output results
        print(f"\n【市场晴雨表 {datetime.now().strftime('%Y-%m-%d %H:%M')}】")
        print(f"黄金变动: {gold_data['change_pct']:.2f}% → 因子: {gold_factor:.1f}")
        print(f"波动率偏度: {'左偏' if vol_factor < 0 else '正常'} → 因子: {vol_factor:.1f}")
        print(f"市场情绪: {sentiment:.2f}")
        print(f"VIX指数: {vix_data['price']:.1f}")
        print(f"综合得分: {composite_score:.2f} → 风险评分: {risk_score:.1f}")

        # Visualize
        visualizer.plot_3d_volsurface(strikes, maturities, iv_matrix)
        visualizer.plot_heatmap([analyzer.history['score'].values[-24:]])
        visualizer.plot_sentiment_trend(analyzer.history['sentiment'].values[-24:])
        
        plt.tight_layout()
        plt.show()
    else:
        print("无法获取完整数据，请检查网络连接和API配置")
