import MetaTrader5 as mt5
import streamlit as st
import pandas as pd
import numpy as np
import talib
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import plotly.graph_objects as go

class VolatilityMLModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def calculate_advanced_features(self, df):
        """Calculate advanced technical features based on trading tips"""
        
        # 1. TRENDLINES Features
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['TREND_STRENGTH'] = df['ADX'] / 100
        
        # Trend direction
        df['TREND_DIRECTION'] = np.where(df['close'] > df['SMA_20'], 1, 
                                       np.where(df['close'] < df['SMA_20'], -1, 0))
        
        # 2. HERD BEHAVIOR Features
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['WILLIAMS_R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['OVERBOUGHT'] = np.where(df['RSI'] > 70, 1, 0)
        df['OVERSOLD'] = np.where(df['RSI'] < 30, 1, 0)
        df['HERD_EXTREME'] = np.where((df['RSI'] > 80) | (df['RSI'] < 20), 1, 0)
        
        # 3. GAP Features
        df['GAP'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        df['GAP_FILLED'] = np.where(
            (df['GAP'] > 0) & (df['low'] <= df['close'].shift(1)), 1,
            np.where((df['GAP'] < 0) & (df['high'] >= df['close'].shift(1)), 1, 0)
        )
        df['GAP_SIZE'] = abs(df['GAP'])
        
        # 4. VOLATILITY Features
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['VOLATILITY_RATIO'] = df['ATR'] / df['close'] * 100
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(df['close'], timeperiod=20)
        df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
        
        # 5. MOMENTUM Features
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(df['close'])
        df['MOMENTUM'] = talib.MOM(df['close'], timeperiod=10)
        df['PRICE_ACCELERATION'] = df['MOMENTUM'].diff()
        
        # 6. VOLUME Analysis
        if 'volume' in df.columns:
            df['VOLUME_SMA'] = talib.SMA(df['volume'], timeperiod=20)
            df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_SMA']
            df['VOLUME_SPIKE'] = np.where(df['VOLUME_RATIO'] > 1.5, 1, 0)
        
        return df
    
    def create_target_variable(self, df, lookahead=5):
        """Create target variable for ML model"""
        future_return = (df['close'].shift(-lookahead) - df['close']) / df['close'] * 100
        volatility = df['VOLATILITY_RATIO'].rolling(10).mean()
        threshold = volatility * 0.5
        
        df['TARGET'] = np.where(
            future_return > threshold, 1,
            np.where(future_return < -threshold, 0, -1)
        )
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        feature_columns = [
            'TREND_STRENGTH', 'TREND_DIRECTION', 'RSI', 'WILLIAMS_R',
            'OVERBOUGHT', 'OVERSOLD', 'HERD_EXTREME', 'GAP', 'GAP_SIZE',
            'VOLATILITY_RATIO', 'BB_POSITION', 'MACD', 'MACD_HIST',
            'MOMENTUM', 'PRICE_ACCELERATION'
        ]
        
        available_features = [f for f in feature_columns if f in df.columns]
        return df[available_features]
    
    def train_model(self, symbol, period="1y"):
        """Train ML model on historical data"""
        st.info(f"🤖 Training ML Model for {symbol}...")
        
        try:
            # Map MT5 symbols to yfinance symbols
            symbol_map = {
                "SP500m": "SPY", "US500": "SPY", "US500M": "SPY",
                "ND100m": "QQQ", "USTECH100": "QQQ", "USTECH100M": "QQQ", "USTEC": "QQQ",
                "US30M": "DIA", "US30": "DIA",
                "GER30": "EWG", "GER30M": "EWG", "DE40": "EWG",
                "UK100": "EWU", "FTSE100": "EWU",
                "FCH40": "EWQ", "FRA40": "EWQ",
                "NJ225": "EWJ", "JPN225": "EWJ",
                "HSI50": "EWH", "HK50": "EWH",
                "AUS200": "EWA",
                "GDAXlm": "EWG",
                "SPN35": "EWP", "ESP35": "EWP",
                "EUSTX50": "FEZ", "STOX50": "FEZ",
                "SSE_Comp": "FXI",
                "MXSHAR": "EWW", "MXSHARTR": "EWW",
                "RVL1": "RWL"
            }
            
            yf_symbol = symbol_map.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return False
            
            df = self.calculate_advanced_features(df)
            df = self.create_target_variable(df)
            df = df.dropna()
            
            if len(df) < 100:
                return False
            
            X = self.prepare_features(df)
            y = df['TARGET']
            
            if len(X) == 0:
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            st.success(f"✅ ML Model Trained - Accuracy: {accuracy:.2%}")
            return True
            
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            return False
    
    def predict_signal(self, df):
        """Predict trading signal using ML model"""
        if not self.is_trained or self.model is None:
            return "HOLD", 50
        
        try:
            df_features = self.calculate_advanced_features(df)
            latest_features = self.prepare_features(df_features).iloc[-1:].fillna(0)
            
            if latest_features.empty:
                return "HOLD", 50
            
            features_scaled = self.scaler.transform(latest_features)
            prediction = self.model.predict(features_scaled)[0]
            probability = np.max(self.model.predict_proba(features_scaled))
            
            if prediction == 1:
                signal = "BUY"
            elif prediction == 0:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            confidence = probability * 100
            return signal, confidence
            
        except Exception as e:
            return "HOLD", 50
    
    def generate_trading_insights(self, df, symbol):
        """Generate insights based on volatility trading tips"""
        df = self.calculate_advanced_features(df)
        
        insights = {
            'trend_analysis': '',
            'herd_sentiment': '',
            'gap_analysis': '',
            'volatility_opportunity': '',
            'risk_assessment': ''
        }
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_trend = df['TREND_DIRECTION'].iloc[-1]
        gap_size = df['GAP'].iloc[-1]
        volatility = df['VOLATILITY_RATIO'].iloc[-1]
        
        # Trend Analysis
        if current_trend == 1:
            insights['trend_analysis'] = "📈 Strong Uptrend - Follow trend direction"
        elif current_trend == -1:
            insights['trend_analysis'] = "📉 Strong Downtrend - Consider short opportunities"
        else:
            insights['trend_analysis'] = "➡️ Sideways Market - Range trading opportunities"
        
        # Herd Behavior
        if current_rsi > 70:
            insights['herd_sentiment'] = "🚨 OVERBOUGHT - Herd is buying, consider contrarian position"
        elif current_rsi < 30:
            insights['herd_sentiment'] = "💎 OVERSOLD - Herd is selling, potential buying opportunity"
        else:
            insights['herd_sentiment'] = "⚖️ Neutral sentiment - No extreme herd behavior"
        
        # Gap Analysis
        if abs(gap_size) > 1:
            if gap_size > 0:
                insights['gap_analysis'] = f"🔼 Gap UP {gap_size:.2f}% - Watch for gap fill opportunity"
            else:
                insights['gap_analysis'] = f"🔽 Gap DOWN {abs(gap_size):.2f}% - Potential gap fill play"
        else:
            insights['gap_analysis'] = "✅ No significant gaps - Normal trading"
        
        # Volatility Opportunities
        if volatility > 3:
            insights['volatility_opportunity'] = f"🌪️ High Volatility ({volatility:.1f}%) - Great for volatility strategies"
        elif volatility < 1:
            insights['volatility_opportunity'] = f"🍃 Low Volatility ({volatility:.1f}%) - Consider breakout plays"
        else:
            insights['volatility_opportunity'] = f"📊 Normal Volatility ({volatility:.1f}%) - Standard trading"
        
        # Risk Assessment
        atr = df['ATR'].iloc[-1]
        risk_per_share = atr * 2
        insights['risk_assessment'] = f"🎯 Recommended Stop Loss: ${risk_per_share:.2f} ({risk_per_share/current_price*100:.1f}%)"
        
        return insights

class VolatilityTradingManager:
    def __init__(self):
        self.connected = False
        self.ml_model = VolatilityMLModel()
        # Updated with indices from your screenshot
        self.volatility_symbols = [
            # US Indices
            "SP500m", "ND100m", "US30M", "US500M", "USTECH100M", "US30", "US500", "USTECH100", "NASDAQ",
            # European Indices
            "GER30", "GER30M", "DE40", "UK100", "FTSE100", "FCH40", "FRA40", "EUSTX50", "STOX50", "SPN35", "ESP35",
            # Asian Indices
            "NJ225", "JPN225", "HSI50", "HK50", "AUS200", "SSE_Comp",
            # Other Global Indices
            "GDAXlm", "MXSHAR", "MXSHARTR", "RVL1"
        ]
    
    def connect_terminal_only(self):
        """Simple terminal-only connection"""
        try:
            st.info("🔗 Connecting to MT5 Terminal...")
            if mt5.initialize():
                self.connected = True
                st.success("✅ SUCCESS! Connected to MT5 Terminal")
                return True
            else:
                st.error("❌ MT5 Connection Failed")
                return False
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            return False
    
    def get_market_data(self, symbol, period="6mo"):
        """Get market data from Yahoo Finance with symbol mapping"""
        try:
            # Map MT5 symbols to yfinance symbols
            symbol_map = {
                "SP500m": "SPY", "US500": "SPY", "US500M": "SPY",
                "ND100m": "QQQ", "USTECH100": "QQQ", "USTECH100M": "QQQ", "USTEC": "QQQ",
                "US30M": "DIA", "US30": "DIA",
                "GER30": "EWG", "GER30M": "EWG", "DE40": "EWG",
                "UK100": "EWU", "FTSE100": "EWU",
                "FCH40": "EWQ", "FRA40": "EWQ",
                "NJ225": "EWJ", "JPN225": "EWJ",
                "HSI50": "EWH", "HK50": "EWH",
                "AUS200": "EWA",
                "GDAXlm": "EWG",
                "SPN35": "EWP", "ESP35": "EWP",
                "EUSTX50": "FEZ", "STOX50": "FEZ",
                "SSE_Comp": "FXI",
                "MXSHAR": "EWW", "MXSHARTR": "EWW",
                "RVL1": "RWL",
                "NASDAQ": "QQQ"
            }
            
            yf_symbol = symbol_map.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return None
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
            
            return df
            
        except Exception as e:
            st.error(f"❌ Data error for {symbol}: {str(e)}")
            return None
    
    def get_current_price(self, symbol):
        """Get current price with symbol mapping"""
        try:
            symbol_map = {
                "SP500m": "SPY", "US500": "SPY", "US500M": "SPY",
                "ND100m": "QQQ", "USTECH100": "QQQ", "USTECH100M": "QQQ", "USTEC": "QQQ",
                "US30M": "DIA", "US30": "DIA",
                "GER30": "EWG", "GER30M": "EWG", "DE40": "EWG",
                "UK100": "EWU", "FTSE100": "EWU",
                "FCH40": "EWQ", "FRA40": "EWQ",
                "NJ225": "EWJ", "JPN225": "EWJ",
                "HSI50": "EWH", "HK50": "EWH",
                "AUS200": "EWA",
                "GDAXlm": "EWG",
                "SPN35": "EWP", "ESP35": "EWP",
                "EUSTX50": "FEZ", "STOX50": "FEZ",
                "SSE_Comp": "FXI",
                "MXSHAR": "EWW", "MXSHARTR": "EWW",
                "RVL1": "RWL",
                "NASDAQ": "QQQ"
            }
            
            yf_symbol = symbol_map.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d")
            return hist['Close'].iloc[-1] if not hist.empty else None
        except:
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate basic technical indicators"""
        if df is None or len(df) < 20:
            return df
        
        try:
            high = df['high'].values
            low = df['low'].values 
            close = df['close'].values
            
            df['SMA_20'] = talib.SMA(close, timeperiod=20)
            df['RSI'] = talib.RSI(close, timeperiod=14)
            df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            df['MACD'], df['MACD_Signal'], _ = talib.MACD(close)
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(close, timeperiod=20)
            df['Volatility_Ratio'] = df['ATR'] / df['close'] * 100
            
            return df
        except:
            return df
    
    def get_market_analysis(self, symbol):
        """Basic technical analysis"""
        st.info(f"📊 Analyzing {symbol}...")
        
        df = self.get_market_data(symbol)
        if df is None:
            return None
        
        df = self.calculate_technical_indicators(df)
        current_price = self.get_current_price(symbol) or df['close'].iloc[-1]
        
        # Simple signal logic
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
        
        if rsi < 30:
            signal = "BUY"
            confidence = 75
            tp = current_price + (atr * 2)
            sl = current_price - atr
        elif rsi > 70:
            signal = "SELL"
            confidence = 75  
            tp = current_price - (atr * 2)
            sl = current_price + atr
        else:
            signal = "HOLD"
            confidence = 50
            tp = current_price
            sl = current_price
        
        result = {
            'symbol': symbol,
            'current_price': current_price,
            'signal': signal,
            'confidence': confidence,
            'take_profit': round(tp, 4),
            'stop_loss': round(sl, 4),
            'risk_reward': abs((tp - current_price) / (current_price - sl)) if sl != current_price else 0,
            'indicators': {
                'RSI': rsi,
                'ATR': atr,
                'ADX': df['ADX'].iloc[-1] if 'ADX' in df.columns else 0,
                'MACD': df['MACD'].iloc[-1] if 'MACD' in df.columns else 0,
                'BB_Upper': df['BB_Upper'].iloc[-1] if 'BB_Upper' in df.columns else 0,
                'BB_Lower': df['BB_Lower'].iloc[-1] if 'BB_Lower' in df.columns else 0,
                'SMA_20': df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else 0,
                'Volatility_Ratio': df['Volatility_Ratio'].iloc[-1] if 'Volatility_Ratio' in df.columns else 0
            },
            'data': df
        }
        
        return result
    
    def analyze_with_ml(self, symbol):
        """ML-powered analysis"""
        st.info(f"🤖 ML Analysis for {symbol}...")
        
        df = self.get_market_data(symbol)
        if df is None:
            return None
        
        # Train ML model if not trained
        if not self.ml_model.is_trained:
            self.ml_model.train_model(symbol)
        
        # Get ML prediction
        ml_signal, ml_confidence = self.ml_model.predict_signal(df)
        
        # Generate insights
        insights = self.ml_model.generate_trading_insights(df, symbol)
        
        # Calculate TP/SL
        current_price = df['close'].iloc[-1]
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        
        if ml_signal == "BUY":
            tp = current_price + (atr * 2.5)
            sl = current_price - (atr * 1.5)
        elif ml_signal == "SELL":
            tp = current_price - (atr * 2.5)
            sl = current_price + (atr * 1.5)
        else:
            tp = sl = current_price
        
        result = {
            'symbol': symbol,
            'current_price': current_price,
            'signal': ml_signal,
            'confidence': ml_confidence,
            'take_profit': round(tp, 4),
            'stop_loss': round(sl, 4),
            'risk_reward': abs((tp - current_price) / (current_price - sl)) if sl != current_price else 0,
            'insights': insights,
            'data': df
        }
        
        return result
    
    def create_ml_analysis_dashboard(self, result):
        """Create ML analysis dashboard"""
        st.subheader("🧠 ML Trading Insights")
        
        insights = result['insights']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Trendlines**: {insights['trend_analysis']}")
            st.info(f"**Herd Behavior**: {insights['herd_sentiment']}")
        
        with col2:
            st.info(f"**Gap Analysis**: {insights['gap_analysis']}")
            st.info(f"**Volatility**: {insights['volatility_opportunity']}")
        
        st.info(f"**Risk**: {insights['risk_assessment']}")
    
    def create_price_chart(self, df, symbol, current_price, tp, sl, signal):
        """Create simple price chart"""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['close'], 
            mode='lines', 
            name='Price',
            line=dict(color='blue', width=2)
        ))
        
        # Current price
        fig.add_hline(y=current_price, line_dash="dash", line_color="green",
                     annotation_text=f"Current: ${current_price:.2f}")
        
        # TP/SL lines
        if signal in ["BUY", "SELL"]:
            fig.add_hline(y=tp, line_dash="dot", line_color="green",
                         annotation_text=f"TP: ${tp:.2f}")
            fig.add_hline(y=sl, line_dash="dot", line_color="red",
                         annotation_text=f"SL: ${sl:.2f}")
        
        fig.update_layout(
            title=f"{symbol} Price - Signal: {signal}",
            height=400
        )
        
        return fig
    
    def get_available_symbols(self):
        return self.volatility_symbols
    
    def place_trade(self, symbol, signal, volume, tp, sl):
        """Place trade in MT5"""
        try:
            if not self.connected:
                st.error("❌ Not connected to MT5")
                return False
            
            if signal not in ["BUY", "SELL"]:
                st.error("❌ Invalid signal")
                return False
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                st.error(f"❌ Symbol {symbol} not found")
                return False
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                st.error("❌ Cannot get current price")
                return False
            
            order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "price": tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid,
                "sl": float(sl),
                "tp": float(tp),
                "deviation": 20,
                "magic": 234000,
                "comment": "ML Volatility Trading",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                st.error(f"❌ Trade failed")
                return False
            else:
                st.success(f"✅ {signal} {symbol} executed!")
                return True
                
        except Exception as e:
            st.error(f"❌ Trade error: {str(e)}")
            return False
    
    def disconnect_mt5(self):
        """Disconnect from MT5"""
        try:
            mt5.shutdown()
            self.connected = False
            st.info("🔌 Disconnected from MT5")
            return True
        except:
            return False