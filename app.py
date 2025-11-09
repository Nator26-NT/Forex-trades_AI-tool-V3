import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import MetaTrader5 as mt5

# Import our volatility trading manager
from mt5_manager import VolatilityTradingManager

def main():
    st.set_page_config(
        page_title="AI Volatility Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-buy {
        background-color: #00ff00 !important;
        color: black !important;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .signal-sell {
        background-color: #ff0000 !important;
        color: white !important;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .signal-hold {
        background-color: #ffff00 !important;
        color: black !important;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .ml-insight {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤖 AI Volatility Trading System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'mt5_manager' not in st.session_state:
        st.session_state.mt5_manager = VolatilityTradingManager()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'ml_analysis' not in st.session_state:
        st.session_state.ml_analysis = {}
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = "VIX"
    
    # Sidebar
    st.sidebar.title("⚙️ Trading Controls")
    
    # Connection Section
    st.sidebar.subheader("🔗 Connection")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Connect to MT5"):
            with st.spinner("Connecting to MT5..."):
                if st.session_state.mt5_manager.connect_terminal_only():
                    st.sidebar.success("Connected!")
    
    with col2:
        if st.button("Disconnect"):
            st.session_state.mt5_manager.disconnect_mt5()
            st.sidebar.info("Disconnected")
    
    # Volatility Symbol Selection
    st.sidebar.subheader("📊 Volatility Symbols")
    available_symbols = st.session_state.mt5_manager.get_available_symbols()
    selected_symbol = st.sidebar.selectbox(
        "Select Volatility Index:",
        available_symbols,
        index=0 if available_symbols else None
    )
    
    # Trading Parameters
    st.sidebar.subheader("💰 Trading Parameters")
    trade_volume = st.sidebar.number_input("Trade Volume (Lots):", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
    
    # ML Model Section
    st.sidebar.subheader("🤖 ML Model")
    ml_col1, ml_col2 = st.sidebar.columns(2)
    
    with ml_col1:
        if st.button("Train ML Model"):
            if selected_symbol:
                with st.spinner("Training ML Model..."):
                    if st.session_state.mt5_manager.ml_model.train_model(selected_symbol):
                        st.sidebar.success("ML Model Trained!")
            else:
                st.sidebar.error("Please select a symbol")
    
    with ml_col2:
        st.session_state.ml_enabled = st.checkbox("Use ML", value=True)
    
    # Auto-refresh option
    st.sidebar.subheader("🔄 Auto Refresh")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (15 seconds)")
    
    # Analysis Type Selection
    st.sidebar.subheader("🔍 Analysis Type")
    analysis_type = st.sidebar.radio(
        "Choose Analysis Method:",
        ["ML Smart Analysis", "Basic Technical Analysis"]
    )
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Trading Signals")
        
        # Analysis Buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if analysis_type == "ML Smart Analysis":
                if st.button("🧠 Run ML Analysis", type="primary", use_container_width=True):
                    if selected_symbol:
                        with st.spinner(f"Running ML Analysis on {selected_symbol}..."):
                            result = st.session_state.mt5_manager.analyze_with_ml(selected_symbol)
                            
                            if result:
                                st.session_state.ml_analysis[selected_symbol] = result
                                st.session_state.current_symbol = selected_symbol
                                display_ml_results(result)
                    else:
                        st.error("Please select a volatility symbol")
            else:
                if st.button("📊 Basic Analysis", type="primary", use_container_width=True):
                    if selected_symbol:
                        with st.spinner(f"Analyzing {selected_symbol}..."):
                            result = st.session_state.mt5_manager.get_market_analysis(selected_symbol)
                            
                            if result:
                                st.session_state.analysis_results[selected_symbol] = result
                                st.session_state.current_symbol = selected_symbol
                                display_basic_results(result, selected_symbol)
                    else:
                        st.error("Please select a volatility symbol")
        
        with col_btn2:
            if st.button("📈 Show Chart", use_container_width=True):
                if selected_symbol in st.session_state.analysis_results:
                    result = st.session_state.analysis_results[selected_symbol]
                    display_chart(result, selected_symbol)
                elif selected_symbol in st.session_state.ml_analysis:
                    result = st.session_state.ml_analysis[selected_symbol]
                    display_chart(result, selected_symbol)
                else:
                    st.warning("Please run analysis first")
        
        # Display current analysis results
        if selected_symbol in st.session_state.ml_analysis and analysis_type == "ML Smart Analysis":
            display_ml_results(st.session_state.ml_analysis[selected_symbol])
        elif selected_symbol in st.session_state.analysis_results and analysis_type == "Basic Technical Analysis":
            display_basic_results(st.session_state.analysis_results[selected_symbol], selected_symbol)
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(15)
            st.rerun()
    
    with col2:
        st.subheader("⚡ Quick Actions")
        
        # Determine which result to use
        current_result = None
        if selected_symbol in st.session_state.ml_analysis:
            current_result = st.session_state.ml_analysis[selected_symbol]
        elif selected_symbol in st.session_state.analysis_results:
            current_result = st.session_state.analysis_results[selected_symbol]
        
        if current_result:
            display_quick_actions(current_result, selected_symbol, trade_volume)
    
    # Market Overview Section
    st.markdown("---")
    st.subheader("📈 Volatility Market Overview")
    
    # Multi-symbol analysis
    if st.button("🔄 Scan All Volatility Symbols"):
        with st.spinner("Scanning volatility market..."):
            results = []
            symbols_to_scan = available_symbols[:6]  # Limit to 6 symbols
            
            progress_bar = st.progress(0)
            for i, symbol in enumerate(symbols_to_scan):
                if st.session_state.ml_enabled:
                    result = st.session_state.mt5_manager.analyze_with_ml(symbol)
                else:
                    result = st.session_state.mt5_manager.get_market_analysis(symbol)
                    
                if result:
                    results.append(result)
                progress_bar.progress((i + 1) / len(symbols_to_scan))
                time.sleep(1)  # Rate limiting
            
            if results:
                display_market_overview(results)
    
    # Strategy Explanation
    st.markdown("---")
    st.subheader("🎯 Trading Strategy")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("""
        **🤖 ML Smart Analysis:**
        - **Trendlines**: Moving averages, ADX, trend strength
        - **Herd Behavior**: RSI extremes, contrarian signals
        - **Gap Analysis**: Gap size, fill probability
        - **Volatility Opportunities**: ATR, Bollinger Bands
        - **Risk Assessment**: Dynamic stop losses
        """)
        
        st.markdown("""
        **📊 Technical Indicators:**
        - **RSI** - Momentum and overbought/oversold
        - **ATR** - Volatility measurement  
        - **MACD** - Trend direction
        - **Bollinger Bands** - Volatility channels
        - **ADX** - Trend strength
        """)
    
    with exp_col2:
        st.markdown("""
        **💡 Volatility Trading Tips:**
        - **Use trendlines** - Follow the dominant trend
        - **Don't follow the herd** - Look for contrarian opportunities
        - **Take position early** - Act on news and gaps quickly
        - **Fill the gap** - Trade gap fill opportunities
        - **Venture a guess** - Use probability-based entries
        """)
        
        st.markdown("""
        **⚡ Supported Symbols:**
        - VIX, VXX, UVXY, SVXY, VXZ
        - SPY, QQQ, IWM, DIA, GLD
        - USO, TLT, HYG, LQD, EEM
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>⚡ AI Volatility Trading System</strong> - Powered by Machine Learning</p>
        <p><em>Advanced volatility analysis using ML | Always practice risk management</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_ml_results(result):
    """Display ML analysis results"""
    st.success(f"✅ ML Analysis Complete for {result['symbol']}")
    
    # Main Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${result['current_price']:.4f}")
    
    with col2:
        signal_class = ""
        if "BUY" in result['signal']:
            signal_class = "signal-buy"
        elif "SELL" in result['signal']:
            signal_class = "signal-sell"
        else:
            signal_class = "signal-hold"
        
        st.markdown(f'<div class="{signal_class}">{result["signal"]}</div>', unsafe_allow_html=True)
    
    with col3:
        if result['confidence'] > 70:
            st.metric("ML Confidence", f"{result['confidence']:.1f}%", delta="High")
        elif result['confidence'] > 50:
            st.metric("ML Confidence", f"{result['confidence']:.1f}%", delta="Medium")
        else:
            st.metric("ML Confidence", f"{result['confidence']:.1f}%", delta="Low")
    
    with col4:
        if result['risk_reward'] > 2:
            st.metric("Risk/Reward", f"{result['risk_reward']:.2f}:1", delta="Excellent")
        elif result['risk_reward'] > 1:
            st.metric("Risk/Reward", f"{result['risk_reward']:.2f}:1", delta="Good")
        else:
            st.metric("Risk/Reward", f"{result['risk_reward']:.2f}:1", delta="Poor")
    
    # ML Insights
    st.subheader("🧠 ML Trading Insights")
    
    insights = result['insights']
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown(f'<div class="ml-insight">📈 <strong>Trend Analysis:</strong><br>{insights["trend_analysis"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ml-insight">👥 <strong>Herd Behavior:</strong><br>{insights["herd_sentiment"]}</div>', unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown(f'<div class="ml-insight">↔️ <strong>Gap Analysis:</strong><br>{insights["gap_analysis"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ml-insight">🌪️ <strong>Volatility Opportunity:</strong><br>{insights["volatility_opportunity"]}</div>', unsafe_allow_html=True)
    
    # Risk Assessment
    st.markdown(f'<div class="ml-insight">🎯 <strong>Risk Assessment:</strong><br>{insights["risk_assessment"]}</div>', unsafe_allow_html=True)
    
    # TP/SL Information
    st.subheader("🎯 Take Profit & Stop Loss")
    tp_col, sl_col, rr_col = st.columns(3)
    
    with tp_col:
        st.info(f"**TAKE PROFIT:** ${result['take_profit']:.4f}")
        tp_distance = abs(result['take_profit'] - result['current_price'])
        st.metric("TP Distance", f"${tp_distance:.4f}")
    
    with sl_col:
        st.error(f"**STOP LOSS:** ${result['stop_loss']:.4f}")
        sl_distance = abs(result['stop_loss'] - result['current_price'])
        st.metric("SL Distance", f"${sl_distance:.4f}")
    
    with rr_col:
        profit_potential = tp_distance / result['current_price'] * 100
        risk_amount = sl_distance / result['current_price'] * 100
        st.metric("Profit Potential", f"{profit_potential:.2f}%")
        st.metric("Risk Amount", f"{risk_amount:.2f}%")
    
    # Trading Recommendation
    st.subheader("💡 ML Trading Recommendation")
    
    if result['signal'] == "BUY" and result['confidence'] > 65:
        st.success("""
        **🎯 STRONG BUY RECOMMENDATION**
        
        **Action:** Enter long position
        **Strategy:** Follow trend with volatility-adjusted position sizing
        **Risk Management:** Use tight stop loss below support
        **Target:** Volatility expansion moves
        
        *ML Confidence: High - Favorable risk/reward ratio*
        """)
    elif result['signal'] == "SELL" and result['confidence'] > 65:
        st.error("""
        **🎯 STRONG SELL RECOMMENDATION**
        
        **Action:** Enter short position  
        **Strategy:** Capitalize on overbought conditions
        **Risk Management:** Use volatility-based stops above resistance
        **Target:** Trend continuation or reversal
        
        *ML Confidence: High - Clear contrarian signal*
        """)
    else:
        st.warning("""
        **⚖️ WAIT FOR BETTER SETUP**
        
        **Action:** Remain in cash or reduce position size
        **Strategy:** Wait for clearer ML signals
        **Risk Management:** Preserve capital
        **Target:** Better risk/reward opportunities
        
        *ML Confidence: Moderate - Market conditions uncertain*
        """)

def display_basic_results(result, symbol):
    """Display basic technical analysis results"""
    st.success(f"✅ Analysis Complete for {symbol}")
    
    # Main Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${result['current_price']:.4f}")
    
    with col2:
        signal_class = ""
        if "BUY" in result['signal']:
            signal_class = "signal-buy"
        elif "SELL" in result['signal']:
            signal_class = "signal-sell"
        else:
            signal_class = "signal-hold"
        
        st.markdown(f'<div class="{signal_class}">{result["signal"]}</div>', unsafe_allow_html=True)
    
    with col3:
        st.metric("Confidence", f"{result['confidence']:.1f}%")
    
    with col4:
        st.metric("Risk/Reward", f"{result['risk_reward']:.2f}:1")
    
    # TP/SL Information
    st.subheader("🎯 Take Profit & Stop Loss")
    tp_col, sl_col, rr_col = st.columns(3)
    
    with tp_col:
        st.info(f"**TAKE PROFIT:** ${result['take_profit']:.4f}")
        tp_distance = abs(result['take_profit'] - result['current_price'])
        st.metric("TP Distance", f"${tp_distance:.4f}")
    
    with sl_col:
        st.error(f"**STOP LOSS:** ${result['stop_loss']:.4f}")
        sl_distance = abs(result['stop_loss'] - result['current_price'])
        st.metric("SL Distance", f"${sl_distance:.4f}")
    
    with rr_col:
        profit_potential = tp_distance / result['current_price'] * 100
        risk_amount = sl_distance / result['current_price'] * 100
        st.metric("Profit Potential", f"{profit_potential:.2f}%")
        st.metric("Risk Amount", f"{risk_amount:.2f}%")
    
    # Technical Indicators
    st.subheader("📊 Technical Indicators")
    indicators = result['indicators']
    
    ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
    
    with ind_col1:
        st.metric("RSI", f"{indicators['RSI']:.1f}")
        st.metric("ATR", f"{indicators['ATR']:.4f}")
    
    with ind_col2:
        st.metric("ADX", f"{indicators['ADX']:.1f}")
        st.metric("MACD", f"{indicators['MACD']:.4f}")
    
    with ind_col3:
        st.metric("BB Upper", f"{indicators['BB_Upper']:.4f}")
        st.metric("BB Lower", f"{indicators['BB_Lower']:.4f}")
    
    with ind_col4:
        st.metric("SMA 20", f"{indicators['SMA_20']:.4f}")
        st.metric("Volatility %", f"{indicators['Volatility_Ratio']:.2f}%")

def display_chart(result, symbol):
    """Display price chart"""
    st.subheader("📈 Price Chart")
    
    chart = st.session_state.mt5_manager.create_price_chart(
        result['data'], 
        symbol, 
        result['current_price'],
        result['take_profit'],
        result['stop_loss'],
        result['signal']
    )
    st.plotly_chart(chart, use_container_width=True)

def display_quick_actions(result, symbol, trade_volume):
    """Display quick action buttons and trade info"""
    
    # Execute Trade Buttons
    trade_col1, trade_col2 = st.columns(2)
    
    with trade_col1:
        if "BUY" in result['signal']:
            if st.button("✅ EXECUTE BUY", type="primary", use_container_width=True):
                if st.session_state.mt5_manager.place_trade(
                    symbol=symbol,
                    signal=result['signal'],
                    volume=trade_volume,
                    tp=result['take_profit'],
                    sl=result['stop_loss']
                ):
                    st.success("Buy trade executed!")
        else:
            st.button("✅ EXECUTE BUY", disabled=True, use_container_width=True)
    
    with trade_col2:
        if "SELL" in result['signal']:
            if st.button("🔻 EXECUTE SELL", type="primary", use_container_width=True):
                if st.session_state.mt5_manager.place_trade(
                    symbol=symbol,
                    signal=result['signal'],
                    volume=trade_volume,
                    tp=result['take_profit'],
                    sl=result['stop_loss']
                ):
                    st.success("Sell trade executed!")
        else:
            st.button("🔻 EXECUTE SELL", disabled=True, use_container_width=True)
    
    # Trade Details Card
    st.subheader("📋 Trade Setup")
    
    st.info(f"""
    **Symbol:** {result['symbol']}
    **Signal:** {result['signal']}
    **Volume:** {trade_volume} lots
    **Entry Price:** ${result['current_price']:.4f}
    **Take Profit:** ${result['take_profit']:.4f}
    **Stop Loss:** ${result['stop_loss']:.4f}
    **Confidence:** {result['confidence']:.1f}%
    **Risk/Reward:** {result['risk_reward']:.2f}:1
    """)
    
    # Quick Analysis
    st.subheader("🔍 Quick Analysis")
    
    # Confidence Level
    if result['confidence'] > 70:
        st.success("**High Confidence Trade** - Strong signal quality")
    elif result['confidence'] > 50:
        st.warning("**Medium Confidence Trade** - Moderate signal quality")
    else:
        st.error("**Low Confidence** - Wait for better setup")
    
    # Risk/Reward Assessment
    if result['risk_reward'] > 2:
        st.success("**Excellent Risk/Reward** - Favorable ratio")
    elif result['risk_reward'] > 1:
        st.info("**Good Risk/Reward** - Acceptable ratio")
    else:
        st.warning("**Poor Risk/Reward** - Consider adjusting stops")
    
    # Position Size Calculator
    st.subheader("🧮 Position Calculator")
    
    account_balance = st.number_input("Account Balance ($):", min_value=1000, max_value=1000000, value=10000, step=1000)
    risk_percent = st.slider("Risk per Trade (%):", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    
    risk_amount = account_balance * (risk_percent / 100)
    price_distance = abs(result['current_price'] - result['stop_loss'])
    
    if price_distance > 0:
        position_size = risk_amount / price_distance
        st.metric("Recommended Position Size", f"{position_size:.2f} units")
        st.metric("Risk Amount", f"${risk_amount:.2f}")

def display_market_overview(results):
    """Display market overview table"""
    st.subheader("📊 Market Overview Results")
    
    # Create results dataframe
    df_results = pd.DataFrame([{
        'symbol': r['symbol'],
        'price': r['current_price'],
        'signal': r['signal'],
        'confidence': r['confidence'],
        'tp': r['take_profit'],
        'sl': r['stop_loss'],
        'risk_reward': r['risk_reward']
    } for r in results])
    
    # Display styled dataframe
    st.dataframe(
        df_results.style.applymap(
            lambda x: 'background-color: #90EE90' if 'BUY' in str(x) else ('background-color: #FFB6C1' if 'SELL' in str(x) else ''),
            subset=['signal']
        ).format({
            'price': '${:.2f}',
            'confidence': '{:.1f}%',
            'tp': '${:.2f}',
            'sl': '${:.2f}',
            'risk_reward': '{:.2f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Summary Statistics
    st.subheader("📈 Market Summary")
    
    buy_signals = len([r for r in results if "BUY" in r['signal']])
    sell_signals = len([r for r in results if "SELL" in r['signal']])
    hold_signals = len([r for r in results if "HOLD" in r['signal']])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Buy Signals", buy_signals)
    
    with col2:
        st.metric("Sell Signals", sell_signals)
    
    with col3:
        st.metric("Hold Signals", hold_signals)

if __name__ == "__main__":
    main()