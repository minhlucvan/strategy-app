import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt 
import os 
from dotenv import load_dotenv
from utils.component import input_SymbolsDate, check_password, params_selector, form_SavePortfolio
from utils.sector_analysis import SectorType, SectorMetrics
from screeners.StockScreener import StockScreener

def get_api_key():
    """Get API key from environment variable"""
    return os.getenv("FMP_API_KEY")

def main():
    st.title("Advanced Stock Screener with Enhanced Analysis")

    # Add description
    st.markdown("""
    This stock screener helps you analyze stocks across different sectors using:
    - 📊 Fundamental Analysis (P/E ratios, margins, growth)
    - 📈 Technical Analysis (RSI, Moving Averages)
    - 🏢 Sector-Specific Metrics
    - ⚠️ Risk Assessment
    """)
    
    # Create tabs for main content and educational content
    main_tab, education_tab = st.tabs(["Stock Analysis", "Learn More"])
    
    # Educational Content
    with education_tab:
        st.header("📚 Understanding Stock Analysis")
        
        with st.expander("Valuation Metrics Explained"):
            st.markdown("""
            ### Valuation Score (0-100)
            - **80+ : Significantly Undervalued** - Stock may be trading well below its fair value
            - **60-79: Moderately Undervalued** - Stock appears somewhat undervalued
            - **40-59: Fairly Valued** - Stock is trading close to its fair value
            - **20-39: Moderately Overvalued** - Stock appears somewhat overvalued
            - **<20: Significantly Overvalued** - Stock may be trading well above its fair value
            
            The valuation score combines multiple factors including:
            - P/E Ratio (Price to Earnings)
            - P/B Ratio (Price to Book)
            - PEG Ratio (Price/Earnings to Growth)
            - DCF Valuation (Discounted Cash Flow)
            """)
            
        with st.expander("Technical Analysis Explained"):
            st.markdown("""
            ### Technical Indicators
            - **RSI (Relative Strength Index)**
                - Above 70: Potentially overbought
                - Below 30: Potentially oversold
                - Between 30-70: Neutral territory
            
            - **Moving Averages**
                - Golden Cross (50-day MA crosses above 200-day MA): Bullish signal
                - Death Cross (50-day MA crosses below 200-day MA): Bearish signal
            """)
            
        with st.expander("Understanding Sector Analysis"):
            st.markdown("""
            ### Sector-Specific Metrics
            Different sectors require different analytical approaches:
            
            - **Technology**: R&D spending, margin trends, innovation metrics
            - **Financial**: Interest margins, loan quality, capital ratios
            - **Healthcare**: Drug pipeline, R&D efficiency, regulatory risks
            - **Consumer**: Brand value, market share, inventory turnover
            """)
    
    # Main Analysis Tab
    with main_tab:
        try:
            api_key = get_api_key()
            screener = StockScreener(api_key)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        
        # Sidebar with enhanced help
        with st.sidebar:
                        
            symbolsDate_dict = input_SymbolsDate(group=True)

            symbol_benchmark = symbolsDate_dict['benchmark']
            
            # Initialize session state for watchlist
            if len(symbolsDate_dict['symbols']) > 0:
                st.session_state.watchlist = symbolsDate_dict['symbols']
            else:
                st.session_state.watchlist = []
            
            
            # Display and manage watchlist
            st.subheader("Analysis List")
            for stock in st.session_state.watchlist:
                col1, col2 = st.columns([3, 1])
                col1.write(stock)
                if col2.button("🗑️", key=f"remove_{stock}"):
                    st.session_state.watchlist.remove(stock)
            
            if st.button("Clear List"):
                st.session_state.watchlist = []

            # Analysis button (moved back to sidebar)
            if st.session_state.watchlist:
                if st.button("Analyze Stocks"):
                    st.session_state.analyze_clicked = True
            else:
                st.info("Add stocks to analyze")

        # Main content area (analysis results only)
        if st.session_state.watchlist:
            if getattr(st.session_state, 'analyze_clicked', False):
                with st.spinner('Performing comprehensive analysis...'):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, ticker in enumerate(st.session_state.watchlist):
                        try:
                            profile = screener._get_company_profile(ticker)
                            st.write(profile)
                            if profile:
                                sector = profile.get('sector')
                                sector_type = map_sector_to_type(sector)
                                analysis = screener.analyze_stock(ticker, sector_type)
                                if analysis:
                                    results.append(analysis)
                            progress_bar.progress((i + 1) / len(st.session_state.watchlist))
                        except Exception as e:
                            st.error(f"Error analyzing {ticker}: {str(e)}")
                    
                    if results:
                        df = pd.DataFrame(results)
                        
                        # Create tabs for different views
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "Summary Dashboard",
                            "Technical Analysis",
                            "Sector Analysis",
                            "Stock Comparison",
                            "Detailed Analysis"
                        ])
                        
                        with tab1:
                            display_summary_dashboard(df)
                        
                        with tab2:
                            display_technical_analysis(df)
                        
                        with tab3:
                            display_sector_analysis(df, screener)
                        
                        with tab4:
                            display_stock_comparison(df)
                        
                        with tab5:
                            display_detailed_analysis(df)
                    else:
                        st.warning("No analysis results were generated")
        else:
            st.info("👈 Add stocks to your analysis list using the sidebar")

def display_summary_dashboard(df):
    """Display enhanced summary dashboard"""
    st.subheader("📊 Market Overview")
    
    # Enhanced metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_score = df['Valuation Score'].mean()
        delta = avg_score - 50
        st.metric(
            "Average Valuation Score", 
            f"{avg_score:.1f}",
            delta=f"{delta:.1f} vs Neutral",
            help="Scores above 50 indicate undervaluation"
        )
    
    with col2:
        best_stock = df.loc[df['Valuation Score'].idxmax()]
        st.metric(
            "Top Pick", 
            best_stock['Ticker'],
            f"Score: {best_stock['Valuation Score']:.1f}",
            help="Stock with the highest overall score"
        )
    
    with col3:
        avg_tech_score = df['Technical Score'].mean()
        tech_delta = avg_tech_score - 50
        st.metric(
            "Average Technical Score",
            f"{avg_tech_score:.1f}",
            delta=f"{tech_delta:.1f} vs Neutral",
            help="Technical strength indicator"
        )
    
    # Market interpretation
    if avg_score > 60:
        st.success("🔥 Overall market segment appears undervalued")
    elif avg_score < 40:
        st.warning("⚠️ Overall market segment appears overvalued")
    else:
        st.info("📊 Overall market segment appears fairly valued")
    
    # Enhanced recommendation distribution
    st.subheader("Investment Recommendations")
    rec_df = df.groupby('Recommendation').size().reset_index(name='Count')
    fig_rec = px.pie(
        rec_df, 
        values='Count', 
        names='Recommendation',
        color='Recommendation',
        color_discrete_map={
            'Strong Buy': '#2E7D32',
            'Buy': '#4CAF50',
            'Hold': '#FFC107',
            'Sell': '#F44336',
            'Strong Sell': '#B71C1C'
        },
        hole=0.4
    )
    st.plotly_chart(fig_rec)
    
    # Key insights
    display_key_insights(df)

def display_technical_analysis(df):
    """Display enhanced technical analysis"""
    st.subheader("Technical Indicators")
    
    # Add technical analysis explanation
    with st.expander("Understanding Technical Indicators"):
        st.markdown("""
        - **RSI (Relative Strength Index)**: Momentum indicator showing overbought/oversold conditions
        - **Moving Averages**: Trend indicators showing short-term vs long-term price movements
        - **Volume**: Trading activity indicator
        """)
    
    # RSI Analysis with interpretation
    st.subheader("RSI Analysis")
    fig_rsi = px.scatter(
        df,
        x='Ticker',
        y='RSI',
        color='Recommendation',
        title='Relative Strength Index (RSI)',
        color_discrete_map={
            'Strong Buy': '#2E7D32',
            'Buy': '#4CAF50',
            'Hold': '#FFC107',
            'Sell': '#F44336',
            'Strong Sell': '#B71C1C'
        }
    )
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    st.plotly_chart(fig_rsi)
    
    # Moving Averages with crossover analysis
    st.subheader("Moving Average Analysis")
    ma_data = df[['Ticker', 'MA50', 'MA200']].melt(
        id_vars=['Ticker'],
        var_name='MA Type',
        value_name='Value'
    )
    fig_ma = px.line(
        ma_data,
        x='Ticker',
        y='Value',
        color='MA Type',
        title='Moving Averages Comparison'
    )
    st.plotly_chart(fig_ma)
    
    # Add MA crossover signals
    for _, row in df.iterrows():
        if row['MA50'] > row['MA200']:
            st.info(f"🔵 {row['Ticker']}: Golden Cross (Bullish Signal)")
        elif row['MA50'] < row['MA200']:
            st.warning(f"🔴 {row['Ticker']}: Death Cross (Bearish Signal)")

def display_stock_comparison(df):
    """Display stock comparison analysis"""
    st.subheader("📊 Stock Comparison")
    
    # Select stocks to compare
    stocks_to_compare = st.multiselect(
        "Select stocks to compare (max 3)",
        df['Ticker'].tolist(),
        max_selections=3
    )
    
    if stocks_to_compare:
        comparison_df = df[df['Ticker'].isin(stocks_to_compare)]
        
        # Radar chart comparison
        metrics = ['Valuation Score', 'Technical Score', 'Sector Score']
        fig = go.Figure()
        
        for ticker in stocks_to_compare:
            stock_data = comparison_df[comparison_df['Ticker'] == ticker]
            fig.add_trace(go.Scatterpolar(
                r=[stock_data[metric].iloc[0] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=ticker
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Comparative Analysis"
        )
        st.plotly_chart(fig)
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        comparison_metrics = ['Ticker', 'Current Price', 'Valuation Score', 
                            'Technical Score', 'Sector Score', 'Recommendation']
        st.dataframe(comparison_df[comparison_metrics])

def display_key_insights(df):
    """Display key insights from analysis"""
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Best Value Opportunities")
        best_value = df.nlargest(3, 'Valuation Score')
        for _, stock in best_value.iterrows():
            st.write(f"**{stock['Ticker']}**: Score {stock['Valuation Score']:.1f}")
            st.write(f"Recommendation: {stock['Recommendation']}")
            st.write("---")
    
    with col2:
        st.markdown("#### Technical Standouts")
        best_tech = df.nlargest(3, 'Technical Score')
        for _, stock in best_tech.iterrows():
            st.write(f"**{stock['Ticker']}**: Score {stock['Technical Score']:.1f}")
            st.write(f"RSI: {stock.get('RSI', 'N/A')}")
            st.write("---")

def display_detailed_analysis(df):
    """Display detailed analysis with filtering"""
    st.subheader("Comprehensive Analysis")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Valuation Score", 0, 100, 0)
    with col2:
        selected_recommendations = st.multiselect(
            "Filter by Recommendation",
            df['Recommendation'].unique().tolist(),
            default=df['Recommendation'].unique().tolist()
        )
    
    # Filter DataFrame
    filtered_df = df[
        (df['Valuation Score'] >= min_score) &
        (df['Recommendation'].isin(selected_recommendations))
    ]
    
    # Display filtered results
    display_cols = ['Ticker', 'Company Name', 'Sector', 'Current Price', 
                   'Valuation Score', 'Technical Score', 'Sector Score',
                   'Recommendation']
    
    styled_df = filtered_df[display_cols].style\
        .background_gradient(subset=['Valuation Score', 'Technical Score', 'Sector Score'], cmap='RdYlGn')\
        .applymap(color_recommendation, subset=['Recommendation'])
    
    st.dataframe(styled_df)
    
    # Export functionality
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis Results",
        data=csv,
        file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def map_sector_to_type(sector):
    """Map FMP sectors to SectorType"""
    sector_map = {
        'Technology': SectorType.TECHNOLOGY,
        'Healthcare': SectorType.HEALTHCARE,
        'Financials': SectorType.FINANCIAL,
        'Financial': SectorType.FINANCIAL,
        'Consumer Discretionary': SectorType.CONSUMER_CYCLICAL,
        'Consumer Staples': SectorType.CONSUMER_DEFENSIVE,
        'Industrials': SectorType.INDUSTRIALS,
        'Materials': SectorType.BASIC_MATERIALS,
        'Energy': SectorType.ENERGY,
        'Utilities': SectorType.UTILITIES,
        'Real Estate': SectorType.REAL_ESTATE,
        'Communication Services': SectorType.COMMUNICATION,
        'Telecommunications': SectorType.COMMUNICATION
    }
    return sector_map.get(sector, SectorType.TECHNOLOGY)

def color_recommendation(val):
    """Style function for recommendations"""
    colors = {
        'Strong Buy': 'background-color: #2E7D32; color: white',
        'Buy': 'background-color: #4CAF50; color: white',
        'Hold': 'background-color: #FFC107',
        'Sell': 'background-color: #F44336; color: white',
        'Strong Sell': 'background-color: #B71C1C; color: white'
    }
    return colors.get(val, '')

def display_stock_info(ticker, screener):
    """Show basic info when stock is added"""
    info = screener.get_company_profile(ticker)
    if info:
        sector = info.get('sector', 'Unknown')
        exchange = info.get('exchange', 'Unknown')
        price = info.get('price', 0)
        
        st.sidebar.markdown(f"""
        **Added: {info['companyName']}**
        - Exchange: {exchange}
        - Sector: {sector}
        - Current Price: ${price:.2f}
        """)

def display_sector_analysis(df, screener):
    """Display enhanced sector analysis"""
    st.subheader("Sector Analysis")
    
    # Get unique sectors
    sectors = df['Sector'].unique().tolist()
    
    # Sector selector
    selected_sector = st.selectbox("Select Sector for Analysis", sectors)
    
    # Filter for selected sector
    sector_df = df[df['Sector'] == selected_sector]
    
    if not sector_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector metrics
            st.markdown(f"### {selected_sector} Metrics")
            avg_sector_score = sector_df['Sector Score'].mean()
            st.metric(
                "Average Sector Score",
                f"{avg_sector_score:.1f}",
                delta=f"{avg_sector_score - 50:.1f} vs Market",
                help="Sector-specific performance score"
            )
            
            # Get sector-specific metrics
            sector_type = map_sector_to_type(selected_sector)
            if sector_type in screener.sector_metrics.SECTOR_CONFIGS:
                specific_metrics = screener.sector_metrics.SECTOR_CONFIGS[sector_type]['specific_metrics']
                st.write("Key Sector Metrics:")
                
                # Display available metrics for the sector
                metrics_found = False
                for metric, config in specific_metrics.items():
                    # Convert metric name to the format used in your DataFrame
                    df_metric_name = metric.lower()
                    if df_metric_name in sector_df.columns:
                        metrics_found = True
                        avg_value = sector_df[df_metric_name].mean()
                        threshold = config['threshold']
                        if threshold is not None:
                            st.metric(
                                metric.replace('_', ' ').title(),
                                f"{avg_value:.2f}",
                                delta=f"{avg_value - threshold:.2f} vs Threshold"
                            )
                
                if not metrics_found:
                    st.info(f"No specific metrics available for {selected_sector}")
        
        with col2:
            # Sector risk analysis
            st.markdown("### Risk Analysis")
            if sector_type in screener.sector_metrics.SECTOR_CONFIGS:
                risk_factors = screener.sector_metrics.SECTOR_CONFIGS[sector_type]['risk_factors']
                st.write("Key Risk Factors:")
                
                # Display available risk factors
                risks_found = False
                for risk in risk_factors:
                    risk_col = f"{risk.lower()}_risk"
                    if risk_col in sector_df.columns:
                        risks_found = True
                        risk_value = sector_df[risk_col].mean()
                        st.progress(risk_value/100)
                        st.write(f"{risk.replace('_', ' ')}: {risk_value:.1f}%")
                
                if not risks_found:
                    st.info(f"No risk factors available for {selected_sector}")
        
        # Sector performance visualization
        st.subheader("Sector Performance Distribution")
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=sector_df['Valuation Score'],
            name='Valuation Score',
            boxpoints='all'
        ))
        fig.add_trace(go.Box(
            y=sector_df['Technical Score'],
            name='Technical Score',
            boxpoints='all'
        ))
        fig.add_trace(go.Box(
            y=sector_df['Sector Score'],
            name='Sector Score',
            boxpoints='all'
        ))
        fig.update_layout(
            title=f"{selected_sector} Score Distribution",
            yaxis_title="Score",
            showlegend=True
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for selected sector")

if __name__ == "__main__":
    main()