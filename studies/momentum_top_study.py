import streamlit as st
from datetime import datetime, timedelta

def run(symbol_benchmark, symbolsDate_dict):
    '## Price History Trend'
    st.info('For optimal viewing, use a PC or tablet or rotate your mobile device to landscape.')
    'This strategy selects stocks for the portfolio based on their past performance ranking among its peers.'
    'The portfolio rebalances monthly at month-end, with new stocks for the following month being selected.'
    st.caption('Check back on the first day of the next month to see the new stock tickers.')
    'The holding period is set for a duration of one month.'

    #pool = st.selectbox('Choose pool of stocks:', ['NASDAQ-100', 'S&P 500'], label_visibility = 'visible')
    'Start creating a customized portfolio by choosing a pool of stocks and then filtering them based on past performance.'
    pool = st.radio('',('VNINDEX', 'NASDAQ-100 (Fast)', 'S&P 500 (Slow)'),horizontal=True,label_visibility='collapsed')

    # select time frame D, W, M
    st.write('**Select the time frame for the strategy:**')
    timeframe = st.radio('',('Monthly','Weekly','Daily'), index=1, horizontal=True,label_visibility='collapsed')

    if timeframe == 'Monthly':
        timeframe = 'M'
    elif timeframe == 'Weekly':
        timeframe = 'W'
    else:
        timeframe = 'D'

    # risk management method - drawdown control
    st.write('**Risk Management Method:**')
    risk_management = st.radio('',('Drawdown Control','None'), index=0, horizontal=True,label_visibility='collapsed')
    target_exposure = 1

    st.caption('If less than 4 filters are needed, put the final number of stocks left in the filters that\'re not needed. For example, if only 3 filters are needed, and you need 10 stocks in your portfolio, just put 10 stocks in your 4th filter as well and choose any month option (It does not matter).')

    # list of lists of # of stocks to stay and # of months to look back
    # filters = [[n,m],[n,m],[n,m],[n,m]]
    filters = []

    # 1st filter
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write('Find the top')
    with col2:
        n = st.number_input('numbers', min_value=1, max_value=250, value=50, label_visibility='collapsed')
    with col3:
        st.write('performers using')
    with col4:
        m = st.number_input('months', min_value=1, max_value=250, value=32, label_visibility='collapsed')
    with col5:
        st.write('months return')
    filters.append([n, m])

    # 2nd filter
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write('Then find the top')
    with col2:
        n = st.number_input('numbers', min_value=1, max_value=250, value=30, label_visibility='collapsed')
    with col3:
        st.write('performers using')
    with col4:
        m = st.number_input('months', min_value=1, max_value=250, value=16, label_visibility='collapsed')
    with col5:
        st.write('months return')
    filters.append([n, m])

    # 2nd filter
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write('Then find the top')
    with col2:
        n = st.number_input('numbers', min_value=1, max_value=250, value=20, label_visibility='collapsed')
    with col3:
        st.write('performers using')
    with col4:
        m = st.number_input('months', min_value=1, max_value=250, value=8, label_visibility='collapsed')
    with col5:
        st.write('months return')
    filters.append([n, m])

    # 3rd filter
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write('Then find the top')
    with col2:
        n = st.number_input('numbers', min_value=1, max_value=250, value=10, label_visibility='collapsed')
    with col3:
        st.write('performers using')
    with col4:
        m = st.number_input('months', min_value=1, max_value=250, value=4, label_visibility='collapsed')
    with col5:
        st.write('months return')
    filters.append([n, m])

    # 4th filter
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write('Then find the top')
    with col2:
        n = st.number_input('numbers', min_value=1, max_value=250, value=2, label_visibility='collapsed')
    with col3:
        st.write('performers using')
    with col4:
        m = st.number_input('months', min_value=1, max_value=250, value=2, label_visibility='collapsed')
    with col5:
        st.write('months return')
    filters.append([n, m])


    'Stocks will be held with equal weight, unless you opt for a one-stock portfolio.'
    'The performance of the strategy is recorded each month for visualization.'
    
    date_input = st.text_input('Strategy started from (YYYY-MM)','2023-01')

    run = st.checkbox('Run the strategy',value=False)

    if run:
        import numpy as np
        import yfinance as yf
        import pandas as pd
        import matplotlib.ticker as ticker
        import matplotlib.pyplot as plt
        from dateutil.relativedelta import relativedelta
        import plotly.express as px
        from utils.stock_utils import get_stock_bars_very_long_term_cached

        @st.cache_data(ttl = 28800,show_spinner="Price data downloading")
        def get_price(pool,date_input, timeframe='M', symbolsDate_dict=None):
            if pool == 'NASDAQ-100 (Fast)':
                ticker_df = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
                tickers = ticker_df.Ticker.to_list()
            elif pool == 'S&P 500 (Slow)':
                ticker_df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
                tickers = ticker_df.Symbol.to_list()
            elif pool == 'VNINDEX':
                tickers =  symbolsDate_dict['symbols']
                return fetch_data(tickers=tickers, resolution=timeframe, use_cache=True)

            tickers = [x.replace('.','-') for x in tickers]
            plot_date = date_input+'-01'
            start_date = datetime.strptime(plot_date,'%Y-%m-%d')-relativedelta(months=13)

            df = yf.download(tickers,start=start_date,interval='1mo')['Adj Close']

            return  df
        
        def get_daily_return(pool, tickers, start_date):
            if pool == 'VNINDEX':
                price_df = fetch_data(tickers=tickers, resolution='D', use_cache=False)
                return_df = price_df.pct_change()
                # transform to daily return
                return return_df

            return yf.download(tickers,start=start_date)['Adj Close'].pct_change()

        def process_stock_data(df, market_data=None):   
            df['date'] = pd.to_datetime(df['tradingDate']).dt.date
            # convert date to datetime
            df['date'] = pd.to_datetime(df['date'])    
            df.set_index('date', inplace=True) 
            return df

        @st.cache_data(ttl = 28800,show_spinner="Price data downloading")
        def fetch_data(
            tickers=[],
            resolution='D',
            use_cache=True
        ):
            price_data = {}
            for ticker in tickers:
                try:
                    stock_data = get_stock_bars_very_long_term_cached(ticker=ticker, stock_type='stock', count_back=300, resolution=resolution, refresh=not use_cache, no_fetch=False)
                    stock_data = process_stock_data(stock_data)
                    price_data[ticker] = stock_data
                except Exception as e:
                    print(e)
                    print(f"Error fetching data for {ticker}")
            
            # for each ticker, add new column to the dataframe with the ticker name and value is close price
            data_df = pd.DataFrame()
            for ticker, data in price_data.items():
                data_df[ticker] = data['close']

            # set the index to be the date
            data_df.index = pd.to_datetime(data_df.index)
            
            return data_df

        def fetch_idx_data(idx, start_date, timeframe='M'):
            if idx == 'VNINDEX':
                market_data = get_stock_bars_very_long_term_cached(ticker='VNINDEX', stock_type='index', resolution=timeframe, count_back=5000)
                market_data['Adj Close'] = market_data['close']

                market_data['date'] = pd.to_datetime(market_data['tradingDate']).dt.date
                market_data['date'] = pd.to_datetime(market_data['date'])

                market_data.set_index('date', inplace=True)

                return market_data

            idx_df = yf.download(idx,start=start_date,interval='1mo')
            return idx_df

        # Tickers
        if pool == 'NASDAQ-100 (Fast)':
            idx = 'QQQ'
        elif pool == 'VNINDEX':
            idx = 'VNINDEX'
        else:
            idx = 'SPY'

        plot_date_str = date_input+'-01'
        plot_date = datetime.strptime(plot_date_str,'%Y-%m-%d')
        start_date = plot_date-relativedelta(months=13)

        df = get_price(pool,date_input, timeframe=timeframe, symbolsDate_dict=symbolsDate_dict)
        
        show_price = st.checkbox('Show price data',value=False)
        
        if show_price:
            st.dataframe(df, use_container_width=True)

        
        @st.cache_data(ttl = 28800,show_spinner=False)
        def return_by_m_month(df,m):
            return df.rolling(m).apply(np.prod)

        def get_top(month):
            top = mom.columns
            for n,ret in months_ret:
                top = ret.loc[month,top].nlargest(n).index
            return top

        def performance(month, strategy_exposure):
            top = get_top(month)
            mon_top = mom.loc[month:,top]
            cur_mon_top = mon_top[:1] # current month performance
            next_mon_top = mon_top[1:2] # next month performance

            # apply transaction cost = 0,1%
            next_mon_top = next_mon_top - 0.001

            # apply tax =  0,1%
            next_mon_top = next_mon_top - 0.001

            portfolio = next_mon_top
            
            # apply strategy exposure
            cash = 1 - strategy_exposure
            portfolio = portfolio * strategy_exposure + cash
            
            perf = portfolio.mean(axis=1).values[0]
            return perf, next_mon_top
        
        # Drawdown Control Algorithm
        def calculate_drawdown(returns_list):
            drawdowns = []
            max_return = 0
            for r in returns_list:
                max_return = max(max_return, r)
                drawdown = (r - max_return) / max_return
                drawdowns.append(drawdown)
            return drawdowns

        def adjust_exposure_for_drawdown(portfolio_values, max_drawdown_threshold, target_exposure):
            drawdown = calculate_drawdown(portfolio_values)
            if drawdown[-1] < -0.3:
                return target_exposure / 3

            if drawdown[-1] < -max_drawdown_threshold:
                return target_exposure / 2  # Reduce exposure by half during significant drawdowns
            return target_exposure

        @st.cache_data(ttl = 28800,show_spinner=False)
        def pd_pct_view(df):
            for c in df.columns:
                df[c] = df[c].apply(lambda x: (f'{x:.2%}'))
            return df

        with st.spinner('Computing...'):
            # Month over month precent change
            mom = df.pct_change()+1
            months_ret = [[n,return_by_m_month(mom,m)] for n,m in filters]

            returns = []
            drawdowns = []
            assets = []
            
            strategy_exposure = target_exposure
            
            for month in mom.index[:-1]:
                if len(returns) > 1 and risk_management == 'Drawdown Control':
                    strategy_exposure = adjust_exposure_for_drawdown(returns, 0.25, target_exposure)
                
                perf, top_df = performance(month, strategy_exposure)
                assets.append(top_df)
                returns.append(perf)
                drawdown = calculate_drawdown(returns)
                drawdowns.append(drawdown[-1])


            # Return of the first 12 months not consider
            returns = pd.Series(returns[12:],index=mom.index[13:])
            # apply index to drawdowns
            drawdowns = pd.Series(drawdowns[12:],index=mom.index[13:])
            
        st.success('Computation success!')
                
        assets_df = pd.concat(assets,axis=0)
        
        # chnage shape of assets_df columns -> ticker, return
        assets_df = assets_df.stack().reset_index()
        # rename columns
        assets_df.columns = ['Date','Ticker','Return']
        
        assets_df = assets_df[assets_df.Date > start_date]

        f'##### See how your strategy has performed since {date_input} vs. Benchmark'
        
        # Cumpound return
        cum_ret = pd.DataFrame(returns[returns.index>=plot_date].cumprod(), columns=['Strategy'])
        idx_df = fetch_idx_data(idx, start_date=start_date, timeframe=timeframe)['Adj Close']
        idx_return = (idx_df.pct_change()+1)[idx_df.index>=plot_date]
        cum_ret[idx] = idx_return.cumprod()
        col1,col2 = st.columns(2)
        col1.metric('Your Cumulative Return', f'{(cum_ret.Strategy[-1]-1):.2%}')
        col2.metric(f'{idx} Cumulative Return', f'{(cum_ret[idx][-1]-1):.2%}')
        
        fig = px.line(cum_ret-1,cum_ret.index,['Strategy',idx],labels={'value':'','variable':'','Date':''})
        fig.update_layout(hovermode="x unified")
        fig.layout.yaxis.tickformat = ',.0%'
        fig.update_traces(hovertemplate = "%{y}")
        st.plotly_chart(fig)
        st.caption('Interact with the chart by hovering over it and selecting an area to enlarge. Double-click to return to full view. Use the legend to hide or show lines.')

        "Drawdown is a measure of the decline from a historical peak in some variable. It"
        "is usually quoted as the percentage between the peak and the subsequent trough."

        drawdown_plot = pd.DataFrame(drawdowns[drawdowns.index>=plot_date],columns=['Drawdown'])

        fig = px.line(drawdown_plot,labels={'value':'Drawdown','index':'Date'},title='Drawdown')
        fig.update_layout(hovermode="x unified")
        fig.layout.yaxis.tickformat = ',.0%'
        fig.update_traces(hovertemplate = "%{y}")
        st.plotly_chart(fig)

        '##### Some statistics for monthly performance:'
        # Stats about strategy and index
        stats_df=pd.DataFrame((returns[returns.index>=plot_date]-1).describe()[1:],columns=['Strategy'])
        stats_df[idx]=(idx_return[idx_return.index>=plot_date]-1).describe()[1:]
        stats_df.index = ['Mean Monthly Return','Standard Deviation', 'Worst Monthly Return','25 Percentile Monthly Return', 'Median Monthly Return','75 Percentile Monthly Return', 'Best Monthly Return']
        st.table(pd_pct_view(stats_df.iloc[[0,1,2,4,6]]))

        diff = returns[returns.index>=plot_date]-idx_return[idx_return.index>=plot_date]
        fig = px.box(diff,x= 0, points='all',title='Distribution of Monthly Alpha',labels={'0':''})
        fig.layout.xaxis.tickformat = '.2%'
        fig.update_traces(hovertemplate = "%{x}")
        st.plotly_chart(fig,use_container_width=True)
        '[Alpha](https://en.wikipedia.org/wiki/Alpha_(finance)) represents the excess return, or the degree by which your strategy outperforms the market.'



        current = mom.index[-2]
        
        # Portfolio returns daily
        top_tickers = get_top(current).to_list()
        daily_return_df = get_daily_return(pool, top_tickers, start_date)
        
        # filter daily return where index > current
        daily_df = daily_return_df[daily_return_df.index>=current]
        
        if len(top_tickers)==1: # daily_df is a Series
            # convert to DataFrame
            daily_df = pd.DataFrame(daily_df)
            
        daily_df['Strategy'] = daily_df.mean(axis=1)    
        
        index_daily_df = fetch_idx_data(idx, start_date=start_date, timeframe='D')
        index_daily_return_df = index_daily_df['Adj Close'].pct_change()
        # filter daily return where index > current
        index_daily_return_mon_df = index_daily_return_df[index_daily_return_df.index>=current]
        daily_df[idx] = index_daily_return_mon_df
        
        daily_df.index = [x.date() for x in daily_df.index]
            
        # Daily chart 
        fig, ax = plt.subplots()
        daily_cumprod = (daily_df[['Strategy',idx]]+1).cumprod()
        tmp = pd.DataFrame([[0,0]],columns=['Strategy',idx],index=[(mom.index[-1]-pd.DateOffset(1)).date()])
        daily_cumprod = pd.concat([tmp,(daily_cumprod-1)])
        fig = px.line(daily_cumprod,daily_cumprod.index,['Strategy',idx],labels={'index':'', 'value':'','variable':''},title='Current Month Cumulative Return')
        fig.update_layout(hovermode="x unified")
        fig.update_traces(hovertemplate = "%{y}")
        fig.layout.yaxis.tickformat = '.2%'
        st.plotly_chart(fig)

        # Yahoo bug
        with st.expander('If the chart appears inaccurate at the beginning of a month, click here for info on a bug in Yahoo Finance.'):
            st.caption('''Real-time financial data from Yahoo Finance may occasionally have discrepancies, 
            such as missing values on the 31st day of a month or mislabeling the first day of a new month 
            as the last day of the previous month. These issues can affect the accuracy of percentage 
            calculations and visualizations, especially when transitioning from one month to the next.
            These issues are typically fixed by Yahoo within a day or two. If the plot appears suspicious, or 
            if data from a previous month is still present when it's already a new month, please check back later for corrected data.''')


        '##### Daily performance of chosen stocks for current month:'
        monthly_df = (mom-1).loc[current:,top_tickers][1:]
        monthly_df.index = ['Current Month total return']
        monthly_df = pd.concat([daily_df[top_tickers],monthly_df])
        monthly_df['Strategy'] = monthly_df.mean(axis=1)
        
        st.table(pd_pct_view(monthly_df))
        
        show_portfolio = st.checkbox('Show assets in portfolio',value=False)
        
        if show_portfolio:
            st.dataframe(assets_df, use_container_width=True)
        
        live_trading = st.checkbox('Live Trading',value=False)
        if live_trading:
            st.write("### Live Trading")
            
            last_month = mom.index[-2]
            last_top = get_top(last_month)
            
            next_month = mom.index[-1]
            next_top = get_top(next_month)
            
            st.write(f'Portfolio turnover for **{next_month}**')
            # sell assets
            for col in last_top:
                if col not in next_top:
                    st.write(f'📉 Sell **{col}**')
            
            for col in next_top:
                if col not in last_top:
                    st.write(f'📈 Buy **{col}**')
                    
            st.write('Current Portfolio:')
            st.table(next_top)