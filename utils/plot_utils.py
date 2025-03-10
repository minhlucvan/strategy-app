
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import pandas as pd
import plotly.subplots as sp

def plot_multi_line(df, title="", x_title="", y_title="", legend_title="", price_df=None):
    df = df.copy().astype(float)  # Ensure numerical dtype
    df = df.where(df.notna(), None)  # Convert NaN to None
    if price_df is not None:
        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode='lines', name=col, connectgaps=False  # Prevent interpolation
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=price_df.index, y=price_df[col], mode='lines', name=col, connectgaps=False  # Prevent interpolation
            ), row=2, col=1)
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
        st.plotly_chart(fig, use_container_width=True)
        return
    
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode='lines', name=col, connectgaps=False  # Prevent interpolation
        ))
    
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    
    # Update marker color by stock name
    for i, stock in enumerate(df.columns):
        fig.data[i].marker.color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
    
    # update_traces
    fig.update_traces(connectgaps=False)  # Prevent interpolation
    
    st.plotly_chart(fig, use_container_width=True)


def plot_multi_bar(df, title="", x_title="", y_title="", legend_title="", price_df=None):
    if price_df is not None:
        price_df = price_df.copy()
        min_event_date = df.index.min()
        price_df = price_df[price_df.index >= min_event_date]
        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        for col in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df[col], name=col), row=1, col=1)
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df[col], mode='lines', name=col), row=2, col=1)
        
        # Adjust y-axis range of scatter plot to accommodate bars
        fig.update_yaxes(range=[0, max(df.max().max(), price_df.max().max()) * 1.1], row=2, col=1)
        
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
        st.plotly_chart(fig, use_container_width=True)
        return
    
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df[col], name=col))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)


def plot_multi_scatter(df, title, x_title="", y_title="", legend_title="", price_df=None):
    if price_df is not None:
        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='markers', name=col), row=1, col=1)
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df[col], mode='lines', name=col), row=2, col=1)
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
        st.plotly_chart(fig, use_container_width=True)
        return
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter
                        (x=df.index, y=df[col], mode='markers', name=col))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)

def plot_single_line(df, title="", x_title="", y_title="", legend_title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df, mode='lines', name=legend_title))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)

def plot_single_line_with_price(df, title, x_title, y_title, legend_title, price_df, price_legend_title=None):
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    fig.add_trace(go.Scatter(x=df.index, y=df, mode='lines', name=legend_title), row=1, col=1)
    fig.add_trace(go.Scatter
                    (x=price_df.index, y=price_df, mode='lines', name=price_legend_title), row=2, col=1)
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)
    
def plot_single_bar(df, title="", x_title="", y_title="", legend_title="", price_df=None, line_df=None):
    if line_df is not None:
        return plot_single_bar_with_line(df, title, x_title, y_title, legend_title, line_df)
    if price_df is not None:
        return plot_single_bar_with_price(df, title, x_title, y_title, legend_title, price_df)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df, name=legend_title))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)
    
def plot_single_bar_with_price(df, title, x_title, y_title, legend_title, price_df):
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    fig.add_trace(go.Bar(x=df.index, y=df, name=legend_title), row=1, col=1)
    fig.add_trace(go.Scatter
                    (x=price_df.index, y=price_df, mode='lines', name=legend_title), row=2, col=1)
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)
    
def plot_single_bar_with_line(df, title, x_title, y_title, legend_title, line_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df, name=legend_title))
    fig.add_trace(go.Scatter
                    (x=line_df.index, y=line_df, mode='lines', name=legend_title))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)
    
def plot_snapshot(df, title, x_title, y_title, legend_title, sorted=False):
    display_series = df.iloc[-1] if not sorted else df.iloc[-1].sort_values(ascending=False)
    # plot bar chart fe each stock
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.columns, y=display_series, name='PEB Ratio'))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    # update marker color by stock name
    for i, stock in enumerate(df.columns):
        fig.data[0].marker.color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
    st.plotly_chart(fig, use_container_width=True)
    


def plot_events(price_series, events_series, label=None, annotate_sign=False):
    start_date = events_series.index.min()
    price_series = price_series[price_series.index >= start_date]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_series.index, y=price_series, mode='lines', name='Price'))
    max_price = price_series.max()
    min_price = price_series.min()

    # Add horizontal line for events, annotation_text = event
    for index in events_series.index:
        events = events_series[index]
        
        events = events if isinstance(events, pd.Series) else pd.Series([events])
        for event in events:
            # Determine the color based on price movement
            if annotate_sign and price_series[index] >= price_series.shift(1)[index]:
                line_color = "green"
            elif annotate_sign and price_series[index] < price_series.shift(1)[index]:
                line_color = "red"
            else:
                line_color = "blue"
            
            # Add horizontal line, x = index, y = max_price
            fig.add_shape(type="line",
                x0=index, y0=max_price, x1=index, y1=min_price,
                line=dict(color=line_color, width=1))
            
            # Add annotation
            fig.add_annotation(x=index, y=max_price, text=label if label is not None else event, showarrow=False, yshift=10)
    
    st.plotly_chart(fig)


def plot_double_side_bars(dataframe, top_bar_col, bottom_bar_col, line_col, top_bar_name, bottom_bar_name, line_name, title):
    fig = go.Figure()

    # Add bottom_bar_col as negative bars
    fig.add_trace(go.Bar(
        x=dataframe.index,
        y=dataframe[bottom_bar_col],
        name=bottom_bar_name,
        marker_color='red'
    ))

    # Add top_bar_col as positive bars
    fig.add_trace(go.Bar(
        x=dataframe.index,
        y=dataframe[top_bar_col],
        name=top_bar_name,
        marker_color='green'
    ))
    
    # Add line_col as line
    fig.add_trace(go.Scatter(
        x=dataframe.index,
        y=dataframe[line_col],
        mode='lines',
        name=line_name,
        line=dict(color='blue', width=2)
    ))

    # Update layout for better visualization
    fig.update_layout(
        barmode='relative',
        title=title,
        xaxis_title='Date',
        yaxis_title='Volume',
        yaxis=dict(
            title='Volume',
            titlefont_size=16,
            tickfont_size=14,
        ),
        xaxis=dict(
            title='Date',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
def plot_cash_and_assets(df, cash_col, assets_col):
    cash_df = df[[cash_col]]
    assets_df = df[[assets_col]]
    
    # Plot stacked bar chart for cash and assets values
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df[cash_col], name='Cash', marker_color='blue'))
    fig.add_trace(go.Bar(x=df.index, y=df[assets_col], name='Assets', marker_color='green'))
    
    # Update layout for better visualization
    fig.update_layout(
        barmode='stack',
        title='Cash and Assets Values',
        xaxis_title='Date',
        yaxis_title='Value',
        yaxis=dict(
            title='Value',
            titlefont_size=16,
            tickfont_size=14,
        ),
        xaxis=dict(
            title='Date',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
