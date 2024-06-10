
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px

def plot_multi_line(df, title, x_title, y_title, legend_title, price_df=None):
    if price_df is not None:
        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col), row=1, col=1)
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df[col], mode='lines', name=col), row=2, col=1)
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
        st.plotly_chart(fig, use_container_width=True)
        return
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    # update marker color by stock name
    for i, stock in enumerate(df.columns):
        fig.data[i].marker.color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
    st.plotly_chart(fig, use_container_width=True)