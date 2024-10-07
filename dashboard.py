import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
        return pd.DataFrame(data) if data else pd.DataFrame()
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()

def main():
    st.title("Brothers Karamazov RAG-Chatbot Monitoring Dashboard")

    # Load data
    interactions_df = load_data('logs/interaction_log.json')
    feedback_df = load_data('logs/feedback_log.json')

    # 1. Interactions over time
    st.subheader("1. Interactions Over Time")
    if not interactions_df.empty:
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        fig_interactions = px.line(interactions_df, x='timestamp', y=interactions_df.index, title="Cumulative Interactions Over Time")
        st.plotly_chart(fig_interactions)
    else:
        st.write("No interaction data available.")

    # 2. User Feedback Distribution
    st.subheader("2. User Feedback Distribution")
    if not feedback_df.empty:
        feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
        feedback_counts = feedback_df['is_helpful'].value_counts()
        fig_feedback = px.pie(values=feedback_counts.values, names=feedback_counts.index, title="User Feedback Distribution")
        st.plotly_chart(fig_feedback)
    else:
        st.write("No feedback data available.")

    # 3. Average Evaluation Scores Over Time
    st.subheader("3. Average Evaluation Scores Over Time")
    if not interactions_df.empty:
        interactions_df['date'] = interactions_df['timestamp'].dt.date
        interactions_df['evaluation_score'] = interactions_df['evaluation'].apply(
            lambda x: x.get('relevance') if isinstance(x, dict) and 'relevance' in x else None
        )
        interactions_df['evaluation_score'] = pd.to_numeric(interactions_df['evaluation_score'], errors='coerce')
        avg_scores = interactions_df.groupby('date')['evaluation_score'].mean().reset_index()

        if not avg_scores.empty:
            fig_avg_scores = go.Figure()
            fig_avg_scores.add_trace(go.Scatter(x=avg_scores['date'], y=avg_scores['evaluation_score'], mode='lines+markers', name='Average Score'))
            fig_avg_scores.update_layout(title="Average Evaluation Scores Over Time", xaxis_title="Date", yaxis_title="Average Score")
            st.plotly_chart(fig_avg_scores)
        else:
            st.write("No evaluation scores available to display.")
    else:
        st.write("No interaction data available for evaluation scores.")

    # 4. Query Length Distribution
    st.subheader("4. Query Length Distribution")
    if not interactions_df.empty:
        interactions_df['query_length'] = interactions_df['query'].str.len()
        fig_query_length = px.histogram(interactions_df, x='query_length', title="Distribution of Query Lengths")
        fig_query_length.update_layout(bargap=0.3)
        st.plotly_chart(fig_query_length)
    else:
        st.write("No query data available for length distribution.")

    # 5. Top 10 Most Common Words in Queries
    st.subheader("5. Top 10 Most Common Words in Queries")
    if not interactions_df.empty:
        all_words = ' '.join(interactions_df['query']).lower().split()
        word_freq = pd.Series(all_words).value_counts().head(10)
        fig_word_freq = px.bar(x=word_freq.index, y=word_freq.values, title="Top 10 Most Common Words in Queries")
        st.plotly_chart(fig_word_freq)
    else:
        st.write("No query data available for word frequency analysis.")

if __name__ == "__main__":
    main()