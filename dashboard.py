import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def main():
    st.title("Brothers Karamazov RAG-Chatbot Monitoring Dashboard")

    # Load data
    interactions_df = load_data('interaction_log.json')
    feedback_df = load_data('feedback_log.json')

    # Convert timestamp strings to datetime objects
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
    feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])

    # 1. Interactions over time
    st.subheader("1. Interactions Over Time")
    fig_interactions = px.line(interactions_df, x='timestamp', y=interactions_df.index, title="Cumulative Interactions Over Time")
    st.plotly_chart(fig_interactions)

    # 2. User Feedback Distribution
    st.subheader("2. User Feedback Distribution")
    feedback_counts = feedback_df['is_helpful'].value_counts()
    fig_feedback = px.pie(values=feedback_counts.values, names=feedback_counts.index, title="User Feedback Distribution")
    st.plotly_chart(fig_feedback)

    # 3. Average Evaluation Scores Over Time
    st.subheader("3. Average Evaluation Scores Over Time")
    interactions_df['date'] = interactions_df['timestamp'].dt.date

    # Extract evaluation relevance score from the evaluation object
    interactions_df['evaluation_score'] = interactions_df['evaluation'].apply(
        lambda x: x.get('relevance') if isinstance(x, dict) and 'relevance' in x else None
    )
    interactions_df['evaluation_score'] = pd.to_numeric(interactions_df['evaluation_score'], errors='coerce')

    avg_scores = interactions_df.groupby('date')['evaluation_score'].mean().reset_index()

    if not avg_scores.empty:
        # Create a figure with both line and scatter plots
        fig_avg_scores = go.Figure()
        
        # Add line plot
        fig_avg_scores.add_trace(go.Scatter(
            x=avg_scores['date'], 
            y=avg_scores['evaluation_score'], 
            mode='lines',
            name='Average Score'
        ))
        
        # Add scatter plot
        fig_avg_scores.add_trace(go.Scatter(
            x=avg_scores['date'], 
            y=avg_scores['evaluation_score'], 
            mode='markers',
            name='Data Points',
            marker=dict(size=8)
        ))
        
        # Update layout
        fig_avg_scores.update_layout(
            title="Average Evaluation Scores Over Time",
            xaxis_title="Date",
            yaxis_title="Average Score",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_avg_scores)
    else:
        st.write("No evaluation scores available to display.")

    # 4. Query Length Distribution
    st.subheader("4. Query Length Distribution")
    interactions_df['query_length'] = interactions_df['query'].str.len()
    fig_query_length = px.histogram(interactions_df, x='query_length', title="Distribution of Query Lengths")
    # {{ edit_3 }} Add more space between bars
    fig_query_length.update_layout(bargap=0.3)  # Increased bargap for better spacing
    st.plotly_chart(fig_query_length)

    # 5. Top 10 Most Common Words in Queries
    st.subheader("5. Top 10 Most Common Words in Queries")
    all_words = ' '.join(interactions_df['query']).lower().split()
    word_freq = pd.Series(all_words).value_counts().head(10)
    fig_word_freq = px.bar(x=word_freq.index, y=word_freq.values, title="Top 10 Most Common Words in Queries")
    st.plotly_chart(fig_word_freq)

if __name__ == "__main__":
    main()