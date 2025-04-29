import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load job data
df = pd.read_csv('jobs_data.csv')

# Vectorize job titles and descriptions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['job_title'] + ' ' + df['description'])

def search_jobs(city, keywords, top_n=5):
    city_jobs = df[df['location'].str.contains(city, case=False, na=False)]
    if city_jobs.empty:
        return pd.DataFrame()
    query_vec = vectorizer.transform([keywords])
    cosine_sim = cosine_similarity(query_vec, X[city_jobs.index])
    top_indices = cosine_sim.argsort()[0][-top_n:][::-1]
    return city_jobs.iloc[top_indices]

# Streamlit UI
st.title("🎯 Fresher Job Search (India)")

city_input = st.text_input("Enter City Name (e.g., Mumbai, Bangalore, Pune)", "")
keywords_input = st.text_input("Enter Job Title / Keywords (e.g., Python Developer, Data Analyst)", "")

if st.button("Search"):
    if city_input and keywords_input:
        results = search_jobs(city_input, keywords_input)
        if results.empty:
            st.warning("No jobs found! Try another city or keywords.")
        else:
            for idx, row in results.iterrows():
                st.subheader(row['job_title'])
                st.write(f"**Company:** {row['company']}")
                st.write(f"**Location:** {row['location']}")
                st.write(f"[Apply Here]({row['apply_link']})")
                st.write("---")
    else:
        st.error("Please enter both City and Keywords to search.")
