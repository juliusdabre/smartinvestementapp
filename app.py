import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
house_price = pd.read_excel("Region Charts Master.xlsx", sheet_name="House Price Sa3")
vacancy = pd.read_excel("Region Charts Master.xlsx", sheet_name="Vacancy Rates")
rents = pd.read_excel("Region Charts Master.xlsx", sheet_name="House Rents")
seifa = pd.read_excel("Region Charts Master.xlsx", sheet_name="Average SEIFA")
ai = pd.read_excel("Region Charts Master.xlsx", sheet_name="AI Impact")
jobs = pd.read_excel("Region Charts Master.xlsx", sheet_name="Jobs By Category")
suburbs = pd.read_excel("Region Charts Master.xlsx", sheet_name="Suburbs Per SA3")

st.set_page_config(page_title="Smart Investment Dashboard", layout="wide")
st.title("📊 Smart Investment Insights Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
regions = sorted(list(set(house_price['SA3'].dropna())))
selected_regions = st.sidebar.multiselect("Select SA3 Region(s)", regions, default=regions[:5])

# Filter data
house_filtered = house_price[house_price['SA3'].isin(selected_regions)]
vacancy_filtered = vacancy[vacancy['Region'].isin(selected_regions)]
rents_filtered = rents[rents['Region'].isin(selected_regions)]
seifa_filtered = seifa[seifa['Region'].isin(selected_regions)]
ai_filtered = ai[ai['Region'].isin(selected_regions)]
jobs_filtered = jobs[jobs['Row Labels'].isin(selected_regions)]
suburbs_filtered = suburbs[suburbs['SA3_NAME21'].isin(selected_regions)]

# Merge key datasets for composite index
combined = seifa_filtered[['Region', 'SEIFA Score']].merge(
    ai_filtered[['Region', 'AI Impact Score']], on='Region', how='outer')
combined = combined.merge(
    jobs_filtered[['Row Labels', 'Concentration Risk']], left_on='Region', right_on='Row Labels', how='outer')
combined = combined.merge(
    suburbs[['SA3_NAME21', 'Latitude', 'Longitude']], left_on='Region', right_on='SA3_NAME21', how='left')
combined['Investment Score'] = (
    (combined['SEIFA Score'].rank(ascending=True) + 
     combined['AI Impact Score'].rank(ascending=True) + 
     combined['Concentration Risk'].rank(ascending=False)) / 3
)
combined = combined.sort_values("Investment Score")

# Display Data Sections
st.header("🏠 House Price Trends")
month_columns = [col for col in house_filtered.columns if col not in ['SA4', 'SA3']]
house_melted = house_filtered.melt(id_vars=['SA3'], value_vars=month_columns, var_name='Month', value_name='Median Price')
fig_price = px.line(house_melted, x='Month', y='Median Price', color='SA3', title="House Price Trend")
st.plotly_chart(fig_price, use_container_width=True)

st.header("📉 Vacancy & Rent Trends")
col1, col2 = st.columns(2)
with col1:
    fig_vacancy = px.line(vacancy_filtered, x='Month', y='Vacancy Rate', color='Region', title="Vacancy Rate")
    st.plotly_chart(fig_vacancy, use_container_width=True)
with col2:
    fig_rent = px.line(rents_filtered, x='Month', y='Median Weekly Rent', color='Region', title="Rent Trend")
    st.plotly_chart(fig_rent, use_container_width=True)

st.header("📊 SEIFA Scores")
st.dataframe(seifa_filtered[['Region', 'SEIFA Score']], use_container_width=True)

st.header("🧠 AI Impact Scores")
st.dataframe(ai_filtered[['Region', 'AI Impact Score']], use_container_width=True)

st.header("💼 Job Concentration Risk")
st.dataframe(jobs_filtered[['Row Labels', 'MAX', 'Concentration Risk']], use_container_width=True)

st.header("⭐ Composite Investment Score")
st.dataframe(combined[['Region', 'SEIFA Score', 'AI Impact Score', 'Concentration Risk', 'Investment Score']].dropna(), use_container_width=True)

fig_index = px.bar(combined.dropna(), x='Region', y='Investment Score', title="Composite Investment Score by Region", color='Investment Score')
st.plotly_chart(fig_index, use_container_width=True)

# Geo Map View
st.header("🗺️ Investment Map View")
map_data = combined.dropna(subset=['Latitude', 'Longitude'])
fig_map = px.scatter_mapbox(map_data, lat="Latitude", lon="Longitude", size="Investment Score",
                            color="Investment Score", hover_name="Region", zoom=4,
                            mapbox_style="carto-positron", title="Map: Investment Scores by Region")
st.plotly_chart(fig_map, use_container_width=True)

# Suburb-level Clustering
st.header("🔍 Investment Clustering by Region")
cluster_data = combined[['SEIFA Score', 'AI Impact Score', 'Concentration Risk']].dropna()
scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_data)
kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled)
combined['Cluster'] = kmeans.labels_
fig_cluster = px.scatter_matrix(combined, dimensions=['SEIFA Score', 'AI Impact Score', 'Concentration Risk'],
                                color='Cluster', title="Clustering Analysis")
st.plotly_chart(fig_cluster, use_container_width=True)

# Download Section
st.subheader("📥 Download Filtered Data")
st.download_button("Download House Price Data", data=house_filtered.to_csv(index=False), file_name="house_price_data.csv")
st.download_button("Download Vacancy Data", data=vacancy_filtered.to_csv(index=False), file_name="vacancy_data.csv")
st.download_button("Download Rent Data", data=rents_filtered.to_csv(index=False), file_name="rent_data.csv")
st.download_button("Download SEIFA Data", data=seifa_filtered.to_csv(index=False), file_name="seifa_data.csv")
st.download_button("Download AI Impact Data", data=ai_filtered.to_csv(index=False), file_name="ai_impact_data.csv")
st.download_button("Download Job Risk Data", data=jobs_filtered.to_csv(index=False), file_name="job_risk_data.csv")
st.download_button("Download Investment Score", data=combined.to_csv(index=False), file_name="investment_score.csv")
