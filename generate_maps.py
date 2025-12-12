"""
Generate static map visualizations to avoid runtime plotting
Run this script once to create the map HTML files
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq

print("Loading data...")
df = pd.read_parquet('UK_Accidents_Fully_Cleaned.parquet')

print(f"Loaded {len(df):,} rows")

# Create maps directory if it doesn't exist
import os
os.makedirs('maps', exist_ok=True)

# ============ MAP 1: Density Heatmap ============
print("\nGenerating density heatmap...")
sample_df = df.sample(n=min(20000, len(df)), random_state=42)

fig1 = px.density_mapbox(
    sample_df,
    lat='Latitude',
    lon='Longitude',
    radius=8,
    center=dict(lat=54.5, lon=-3.5),
    zoom=5,
    mapbox_style="open-street-map",
    title="UK Road Accidents Density Heatmap - Hotspots Clearly Visible",
    height=700
)

fig1.write_html('maps/density_heatmap.html')
print("✓ Saved: maps/density_heatmap.html")

# ============ MAP 2: Urban vs Rural Map ============
print("\nGenerating urban vs rural map...")

# Calculate predominant area type for each local authority
authority_summary = df.groupby('Local_Authority_(District)').agg({
    'Urban_or_Rural_Area': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown',
    'Latitude': 'mean',
    'Longitude': 'mean'
}).reset_index()

authority_summary.columns = ['Local_Authority_(District)', 'Predominant_Type', 'Latitude', 'Longitude']

# Add total count
authority_summary['Total'] = df.groupby('Local_Authority_(District)').size().values

# Split into urban and rural
urban_data = df[df['Urban_or_Rural_Area'] == 'Urban'].groupby('Local_Authority_(District)').size().reset_index(name='Urban')
rural_data = df[df['Urban_or_Rural_Area'] == 'Rural'].groupby('Local_Authority_(District)').size().reset_index(name='Rural')

authority_map_data = authority_summary.merge(urban_data, on='Local_Authority_(District)', how='left')
authority_map_data = authority_map_data.merge(rural_data, on='Local_Authority_(District)', how='left')
authority_map_data = authority_map_data.fillna(0)

fig2 = px.scatter_mapbox(
    authority_map_data,
    lat='Latitude',
    lon='Longitude',
    size='Total',
    color='Predominant_Type',
    hover_name='Local_Authority_(District)',
    hover_data={
        'Total': True,
        'Urban': True,
        'Rural': True,
        'Latitude': False,
        'Longitude': False,
        'Predominant_Type': False
    },
    color_discrete_map={'Urban': '#FF6B6B', 'Rural': '#4ECDC4'},
    size_max=50,
    zoom=5,
    center=dict(lat=54.5, lon=-3.5),
    mapbox_style="open-street-map",
    title="UK Accidents: Urban vs Rural by Local Authority (Bubble Size = Total Accidents)",
    height=800
)

fig2.update_layout(
    legend=dict(
        title="Area Type",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig2.write_html('maps/urban_vs_rural_map.html')
print("✓ Saved: maps/urban_vs_rural_map.html")

# ============ MAP 3-6: Urban vs Rural Detailed Comparison (Split into 4 charts) ============
print("\nGenerating urban vs rural detailed comparison charts (4 separate files)...")

# Chart 1: Accident Count by Area Type
print("  - Chart 1: Accident Count...")
area_counts = df['Urban_or_Rural_Area'].value_counts()
fig3 = go.Figure(data=[
    go.Bar(x=area_counts.index, y=area_counts.values, 
           marker_color=['#FF6B6B', '#4ECDC4'],
           text=area_counts.values,
           textposition='auto',
           texttemplate='%{text:,.0f}')
])
fig3.update_layout(
    title='Accident Count by Area Type',
    xaxis_title='Area Type',
    yaxis_title='Number of Accidents',
    height=400,
    showlegend=False
)
fig3.write_html('maps/urban_rural_chart1_count.html')
print("✓ Saved: maps/urban_rural_chart1_count.html")

# Chart 2: Urban Severity Distribution
print("  - Chart 2: Urban Severity Distribution...")
urban_severity = df[df['Urban_or_Rural_Area'] == 'Urban']['Accident_Severity'].value_counts()
fig4 = go.Figure(data=[
    go.Pie(labels=urban_severity.index, values=urban_severity.values,
           marker_colors=['lightcoral', 'orange', 'darkred'],
           hole=0.3,
           textinfo='label+percent',
           textposition='auto')
])
fig4.update_layout(
    title='Severity Distribution in Urban Areas',
    height=400
)
fig4.write_html('maps/urban_rural_chart2_urban_severity.html')
print("✓ Saved: maps/urban_rural_chart2_urban_severity.html")

# Chart 3: Rural Severity Distribution
print("  - Chart 3: Rural Severity Distribution...")
rural_severity = df[df['Urban_or_Rural_Area'] == 'Rural']['Accident_Severity'].value_counts()
fig5 = go.Figure(data=[
    go.Pie(labels=rural_severity.index, values=rural_severity.values,
           marker_colors=['lightblue', 'orange', 'darkred'],
           hole=0.3,
           textinfo='label+percent',
           textposition='auto')
])
fig5.update_layout(
    title='Severity Distribution in Rural Areas',
    height=400
)
fig5.write_html('maps/urban_rural_chart3_rural_severity.html')
print("✓ Saved: maps/urban_rural_chart3_rural_severity.html")

# Chart 4: Casualties Comparison (sampled for performance)
print("  - Chart 4: Casualties Comparison (sampling for optimization)...")
urban_sample = df[df['Urban_or_Rural_Area'] == 'Urban']['Number_of_Casualties'].sample(
    n=min(50000, len(df[df['Urban_or_Rural_Area'] == 'Urban'])), 
    random_state=42
)
rural_sample = df[df['Urban_or_Rural_Area'] == 'Rural']['Number_of_Casualties'].sample(
    n=min(50000, len(df[df['Urban_or_Rural_Area'] == 'Rural'])), 
    random_state=42
)

fig6 = go.Figure()
fig6.add_trace(go.Box(y=urban_sample, name='Urban', marker_color='#FF6B6B'))
fig6.add_trace(go.Box(y=rural_sample, name='Rural', marker_color='#4ECDC4'))
fig6.update_layout(
    title='Casualties Comparison: Urban vs Rural',
    yaxis_title='Number of Casualties',
    height=400,
    showlegend=True
)
fig6.write_html('maps/urban_rural_chart4_casualties.html')
print("✓ Saved: maps/urban_rural_chart4_casualties.html")

print("\n✅ All maps generated successfully!")
print("Maps saved in 'maps/' folder and can be loaded instantly in the dashboard.")
