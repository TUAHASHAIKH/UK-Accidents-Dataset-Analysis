# UK Accidents Dataset Analysis

An interactive data visualization dashboard for analyzing UK road accidents data. This project provides comprehensive insights into accident patterns, severity, and geographical distributions across urban and rural areas.

## Features

- **Interactive Streamlit Dashboard**: Explore accident data through multiple visualization tabs
- **Density Heatmaps**: Visualize accident hotspots across the UK
- **Urban vs Rural Analysis**: Compare accident patterns between urban and rural areas
- **Severity Analysis**: Deep dive into accident severity metrics
- **Casualty Statistics**: Analyze casualty data across different regions
- **Pre-generated Maps**: HTML maps for quick viewing without running the dashboard

## Prerequisites

- Python 3.8 or higher
- Windows OS (batch files included for Windows users)
- Internet connection for initial setup

## Quick Start Guide

### Option 1: Download ZIP (Recommended for beginners)

1. **Download the Project**
   - Click the green `Code` button at the top of this repository
   - Select `Download ZIP`
   - Extract the ZIP file to your desired location

2. **Run Setup**
   - Navigate to the extracted folder
   - Double-click `Setup.bat`
   - Wait for all dependencies to install (this may take a few minutes)

3. **Launch Dashboard**
   - Double-click `Run_Dashboard.bat`
   - Your default browser will automatically open the dashboard
   - Start exploring the visualizations!

### Option 2: Git Clone

```bash
git clone https://github.com/TUAHASHAIKH/UK-Accidents-Dataset-Analysis.git
cd UK-Accidents-Dataset-Analysis
```

Then follow steps 2-3 from Option 1 above.

## Manual Setup (Advanced Users)

If you prefer manual installation:

```bash
# Install required packages
pip install -r requirements.txt

# Generate maps (optional - pre-generated maps are included)
python generate_maps.py

# Run the dashboard
streamlit run streamlit_dashboard.py
```

## Project Structure

```
UK-Accidents-Dataset-Analysis/
│
├── streamlit_dashboard.py               # Main dashboard application
├── generate_maps.py                     # Script to generate map visualizations
├── UK_Accidents_Fully_Cleaned.parquet   # Cleaned accidents dataset
├── UK_Vehicles_Fully_Cleaned.parquet    # Cleaned vehicles dataset
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── Setup.bat                            # Automated setup script
├── Run_Dashboard.bat                    # Dashboard launcher script
│
└── maps/                                # Pre-generated HTML maps
    ├── README.md
    ├── density_heatmap.html
    ├── urban_vs_rural_map.html
    ├── urban_rural_chart1_count.html
    ├── urban_rural_chart2_urban_severity.html
    ├── urban_rural_chart3_rural_severity.html
    └── urban_rural_chart4_casualties.html
```

## Dashboard Features

### Navigation Tabs:

1. **Home**: Overview and introduction to the analysis
2. **Density Heatmap**: Interactive heatmap showing accident concentration
3. **Urban vs Rural Map**: Geographical comparison of accident distributions
4. **Urban vs Rural Analysis**: Statistical charts comparing urban and rural accidents
5. **About**: Project information and data sources

## Troubleshooting

### Setup.bat not working?
- Make sure you have Python installed and added to PATH
- Run Command Prompt as Administrator
- Manually install: `pip install -r requirements.txt`

### Dashboard won't open?
- Check if port 8501 is already in use
- Manually run: `streamlit run streamlit_dashboard.py`
- Check your firewall settings

### Missing data errors?
- Ensure `UK_Accidents_Fully_Cleaned.parquet` is in the project folder
- Re-download the project if files are corrupted

## Dataset

The project uses a cleaned and processed UK road accidents dataset stored in Parquet format for optimal performance. The dataset includes:
- Accident locations (latitude/longitude)
- Severity levels
- Urban/Rural classification
- Casualty information
- Temporal data

## Technologies Used

- **Python 3.x**: Core programming language
- **Streamlit**: Interactive web dashboard framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Folium**: Map visualizations
- **PyArrow**: Parquet file handling

## Requirements

All dependencies are listed in `requirements.txt`:
- streamlit
- pandas
- plotly
- folium
- pyarrow
- streamlit-folium

## Usage Tips

- **First Time Users**: Use the batch files for easiest setup
- **Data Scientists**: Explore `generate_maps.py` for map generation logic
- **Developers**: Check `streamlit_dashboard.py` for dashboard customization
- **Quick Preview**: Open HTML files in `/maps` folder directly in your browser

## Contributing

Feel free to fork this repository and submit pull requests for any improvements!

## Contact

For questions or issues, please open an issue on GitHub.

## Note

The dataset file (`UK_Accidents_Fully_Cleaned.parquet`) is approximately 57 MB. Ensure you have adequate storage and bandwidth when cloning or downloading.

---

**Happy Analyzing!**
