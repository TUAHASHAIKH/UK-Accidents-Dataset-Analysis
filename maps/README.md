# Pre-generated Map Visualizations

This folder contains pre-generated HTML map files to improve dashboard performance.

## Files
- `density_heatmap.html` - UK road accidents density heatmap
- `urban_vs_rural_map.html` - Urban vs Rural classification by local authority
- `comprehensive_density.html` - Comprehensive analysis density map (optional)

## Generating Maps

If the map files don't exist or need to be regenerated, run:

```bash
python generate_maps.py
```

This script will:
1. Load the UK accidents dataset
2. Generate all map visualizations
3. Save them as static HTML files in this folder

The dashboard will automatically load these pre-generated maps instead of creating them at runtime, significantly improving page load speed.

## Note
These HTML files are large (~5-20MB each) and are excluded from git via .gitignore.
You need to run `generate_maps.py` locally before running the dashboard.
