# CivicFix AI — Group 8 deliverables

## Deliverables
- `CivicFix_AI_Final_Report.pdf` / `.docx`: final updated research report
- `run_analysis.py`: reproducible NYC recurrence evaluation and Istanbul rainfall/DEM analysis
- `results.json`: exact results from the supplied data
- `nyc_rain_complaints.png`, `istanbul_rain.png`, `istanbul_terrain.png`: report figures
- `istanbul_rain_oct2026.csv`: local rainfall export

## Run locally
Place `run_analysis.py` in a folder `CivicFix_Final` next to the supplied original inputs: `nyc311_infra_2025.csv`, `nyc_centralpark_2025(1).csv`, `istanbul_daily_rain_oct2026.csv`, and `rasters_COP30.tar`. Run `pip install pandas numpy matplotlib scikit-learn rasterio scipy` then `python run_analysis.py`.

## Scope and caveats
NYC model is evaluated on NYC only. Istanbul analysis is exploratory terrain screening and single-coordinate rainfall, **not** a validated Istanbul flood predictor. The rainfall coordinate and DEM footprint do not overlap. No Turkish LLM experiment, Sentinel-1 validation or cross-city predictive comparison is claimed.
