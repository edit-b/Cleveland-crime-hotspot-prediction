
import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

#App folder and main project folder
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent


def main(source=PROJECT_ROOT):
    #Getting the LSOA codes 
    features = pd.read_csv(source / "Data/crime_features_model.csv")
    codes = set(features["LSOA code"])
    #Getting the original boundary file 
    shapes = list((source / "Data").rglob("LSOA_2021_EW_BGC_V5.shp"))
    if len(shapes) != 1:
        raise ValueError("Expected exactly one LSOA_2021_EW_BGC_V5.shp beneath Data")
    #Keeping only the study areas and their boundaries
    boundary = gpd.read_file(shapes[0])
    boundary = boundary[boundary.LSOA21CD.isin(codes)][["LSOA21CD", "LSOA21NM", "geometry"]].copy()
    #Checking that every study area has a valid boundary
    assert set(boundary.LSOA21CD) == codes
    assert boundary.is_valid.all() and not boundary.is_empty.any()
    #Renaming the columns, converting coordinates to la and long
    boundary = boundary.rename(columns={"LSOA21CD": "lsoa_code", "LSOA21NM": "boundary_name"}).to_crs(4326)
    #Saving the boundaries in the GeoJSON
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    boundary.to_file(ROOT / "data/lsoa_boundaries.geojson", driver="GeoJSON")
    #Loading monthly crime counts
    history = pd.read_csv(source / "Data/crime_monthly_lsoa.csv")
    history = history[history["LSOA code"].isin(codes)].rename(columns={
        "Month": "month", "LSOA code": "lsoa_code", "LSOA name": "lsoa_name", "Local_Authority": "local_authority"})
    #Saving the crime history for line chart
    history.sort_values(["lsoa_code", "month"]).to_csv(ROOT / "data/crime_history.csv", index=False)
    #Summary
    print(json.dumps({"areas": len(boundary), "history_rows": len(history), "crs": "EPSG:4326", "simplification": "none"}, indent=2))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source", nargs="?", type=Path, default=PROJECT_ROOT,
        help="Main project folder containing Data (detected automatically if omitted)",
    )
    main(parser.parse_args().source)
