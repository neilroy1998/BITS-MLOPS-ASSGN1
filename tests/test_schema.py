import pandas as pd
import pytest

from src.utils.schema import california_housing_schema


def test_schema_valid_passes():
    df = pd.DataFrame(
        {
            "MedInc": [3.5, 2.1],
            "HouseAge": [25.0, 40.0],
            "AveRooms": [6.0, 4.5],
            "AveBedrms": [1.0, 1.2],
            "Population": [1000.0, 500.0],
            "AveOccup": [2.5, 3.0],
            "Latitude": [34.05, 37.77],
            "Longitude": [-118.25, -122.42],
            "MedHouseVal": [2.5, 3.0],
        }
    )

    # Should not raise
    california_housing_schema.validate(df)


def test_schema_invalid_fails():
    df = pd.DataFrame(
        {
            "MedInc": [3.5],
            # Missing 'HouseAge'
            "AveRooms": [6.0],
            "AveBedrms": [1.0],
            "Population": [1000],
            "AveOccup": [2.5],
            "Latitude": [34.05],
            "Longitude": [-118.25],
            "MedHouseVal": [2.5],
        }
    )

    with pytest.raises(Exception):
        california_housing_schema.validate(df)
