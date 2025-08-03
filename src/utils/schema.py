import pandera.pandas as pa  # noqa: F401
from pandera.pandas import Column, DataFrameSchema

# Define schema for input data
california_housing_schema = DataFrameSchema(
    {
        "MedInc": Column(float, nullable=False),
        "HouseAge": Column(float, nullable=False),
        "AveRooms": Column(float, nullable=False),
        "AveBedrms": Column(float, nullable=False),
        "Population": Column(float, nullable=False),
        "AveOccup": Column(float, nullable=False),
        "Latitude": Column(float, nullable=False),
        "Longitude": Column(float, nullable=False),
        "MedHouseVal": Column(float, nullable=False),
    }
)
