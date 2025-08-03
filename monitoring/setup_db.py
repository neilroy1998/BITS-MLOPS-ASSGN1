# monitoring/setup_db.py
import sqlite3
from pathlib import Path

# Define the path for the database in the monitoring directory
DB_PATH = Path(__file__).parent / "predictions.db"


def create_database():
    """Creates the SQLite database and the predictions table if they don't exist."""
    try:
        # connect() will create the file if it doesn't exist
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # SQL statement to create the table
        # We use "IF NOT EXISTS" to prevent errors if the script is run multiple times.
        create_table_query = """
                             CREATE TABLE IF NOT EXISTS predictions (
                                                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                        timestamp DATETIME NOT NULL,
                                                                        MedInc REAL NOT NULL,
                                                                        HouseAge REAL NOT NULL,
                                                                        AveRooms REAL NOT NULL,
                                                                        AveBedrms REAL NOT NULL,
                                                                        Population REAL NOT NULL,
                                                                        AveOccup REAL NOT NULL,
                                                                        Latitude REAL NOT NULL,
                                                                        Longitude REAL NOT NULL,
                                                                        predicted_value REAL NOT NULL
                             ); \
                             """

        cursor.execute(create_table_query)
        conn.commit()
        print(f"Database created successfully at '{DB_PATH}'")
        print("`predictions` table is ready.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    create_database()
