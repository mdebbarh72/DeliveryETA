import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
import sys

logger = get_logger("etracting_data")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "bronze"
FILE_NAME = "FoodDelivery.csv"


def readData() -> pd.DataFrame: 

    filePath = DATA_PATH / FILE_NAME

    if not filePath.exists(): 
        logger.critical(f"source file is missing")
        sys.exit(1)

    try: 
        deliveryData = pd.read_csv(filePath, na_values=['nan', 'NaN', 'NAN', 'NA', 'na'], keep_default_na=True)
        return deliveryData
    except Exception as e:
        logger.error(f"error while trying to read data: {e}")
        return





