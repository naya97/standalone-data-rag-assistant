import csv
import json
from typing import List, Dict, Any

class FileReader:
    @staticmethod
    def read_csv(file_path: str) -> List[Dict[str, Any]]:
        # Read a CSV file and return a list of dictionaries.
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return [row for row in reader]

    @staticmethod
    def read_json(file_path: str) -> List[Dict[str, Any]]:
        # Read a JSON file and return a list of dictionaries.
        with open(file_path, mode='r', encoding='utf-8') as jsonfile:
            return json.load(jsonfile)