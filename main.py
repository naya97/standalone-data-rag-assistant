import os
from src.infrastructure.file_reader import FileReader
from src.infrastructure.db_manager import SQLiteManager


def main():
    #Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path  = os.path.join(base_dir, 'data', 'raw', 'sample.csv')
    json_data_path = os.path.join(base_dir, 'data', 'raw', 'sample.json')
    json_path = os.path.join(base_dir, 'data', 'raw', 'config.json')
    db_path   = os.path.join(base_dir, 'data', 'db', 'local_storage.sqlite')

    #Config
    config = FileReader.read_json(json_path)
    print(f"{config['app_name']} v{config['version']}\n")

    #ETL
    db = SQLiteManager(db_path)
    csv_rows = FileReader.read_csv(csv_path)
    json_rows = FileReader.read_json(json_data_path)
    all_rows = csv_rows + json_rows

    inserted = 0
    query = "INSERT OR IGNORE INTO rag_topics (id, question, category) VALUES (?, ?, ?)"

    for row in all_rows:
        if db.execute_non_query(query, (row['id'], row['question'], row['category'])):
            inserted += 1

    print(f"Inserted {inserted} / {len(all_rows)} rows\n")

    #Queries
    print("Sample Data:")
    for row in db.execute_query("SELECT * FROM rag_topics LIMIT 3"):
        print(f" [{row['category']}] {row['question']}")

    print("\n Pricing Questions:")
    for row in db.execute_query("SELECT question FROM rag_topics WHERE category = ?", ("Pricing",)):
        print(f" {row['question']}")

    print("\n Topics per Category:")
    for row in db.execute_query("""
        SELECT category, COUNT(*) as count 
        FROM rag_topics 
        GROUP BY category 
        ORDER BY count DESC
    """):
        print(f" {row['category']}: {row['count']}")

if __name__ == "__main__":
    main()