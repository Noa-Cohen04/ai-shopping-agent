"""Load customers.json into an in-memory mongomock 'cust_data' collection.
No MongoDB server or Docker required: `pip install mongomock`.
The returned collection supports the standard pymongo API
(find / find_one / count_documents), so agent code is identical to real Mongo."""
import json, os
import mongomock

_DEF_PATH = os.path.join(os.path.dirname(__file__), "customers.json")

def get_cust_data_collection(json_path: str = _DEF_PATH):
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)
    client = mongomock.MongoClient()
    coll = client["company"]["cust_data"]
    if records:
        coll.insert_many(records)
    return coll

if __name__ == "__main__":
    print(f"Loaded {get_cust_data_collection().count_documents({})} demo customers.")
