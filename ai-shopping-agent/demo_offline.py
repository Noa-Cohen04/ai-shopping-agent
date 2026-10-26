"""Exercise real local tools without an API key, model, or network request."""
import json

import hw4


def main():
    hw4._ensure_initialized()
    print("OFFLINE TOOL DEMO (not an LLM-generated conversation)")
    print(f"Catalog: {len(hw4._load_products())} products")
    print(f"Demo customers: {hw4._load_customers().count_documents({})}")
    query = {"name_contains": "Wrist Weight", "max_price": 11.50}
    result = json.loads(hw4._run_tool_call("filter_products", query, {}))
    print(f"Wrist weights within $11.50 before discount: {len(result)}")
    customer = hw4._load_customers().find_one({"status": "premier"})
    session = {}
    auth = json.loads(hw4._run_tool_call(
        "authenticate", {"email": customer["email"], "pw": customer["pw"]}, session
    ))
    print(f"Demo premier account authenticated: {auth['authenticated']}")
    query["apply_premier_discount"] = True
    result = json.loads(hw4._run_tool_call("filter_products", query, session))
    for product in result:
        print(f"Match: {product['product_name']}")
        print(f"Catalog price: ${product['sale_price']:.2f}")
        print(f"Premier price: ${product['effective_price_after_premier_discount']:.2f}")
    print("No model calls or semantic searches were made.")


if __name__ == "__main__":
    main()
