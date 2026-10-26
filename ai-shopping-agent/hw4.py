# AI Shopping Agent: tool calling and semantic product retrieval

# Name: noa cohen

import os
import json
import ast
import operator
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# -----------------------------
# Constants
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent
PRODUCT_DATA_PATH = DATA_DIR / "product_data.csv"
POLICIES_PATH = DATA_DIR / "policies.txt"

NO_MATCH_MESSAGE = "We have no product available today that fits your request"


def _model_config():
    """Read private credentials at runtime; never embed them in source code."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY before running the live agent. See README.md.")
    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENAI_BASE_URL", "").strip() or None,
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
    }


# -----------------------------
# Global lazy-loaded state
# -----------------------------

_products_df: Optional[pd.DataFrame] = None
_policies_text: Optional[str] = None
_cust_data = None

_embedding_model: Optional[Any] = None
_product_embeddings: Optional[np.ndarray] = None
_product_texts: Optional[List[str]] = None

# -----------------------------
# Initialization
# -----------------------------

def _load_products() -> pd.DataFrame:
    global _products_df

    if _products_df is not None:
        return _products_df

    df = pd.read_csv(PRODUCT_DATA_PATH)

    # Convert prices safely
    for col in ["List Price", "Sale Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize text columns
    text_cols = ["Product Name", "Description", "Brand", "Category", "Available"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    _products_df = df
    return _products_df


def _load_policies() -> str:
    global _policies_text

    if _policies_text is not None:
        return _policies_text

    with open(POLICIES_PATH, "r", encoding="utf-8") as f:
        _policies_text = f.read()

    return _policies_text


def _load_customers():
    global _cust_data

    if _cust_data is not None:
        return _cust_data

    from customer_store import get_cust_data_collection

    _cust_data = get_cust_data_collection()
    return _cust_data


def _build_product_text(row: pd.Series) -> str:
    """Standard document representation for RAG."""
    return (
        f"Product Name: {row.get('Product Name', '')}. "
        f"Description: {row.get('Description', '')}. "
        f"Brand: {row.get('Brand', '')}. "
        f"Category: {row.get('Category', '')}."
    )


def _ensure_rag_index():
    """
    Builds an in-memory semantic vector index over product_data.csv.
    Uses SentenceTransformers with normalized vectors for pure Cosine Similarity.
    """
    global _embedding_model, _product_embeddings, _product_texts

    if _embedding_model is not None and _product_embeddings is not None:
        return

    from sentence_transformers import SentenceTransformer

    df = _load_products()
    _product_texts = [_build_product_text(row) for _, row in df.iterrows()]

    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    _product_embeddings = _embedding_model.encode(
        _product_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalizing allows np.dot to equal Cosine Similarity
        show_progress_bar=False,
    )


def _ensure_initialized(include_rag: bool = False):
    _load_products()
    _load_policies()
    _load_customers()

    if include_rag:
        _ensure_rag_index()


# -----------------------------
# Formatting helpers
# -----------------------------

def _product_row_to_dict(row: pd.Series) -> Dict[str, Any]:
    return {
        "uniq_id": row.get("Uniq Id", ""),
        "product_name": row.get("Product Name", ""),
        "description": row.get("Description", ""),
        "list_price": None if pd.isna(row.get("List Price")) else float(row.get("List Price")),
        "sale_price": None if pd.isna(row.get("Sale Price")) else float(row.get("Sale Price")),
        "brand": row.get("Brand", ""),
        "gtin": row.get("Gtin", ""),
        "category": row.get("Category", ""),
        "available": row.get("Available", ""),
    }


def _json_result(data: Any) -> str:
    def make_json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): make_json_safe(v) for k, v in value.items()}
        if isinstance(value, list) or isinstance(value, tuple):
            return [make_json_safe(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return value

    return json.dumps(make_json_safe(data), ensure_ascii=False, indent=2)


def _is_available_value(value: str) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "yes", "y", "1", "available", "in stock"}


# -----------------------------
# Tool implementations
# -----------------------------

def search_products(query: str, k: int = 5) -> str:
    """
    Semantic RAG search over product_data.csv using pure Cosine Similarity.
    """
    _ensure_initialized(include_rag=True)

    if not query or not query.strip():
        return _json_result({"error": "query must not be empty"})

    k = max(1, min(int(k), 10))

    query_embedding = _embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = np.dot(_product_embeddings, query_embedding)

    df = _load_products()
    combined = [(float(scores[idx]), idx) for idx in range(len(df))]
    combined.sort(reverse=True, key=lambda x: x[0])

    results = []
    for score, idx in combined[:k]:
        row = df.iloc[int(idx)]
        item = _product_row_to_dict(row)
        item["similarity_score"] = float(score)
        results.append(item)

    return _json_result(results)


def filter_products(
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    available: Optional[bool] = None,
    name_contains: Optional[str] = None,
    description_contains: Optional[str] = None,
    on_sale: Optional[bool] = None,
    apply_premier_discount: Optional[bool] = False,
    k: int = 10,
) -> str:
    """
    Structured filtering over product_data.csv.
    """
    _ensure_initialized(include_rag=False)

    df = _load_products().copy()
    price_col = "Sale Price"

    if apply_premier_discount:
        df["Effective Price"] = df["Sale Price"] * 0.95
        price_col = "Effective Price"

    if max_price is not None:
        df = df[df[price_col] <= float(max_price)]

    if min_price is not None:
        df = df[df[price_col] >= float(min_price)]

    if brand:
        df = df[df["Brand"].str.contains(str(brand), case=False, na=False, regex=False)]

    if category:
        df = df[df["Category"].str.contains(str(category), case=False, na=False, regex=False)]

    if name_contains:
        df = df[df["Product Name"].str.contains(str(name_contains), case=False, na=False, regex=False)]

    if description_contains:
        df = df[df["Description"].str.contains(str(description_contains), case=False, na=False, regex=False)]

    if available is not None:
        df = df[df["Available"].apply(_is_available_value) == bool(available)]

    if on_sale is not None:
        sale_mask = (
            df["List Price"].notna()
            & df["Sale Price"].notna()
            & (df["Sale Price"] < df["List Price"])
        )
        df = df[sale_mask == bool(on_sale)]

    df = df.sort_values(by=price_col, ascending=True, na_position="last")
    k = max(1, min(int(k), 20))

    results = []
    for _, row in df.head(k).iterrows():
        item = _product_row_to_dict(row)
        if apply_premier_discount:
            effective_price = row.get("Effective Price")
            item["effective_price_after_premier_discount"] = (
                None if pd.isna(effective_price) else float(effective_price)
            )
        results.append(item)

    return _json_result(results)


def authenticate(email: str, pw: str) -> str:
    """
    Authenticate a customer by email and password.
    Returns status only if authentication succeeds.
    """
    _ensure_initialized(include_rag=False)

    coll = _load_customers()
    customer = coll.find_one({"email": email, "pw": pw}, {"_id": 0, "pw": 0})

    if not customer:
        return _json_result({
            "authenticated": False,
            "message": "Authentication failed. Do not reveal customer data."
        })

    return _json_result({
        "authenticated": True,
        "first_name": customer.get("first_name"),
        "last_name": customer.get("last_name"),
        "email": customer.get("email"),
        "status": customer.get("status"),
        "product_interests": customer.get("product-interests", []),
    })


def get_customer(email: Optional[str] = None, name: Optional[str] = None) -> str:
    """
    Look up another customer by full name for gift recommendations.
    Never returns password, email, status, or other private account data.
    """
    _ensure_initialized(include_rag=False)

    if email:
        return _json_result({
            "error": "Customer lookup by email requires authentication. Use authenticate(email, pw) for the customer's own data."
        })

    if not name or not name.strip():
        return _json_result({"error": "Provide a customer full name."})

    coll = _load_customers()
    parts = str(name).strip().split()

    if len(parts) >= 2:
        first = parts[0]
        last = " ".join(parts[1:])
        query = {
            "first_name": {"$regex": f"^{re.escape(first)}$", "$options": "i"},
            "last_name": {"$regex": f"^{re.escape(last)}$", "$options": "i"},
        }
    else:
        query = {
            "$or": [
                {"first_name": {"$regex": re.escape(str(name).strip()), "$options": "i"}},
                {"last_name": {"$regex": re.escape(str(name).strip()), "$options": "i"}},
            ]
        }

    customers = list(coll.find(query, {"_id": 0, "pw": 0}))

    safe_results = []
    for customer in customers:
        safe_results.append({
            "first_name": customer.get("first_name"),
            "last_name": customer.get("last_name"),
            "product_interests": customer.get("product-interests", []),
        })

    return _json_result(safe_results)


def read_policies() -> str:
    _ensure_initialized(include_rag=False)
    return _load_policies()


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node):
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed.")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unary operator not allowed.")
        operand = _eval_ast_node(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)

    raise ValueError("Expression contains unsupported syntax.")


def calculator(expression: str) -> str:
    """
    Safe calculator for arithmetic.
    Does not use eval.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_ast_node(tree)
        return _json_result({"expression": expression, "result": result})
    except Exception as e:
        return _json_result({"expression": expression, "error": str(e)})

# -----------------------------
# Agent prompt and tool schemas
# -----------------------------

SYSTEM_PROMPT = """
You are a customer-support AI agent for an online store.

Your job is to answer customer questions using only the provided tools and the local data files.
The product catalog in product_data.csv is the authoritative source for all product facts.
Never use the Internet, outside knowledge, or invented product information.

Company policies you must strictly follow:
1. A customer seeking a product should always be given the best match to their request.
2. All customers with premier status get a 5% discount on items.
3. A customer gets premier status after $2000 of purchases in a year. The status lasts one year.
4. If a customer asks for information regarding their own customer data, they must provide their email and pw. Customer information is not given to anyone else.
5. Product information in product_data.csv is the authoritative information on the product (e.g., not Internet data).
6. If a customer asks you to suggest a product, provide a list of NO MORE THAN 3 PRODUCTS TOTAL across the entire response. Never output 4 or more under any circumstances.
7. If no product satisfies the user's constraints (e.g., they ask for items on sale and none match), you must output EXACTLY AND ONLY this sentence: "We have no product available today that fits your request". Do not apologize, and do NOT offer alternatives.

Tool Selection & Reasoning Guidance:
- Semantic Needs vs. Exact Claims: If the user asks for products matching an abstract need, lifestyle, or natural language description (e.g., "gear for winter sports", "gifts for my sister", "items to help me relax"), use `search_products` (Semantic RAG). 
- If the user asks for items with specific medical, regulatory, or manufacturing claims, treat those as exact string constraints and use `filter_products` with `description_contains` or `name_contains`. Do not rely solely on semantic similarity for strict certifications.
- Compound Searches & "On Sale": If a user wants items matching interests AND that are "on sale", first authenticate to get their interests. A product is ONLY "on sale" if its `sale_price` is strictly less than its `list_price`. You can either use `filter_products` with `on_sale=true` and search by category, OR use `search_products` and manually verify the prices.
- Respecting User Constraints: If a user explicitly applies a hard constraint to their request (e.g., they only want items "on sale", or they have a strict maximum budget), you must respect that boundary. Do not drop their constraints just to provide non-matching alternatives. If zero products pass their exact filters, conclude that no items match and immediately apply Policy 7.
- Budget constraints & Discounts: If the customer has a budget, verify their status with `authenticate`. If they are a premier customer, use `filter_products` with `apply_premier_discount=true` before comparing product prices to their budget. Do not guess math; always use the `calculator`.
- Usage / Duration Math: If a customer asks how long a product will last, read its detailed description to find explicit usage rates (e.g., exact uses per container). Use the calculator to determine the days. You MUST NOT invent, guess, or assume usage amounts (e.g., do not assume "1 ml per wash"). If the exact usage rate per wash is not explicitly written in the product data, you must state clearly that the exact duration cannot be determined.
- Premier Threshold: To check if a purchase makes someone premier, use the `calculator` to add their stated year-to-date spend to the price of exactly one unit, then compare to $2000.
- Quantity Handling: Assume the customer is buying exactly 1 unit of a product unless they explicitly specify a different quantity. Never extract quantities from numbers that are part of a product's name or brand (e.g., "3M").
- Absolute Limit: Before returning your final answer, count the products you are suggesting. If the count exceeds 3, select the best 3 and silently discard the rest to comply with Policy 6.
- Anti-Hallucination & Exact Quotes: Never paraphrase or invent product features, medical claims, or certifications. If the catalog says "Medically Proven", do not rephrase it as "clinically tested". Rely strictly on the exact wording provided by the tool results.
- Product Details: Always include the Sale Price when describing, recommending, or comparing products, even if the user does not explicitly ask for the price.
- Strict Tone & Formatting: Never end your final response with conversational filler, follow-up questions, or offers for further assistance (e.g., do NOT say "Let me know if you need help"). Just provide the factual answer and stop.
- Always verify that the product's 'intended use' (as described in the catalog) matches the user's explicit purpose. If a user asks for medical products, exclude non-medical or industrial alternatives even if they share keywords.
- Prioritize product relevance and quality over reaching the maximum count of 3
- Best-Match Prioritization: When selecting suggestions, prioritize products that align most closely with the user's intended use-case. If a retrieved product is only tangentially relevant, exclude it in favor of more precise matches, even if you end up with fewer than 3 suggestions.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Semantic RAG search over product_data.csv. "
                "Use this tool for broad, natural-language product needs, abstract ideas, gifts, "
                "comparable products, or lifestyle items. Do not use this tool alone if the user asks "
                "for strict manufacturing/medical certifications; use filter_products for exact text matches instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language product need or semantic search query."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of candidates to retrieve. Usually 5, max 10."
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_products",
            "description": (
                "Structured filter over product_data.csv. Use this for hard constraints: exact names, "
                "specific certifications in descriptions, max/min prices, specific brands, availability, "
                "and whether products are on sale. "
                "If the customer is authenticated as premier and you need to filter by a budget limit, "
                "set apply_premier_discount=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {"type": "number", "description": "Maximum Sale Price."},
                    "min_price": {"type": "number", "description": "Minimum Sale Price."},
                    "brand": {"type": "string", "description": "Brand name to match."},
                    "category": {"type": "string", "description": "Category text to match."},
                    "available": {"type": "boolean", "description": "Filter by availability."},
                    "name_contains": {"type": "string", "description": "Exact substring that must appear in Product Name."},
                    "description_contains": {"type": "string", "description": "Exact substring that must appear in Description. Great for exact claims."},
                    "on_sale": {"type": "boolean", "description": "True returns items where Sale Price is strictly less than List Price."},
                    "k": {"type": "integer", "description": "Maximum results to return."},
                    "apply_premier_discount": {"type": "boolean", "description": "If true, max_price and min_price filters apply to (Sale Price * 0.95)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "authenticate",
            "description": "Authenticate a customer using email and password. Must be used before revealing their status, interests, or applying their personal discounts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "pw": {"type": "string"},
                },
                "required": ["email", "pw"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Look up a customer by name for gift recommendations; returns only names and product interests. Email lookup is blocked. Use authenticate for the customer's own account data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "name": {"type": "string", "description": "Customer full name"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_policies",
            "description": "Read the company policies.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safe arithmetic calculator. Must be used for all mathematical operations: pricing discounts (5%), threshold checks against $2000, quantity math, and days of duration math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression, e.g., '1700 + 349.99' or '11.99 * 0.95'."},
                },
                "required": ["expression"],
            },
        },
    },
]

_TOOL_FUNCTIONS = {
    "search_products": search_products,
    "filter_products": filter_products,
    "authenticate": authenticate,
    "get_customer": get_customer,
    "read_policies": read_policies,
    "calculator": calculator,
}


def _run_tool_call(tool_name: str, arguments: Dict[str, Any], session=None) -> str:
    if tool_name not in _TOOL_FUNCTIONS:
        return _json_result({"error": f"Unknown tool: {tool_name}"})
    if not isinstance(arguments, dict):
        return _json_result({"error": "Tool arguments must be a JSON object."})
    # Authorization belongs in application code, not only in the system prompt.
    if tool_name == "filter_products" and arguments.get("apply_premier_discount"):
        if not session or session.get("status") != "premier":
            return _json_result({"error": "Authenticate a premier customer before applying the discount."})
    if tool_name == "authenticate" and session is not None:
        session.clear()
    try:
        result = _TOOL_FUNCTIONS[tool_name](**arguments)
        if tool_name == "authenticate" and session is not None:
            customer = json.loads(result)
            if customer.get("authenticated"):
                session["status"] = customer.get("status")
        return result
    except Exception as e:
        return _json_result({"error": type(e).__name__, "tool": tool_name})


def _normalize_final_answer(content: str) -> str:
    if not content:
        return ""

    text = content.strip()
    if NO_MATCH_MESSAGE.lower() in text.lower():
        return NO_MATCH_MESSAGE

    return text


# -----------------------------
# Main agent interface
# -----------------------------

def answer_question(question: str) -> str:
    """
    Main single-question interface.
    It runs a raw OpenAI tool-calling agent loop.
    """
    try:
        config = _model_config()
    except ValueError as error:
        return str(error)
    _ensure_initialized(include_rag=False)

    from openai import OpenAI

    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    session = {}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    max_steps = 15

    for _ in range(max_steps):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception:
            return "The model request failed. Check your credentials, endpoint, model access, and network connection."

        message = response.choices[0].message

        assistant_message = {
            "role": "assistant",
            "content": message.content,
        }

        if message.tool_calls:
            assistant_message["tool_calls"] = [
                tool_call.model_dump(exclude_none=True)
                for tool_call in message.tool_calls
            ]

        messages.append(assistant_message)
        
        if not message.tool_calls:
            return _normalize_final_answer(message.content or "")
        
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            tool_result = _run_tool_call(tool_name, arguments, session)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

    return "Sorry, I could not complete the request within the allowed number of tool calls."


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ask the AI shopping agent a question.")
    parser.add_argument("question", nargs="?", default="What scales cost less than $100 and read in pounds and kilograms?")
    print(answer_question(parser.parse_args().question))