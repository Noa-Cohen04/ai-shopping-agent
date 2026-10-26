# AI Shopping Agent

**Noa Cohen | Python, tool calling, semantic retrieval, and business rules**

A customer-support agent for an online store. It uses a language model to choose
Python tools, retrieve catalog evidence, and answer questions about products,
budgets, discounts, and customer interests.

Developed as a university project and prepared here as a portfolio demonstration.
The agent implementation is by Noa Cohen. Course-provided data and helper code
are identified in [SOURCE_NOTES.md](SOURCE_NOTES.md).

## What this demonstrates

- A custom model/tool loop with six tools and a limit of 15 model turns.
- Semantic product retrieval using `all-MiniLM-L6-v2`, normalized embeddings,
  and cosine similarity over an in-memory NumPy index.
- Exact filters for price, availability, brand, and sale status.
- Customer authentication and an application-level gate for premier-discount
  tool calls within each question.
- Arithmetic using a restricted Python AST parser instead of `eval`.
- Instructions for catalog-grounded answers and business-policy compliance.

This is an educational prototype, not a production authentication or commerce system.

## Applied AI and solution delivery skills

The project translates an online store's support requirements into a working
Python agent with explicit tools, data sources, and business rules.

| Capability | Implementation in this project |
| --- | --- |
| Business requirements into technical workflows | Budget limits, availability, sale status, and loyalty discounts are mapped to tool parameters and Python checks. |
| AI agent configuration | A system prompt defines the role and policies; six JSON tool schemas describe when and how the model should call each function. |
| API integration and structured data | The OpenAI Python SDK connects to a configurable model endpoint. Tool arguments are decoded from JSON, executed in Python, and returned as structured results. |
| Connecting multiple data sources | A 787-product CSV catalog, 50 synthetic customer records loaded into an in-memory MongoDB-style collection, and a text policy file supply the agent's context. |
| Semantic retrieval and exact constraints | Sentence-transformer embeddings support natural-language needs; separate filters enforce explicit price, brand, sale, and availability constraints. |
| Python data processing | pandas handles catalog loading, numeric price conversion, missing text values, filtering, and sorting. Results are converted into JSON-compatible values. |
| Testing and troubleshooting | 18 offline regression checks cover tool behavior, authentication failures, discount eligibility, malformed tool arguments, and a mocked tool-calling loop. A separate offline demo provides reproducible output. |
| Reviewable delivery | Setup instructions, configurable credentials, sample questions, prompt documentation, and an offline demonstration make the solution easier for another developer to inspect and run. |

### Example: a budget-aware recommendation

A customer asks for wrist weights costing at most $11.50. The catalog contains
a matching product at $11.99. For an authenticated premier customer, the agent's
tool dispatcher permits the 5% discount, and the filter checks the discounted
price against the budget. The resulting price is approximately $11.39.
Without verified premier status, the agent cannot request the discounted filter.
This illustrates how a business rule changes both the workflow and the result.

### Design choices

- **Use retrieval for product evidence.** Product documents combine name,
  description, brand, and category so the search can match meaning beyond exact
  keywords. Prices and factual claims should come from retrieved catalog data.
- **Separate model decisions from executable operations.** The model selects
  tools; Python performs filtering, authentication, and arithmetic. Discount
  authorization is checked in the dispatcher rather than relying only on a prompt.
- **Keep calculations explicit.** A calculator parses a restricted arithmetic
  syntax tree, giving the model a concrete result to use in its answer.
- **Make setup and failure cases visible.** Missing credentials produce a setup
  message, malformed tool arguments produce an error result, and model failures
  return a message without exposing the raw provider error.

These are prototype capabilities demonstrated in code. The project does not
claim client deployment experience, SQL integration, a custom REST server,
persistent operational logging, or use of the Gentrix platform.

## Review without an API key

The included offline demo calls the real local tools directly. It does **not**
call a language model or perform semantic retrieval. It demonstrates how a
$11.99 wrist-weight product falls within an $11.50 budget after a verified
premier customer's 5% discount, at approximately $11.39.

See [DEMO_OUTPUT.txt](DEMO_OUTPUT.txt) for output captured from an actual local run,
and [TESTING.md](TESTING.md) for the checks performed and their limits.

From the extracted project directory, using Python 3.10 or newer:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the smaller offline dependency set and run:

```sh
python -m pip install -r requirements-offline.txt
python demo_offline.py
python -m unittest discover -s tests -v
```

## Run the live AI agent

Install the complete dependency set:

```sh
python -m pip install -r requirements.txt
```

Use your own authorized API access. For the standard OpenAI API, set the key
and a tool-calling model available to your account. The default model name is
`gpt-4.1-mini`; access and service charges depend on your provider/account.

Windows PowerShell (current terminal only):

```powershell
$key = Read-Host "Enter your API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $key).Password
$env:OPENAI_MODEL = "gpt-4.1-mini"
python hw4.py "What scales cost less than 100 dollars and read in pounds and kilograms?"
```

Never paste a real key into project files or commits.

macOS/Linux (bash):

```bash
read -rs -p "API key: " OPENAI_API_KEY
export OPENAI_API_KEY
echo
export OPENAI_MODEL="gpt-4.1-mini"
python hw4.py "What scales cost less than 100 dollars and read in pounds and kilograms?"
```

For an Azure OpenAI v1 or another compatible endpoint, also set
`OPENAI_BASE_URL` to **your own** API base URL and `OPENAI_MODEL` to the
deployment/model name accepted by that endpoint. The endpoint must support
the Chat Completions interface and tool calling used by the supplied SDK.
An OpenAI key and an Azure key are not interchangeable. This package contains
no course endpoint or course API key.

`.env.example` documents the variable names. The program reads operating-system
environment variables; it does **not** automatically load `.env` files.
An unset key produces a setup message. The first semantic search downloads the
embedding model if it is not already cached; that step needs internet access.

## Example questions

- What scales cost less than $100 and support pounds and kilograms?
- Recommend two available products for snowboarding safety.
- Compare specific thermometers from the catalog.
- Find wrist weights within a budget, applying an authenticated premier discount.

`example_queries.txt` includes additional examples with public, synthetic demo
accounts. All emails in `customers.json` use `example.com`; passwords are demo
fixtures. Never replace these with real customer passwords or data.

## How it works

1. `answer_question()` sends the question, system instructions, and tool schemas
   to the configured model.
2. The model requests one or more tools. Python executes those requests and
   sends the results back to the model.
3. `search_products` ranks products semantically. `filter_products` applies
   exact constraints. Authentication, policy lookup, and the calculator provide
   additional evidence.
4. The loop returns the model's answer, or stops after 15 model turns.

Product embeddings are created lazily and reused within the process. The index
is held in memory, not in a separate vector-database service. Customer data is
loaded into `mongomock`; no MongoDB server is required.

## Files

| File | Purpose |
| --- | --- |
| `hw4.py` | Agent, tool implementations, schemas, and command-line entry point |
| `prompts.md` | Actual system prompt, tool descriptions, and design notes |
| `customer_store.py` | Course-derived in-memory customer loader |
| `customers.json` | 50 anonymized demonstration accounts |
| `product_data.csv` | 787-product catalog included for reproducibility |
| `policies.txt` | Store policies used by the example |
| `example_queries.txt` | Course-derived sample questions with demo credentials |
| `demo_offline.py` | Real local tool demonstration without model calls |
| `tests/test_agent.py` | Offline regression checks and a mocked tool-loop check |
| `TESTING.md` | Validation scope and results |

## Limitations and next steps

- Live model runs were observed for the loyalty-threshold calculation, product
  comparison, and budget-based scale search. The embedding model also downloaded
  and returned semantic search results. The planned full evaluation was not
  completed. In the observed scale-search response, the model exceeded the
  three-product recommendation limit; this remains a known prompt-compliance issue.
- The three-product limit, strict product relevance, and several response policies
  rely on prompt instructions. There is no complete output-policy validator.
- Retrieval returns top-k candidates without a calibrated relevance cutoff.
- The demo authentication passes credentials through the model's conversation
  and uses plaintext fixture passwords. Production authentication must happen
  outside the model, with proper identity, authorization, and session controls.
- Discount gating applies to agent-dispatched tool calls. The underlying Python
  filtering function is a trusted internal helper, not a secured public API.
- Prices use floating-point arithmetic. A production checkout should use a
  defined currency/rounding policy and decimal arithmetic.
- Each question starts a new session. The prototype does not place orders,
  persist purchases, update loyalty status, or implement multi-user chat sessions.
- Provider errors return a safe message; retry/backoff and operational monitoring
  are future work. A model turn can request multiple tools, so the turn cap is
  not a separate total tool-call cap.

## Portfolio preparation changes

The original submission is preserved separately. This version removes embedded
credentials and personal identifiers, configures provider access via environment
variables, resolves data paths relative to the code, anonymizes demo accounts,
escapes customer-name regex input, gates premier-discount tool calls on verified
status, avoids echoing raw provider errors, and adds runnable demonstrations and
regression checks. The original semantic-retrieval approach and agent structure
are retained.
