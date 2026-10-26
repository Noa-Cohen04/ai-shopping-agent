# Validation results

## Update: 2026-10-26

All 18 offline checks also passed in a fresh environment using the pinned
requirements (with CPU PyTorch installed first). The embedding model downloaded
and returned real semantic-search matches. Live model conversations completed
for q1 (loyalty threshold), q2 (thermometer comparison), and q3 (scales within a
budget). The q1 total was $1877.64, correctly below the $2000 threshold. The q3
answer listed five products despite the three-product instruction. These observed
runs are a limited functional check, not a complete quality evaluation. The
planned 13-case run was interrupted; no overall pass rate is claimed.

The earlier preparation record below describes the checks at that earlier stage.


Portfolio preparation date: 2026-10-25.

## Executed locally

- Parsed every Python source file for valid syntax.
- Ran 18 offline regression checks successfully with `python -m unittest discover -s tests -v`.
- Ran `python demo_offline.py`; its actual stdout is in `DEMO_OUTPUT.txt`.
- Verified the packaged data has 787 unique product IDs and 50 demo customers.
- Checked that the known original key, course endpoint, author IDs, and original
  customer credentials are absent from the final archive contents.

The checks cover catalog loading, budget/sale/availability filters, verified
premier discounts, rejection of unauthenticated/regular-customer discounts,
authentication failure, restricted lookup fields, literal-name handling,
calculator behavior, missing-key handling, invalid tool arguments, no-match
formatting, tool-result forwarding through a mocked model loop, and redacted
provider-error responses.

The mocked loop uses programmed responses. It tests orchestration, not model
reasoning or answer quality. No network calls are made by these tests.

## Environment used

Python 3.12.14, NumPy 2.3.5, pandas 2.2.3,
mongomock 4.1.2, and setuptools 80.9.0. The full pinned live dependency set was
not installed and verified in a new virtual environment during preparation.

## Not verified in this run

- Live provider access, billing, endpoint compatibility, or model availability.
- The actual embedding-model download and semantic-search results.
- End-to-end answers to the sample questions using a real model.
- Production security, concurrent sessions, latency, throughput, or deployment.

A fresh full installation and a small live evaluation with authorized credentials
are the next validation steps. No live performance or answer-accuracy score is
claimed.
