"""Offline regression checks. No API key or downloaded embedding model needed."""
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hw4


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hw4._ensure_initialized()
        cls.premier = hw4._load_customers().find_one({"status": "premier"})
        cls.regular = hw4._load_customers().find_one({"status": "regular"})

    def authenticate(self, customer, session):
        return json.loads(hw4._run_tool_call("authenticate", {
            "email": customer["email"], "pw": customer["pw"]
        }, session))

    def test_data_is_complete(self):
        self.assertEqual(len(hw4._load_products()), 787)
        self.assertEqual(hw4._load_customers().count_documents({}), 50)

    def test_budget_without_discount(self):
        self.assertEqual(json.loads(hw4.filter_products(
            name_contains="Wrist Weight", max_price=11.50)), [])

    def test_verified_premier_discount(self):
        session = {}
        self.assertTrue(self.authenticate(self.premier, session)["authenticated"])
        products = json.loads(hw4._run_tool_call("filter_products", {
            "name_contains": "Wrist Weight", "max_price": 11.50,
            "apply_premier_discount": True
        }, session))
        self.assertTrue(products)
        for product in products:
            self.assertLessEqual(product["effective_price_after_premier_discount"], 11.50)
            self.assertAlmostEqual(product["effective_price_after_premier_discount"],
                                   product["sale_price"] * 0.95)

    def test_discount_requires_authentication(self):
        result = json.loads(hw4._run_tool_call("filter_products", {
            "apply_premier_discount": True
        }, {}))
        self.assertIn("error", result)

    def test_regular_account_cannot_get_premier_discount(self):
        session = {}
        self.authenticate(self.regular, session)
        self.assertIn("error", json.loads(hw4._run_tool_call("filter_products", {
            "apply_premier_discount": True
        }, session)))

    def test_bad_password_clears_authorization(self):
        session = {}
        self.authenticate(self.premier, session)
        result = json.loads(hw4._run_tool_call("authenticate", {
            "email": self.premier["email"], "pw": "wrong"
        }, session))
        self.assertFalse(result["authenticated"])
        self.assertNotIn("status", result)
        self.assertEqual(session, {})

    def test_sale_filter(self):
        products = json.loads(hw4.filter_products(on_sale=True))
        self.assertTrue(products)
        self.assertTrue(all(p["sale_price"] < p["list_price"] for p in products))

    def test_availability_and_budget(self):
        products = json.loads(hw4.filter_products(available=True, max_price=100))
        self.assertTrue(products)
        self.assertTrue(all(hw4._is_available_value(p["available"])
                            and p["sale_price"] <= 100 for p in products))

    def test_gift_lookup_has_limited_fields(self):
        name = self.premier["first_name"] + " " + self.premier["last_name"]
        results = json.loads(hw4.get_customer(name=name))
        self.assertEqual(len(results), 1)
        self.assertEqual(set(results[0]), {"first_name", "last_name", "product_interests"})

    def test_name_wildcard_cannot_enumerate_customers(self):
        self.assertEqual(json.loads(hw4.get_customer(name=".*")), [])

    def test_whitespace_name_rejected(self):
        self.assertIn("error", json.loads(hw4.get_customer(name="   ")))

    def test_email_lookup_requires_authentication(self):
        self.assertIn("error", json.loads(hw4.get_customer(email=self.premier["email"])))

    def test_arithmetic_and_unsafe_expression(self):
        self.assertAlmostEqual(json.loads(hw4.calculator("11.99 * 0.95"))["result"], 11.3905)
        self.assertIn("error", json.loads(hw4.calculator("__import__('os').getcwd()")))
        self.assertIn("error", json.loads(hw4.calculator("1/0")))

    def test_no_key_has_helpful_message(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.assertIn("Set OPENAI_API_KEY", hw4.answer_question("Find a product"))

    def test_bad_tool_arguments_and_unknown_tool(self):
        self.assertIn("error", json.loads(hw4._run_tool_call("calculator", [])))
        self.assertIn("error", json.loads(hw4._run_tool_call("unknown", {})))

    def test_no_match_format(self):
        self.assertEqual(hw4._normalize_final_answer("Sorry. " + hw4.NO_MATCH_MESSAGE),
                         hw4.NO_MATCH_MESSAGE)

    def test_loop_returns_tool_result_to_model(self):
        calls = []
        arguments = '{"expression":"2+3"}'
        tool_call = SimpleNamespace(id="test-call", function=SimpleNamespace(
            name="calculator", arguments=arguments))
        tool_call.model_dump = lambda **kwargs: {
            "id": "test-call", "type": "function",
            "function": {"name": "calculator", "arguments": arguments}}

        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                message = SimpleNamespace(content=None, tool_calls=[tool_call])
            else:
                last = kwargs["messages"][-1]
                self.assertEqual(last["role"], "tool")
                self.assertEqual(json.loads(last["content"])["result"], 5)
                message = SimpleNamespace(content="The result is 5.", tool_calls=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=lambda **kwargs: client)}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-placeholder"}):
                self.assertEqual(hw4.answer_question("Calculate 2 + 3"), "The result is 5.")
        self.assertEqual(len(calls), 2)

    def test_api_errors_do_not_echo_credentials(self):
        def fail(**kwargs):
            raise RuntimeError("offline-test-secret")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=lambda **kwargs: client)}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-placeholder"}):
                result = hw4.answer_question("Find a product")
        self.assertIn("request failed", result)
        self.assertNotIn("offline-test-secret", result)


if __name__ == "__main__":
    unittest.main()
