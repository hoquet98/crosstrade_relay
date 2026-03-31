"""
Condition Engine — evaluates JSON rule trees against indicator payloads.

Used by AI Gate to provide programmable entry/exit gating without LLM calls.
Rules are stored as JSON in the ai_bots table (entry_conditions, exit_conditions).

Rule format:
    {
        "op": "AND",
        "conditions": [
            {"field": "rsi14", "op": "<", "value": 70},
            {"field": "htf_bias", "op": "==", "value": "bullish"},
            {"field": "session_bucket", "op": "!=", "value": "midday_chop"}
        ]
    }

Supported operators:
    Numeric: <, <=, >, >=, ==, !=, between (value=[low, high])
    String:  ==, !=, in (value=[list])
    Boolean: ==, !=
    Logical: AND, OR, NOT (grouping)

Safety: No eval(), no code injection. Pure dict/value comparison.
Unknown fields evaluate to False (fail-safe: condition not met).
"""

import logging

logger = logging.getLogger("trade_relay")


def evaluate_conditions(rules: dict, context: dict) -> bool:
    """Evaluate a JSON rule tree against an indicator context dict.

    Args:
        rules: JSON rule tree (dict with 'op' and 'conditions' or 'field')
        context: Flat dict of indicator values (enriched payload)

    Returns:
        True if all conditions are met, False otherwise.
        Returns True if rules is None or empty (no conditions = pass).
    """
    if not rules:
        return True

    if isinstance(rules, str):
        # Parse if string was passed
        import json
        try:
            rules = json.loads(rules)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Invalid condition rules string: {rules[:100]}")
            return False

    if not isinstance(rules, dict):
        return False

    return _evaluate_node(rules, context)


def _evaluate_node(node: dict, context: dict) -> bool:
    """Recursively evaluate a single node in the rule tree."""
    op = node.get("op", "").upper()

    # Logical operators (grouping)
    if op == "AND":
        conditions = node.get("conditions", [])
        if not conditions:
            return True
        return all(_evaluate_node(c, context) for c in conditions)

    elif op == "OR":
        conditions = node.get("conditions", [])
        if not conditions:
            return True
        return any(_evaluate_node(c, context) for c in conditions)

    elif op == "NOT":
        conditions = node.get("conditions", [])
        if not conditions:
            return True
        # NOT applies to the first condition
        return not _evaluate_node(conditions[0], context)

    # Leaf node (comparison)
    elif "field" in node:
        return _evaluate_comparison(node, context)

    else:
        logger.warning(f"Unknown condition node: {node}")
        return False


def _evaluate_comparison(node: dict, context: dict) -> bool:
    """Evaluate a single field comparison.

    Returns False if the field doesn't exist in context (fail-safe).
    """
    field = node.get("field", "")
    op = node.get("op", "==")
    expected = node.get("value")

    # Get actual value from context
    actual = context.get(field)

    # Unknown field = condition not met (fail-safe)
    if actual is None:
        return False

    try:
        # Numeric comparisons
        if op == "<":
            return float(actual) < float(expected)
        elif op == "<=":
            return float(actual) <= float(expected)
        elif op == ">":
            return float(actual) > float(expected)
        elif op == ">=":
            return float(actual) >= float(expected)
        elif op == "==":
            # Handle bool, string, and numeric
            if isinstance(actual, bool) or isinstance(expected, bool):
                return bool(actual) == bool(expected)
            if isinstance(actual, str) or isinstance(expected, str):
                return str(actual).lower() == str(expected).lower()
            return float(actual) == float(expected)
        elif op == "!=":
            if isinstance(actual, bool) or isinstance(expected, bool):
                return bool(actual) != bool(expected)
            if isinstance(actual, str) or isinstance(expected, str):
                return str(actual).lower() != str(expected).lower()
            return float(actual) != float(expected)
        elif op == "between":
            # expected = [low, high]
            if isinstance(expected, (list, tuple)) and len(expected) == 2:
                return float(expected[0]) <= float(actual) <= float(expected[1])
            return False
        elif op == "in":
            # expected = [list of values]
            if isinstance(expected, (list, tuple)):
                actual_str = str(actual).lower()
                return actual_str in [str(v).lower() for v in expected]
            return False
        else:
            logger.warning(f"Unknown operator '{op}' in condition: {node}")
            return False

    except (ValueError, TypeError) as e:
        logger.debug(f"Condition eval error: {field} {op} {expected} (actual={actual}): {e}")
        return False


def parse_text_conditions(text: str) -> dict:
    """Parse a text-based condition string into a JSON rule tree.

    Supports: field op value AND/OR field op value
    Examples:
        "rsi14 < 70 AND htf_bias == bullish"
        "exit_score >= 3 OR ticks_pnl > 50"
        "rsi14 < 70 AND (htf_bias == bullish OR session_bucket == morning)"

    Returns a JSON rule tree dict.
    """
    text = text.strip()
    if not text:
        return {}

    # Split by top-level AND/OR (not inside parentheses)
    # Simple tokenizer — handles most cases
    tokens = _tokenize(text)
    if not tokens:
        return {}

    return _parse_tokens(tokens)


def _tokenize(text: str) -> list:
    """Tokenize a condition string into parts."""
    import re
    # Split on whitespace but preserve quoted strings
    parts = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', text)

    tokens = []
    i = 0
    while i < len(parts):
        part = parts[i]

        if part.upper() in ("AND", "OR"):
            tokens.append({"type": "logic", "value": part.upper()})
            i += 1
        elif part == "(":
            tokens.append({"type": "paren_open"})
            i += 1
        elif part == ")":
            tokens.append({"type": "paren_close"})
            i += 1
        elif i + 2 < len(parts) and parts[i + 1] in ("<", "<=", ">", ">=", "==", "!=", "in", "between"):
            # field op value
            field = part
            op = parts[i + 1]
            value_str = parts[i + 2].strip('"').strip("'")

            # Try to parse value as number or bool
            value = _parse_value(value_str)

            tokens.append({"type": "condition", "field": field, "op": op, "value": value})
            i += 3
        else:
            i += 1  # skip unknown tokens

    return tokens


def _parse_value(s: str):
    """Parse a value string into the appropriate type."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _parse_tokens(tokens: list) -> dict:
    """Parse tokenized conditions into a rule tree."""
    if not tokens:
        return {}

    # Single condition
    conditions = []
    logic_op = "AND"  # default

    for token in tokens:
        if token["type"] == "condition":
            conditions.append({
                "field": token["field"],
                "op": token["op"],
                "value": token["value"],
            })
        elif token["type"] == "logic":
            logic_op = token["value"]

    if len(conditions) == 1:
        return conditions[0]

    return {
        "op": logic_op,
        "conditions": conditions,
    }


def conditions_to_text(rules: dict) -> str:
    """Convert a JSON rule tree back to human-readable text."""
    if not rules:
        return ""

    return _node_to_text(rules)


def _node_to_text(node: dict) -> str:
    """Convert a single node to text."""
    op = node.get("op", "").upper()

    if op in ("AND", "OR"):
        conditions = node.get("conditions", [])
        parts = [_node_to_text(c) for c in conditions]
        joiner = f" {op} "
        return joiner.join(parts)

    elif op == "NOT":
        conditions = node.get("conditions", [])
        if conditions:
            return f"NOT ({_node_to_text(conditions[0])})"
        return "NOT ()"

    elif "field" in node:
        field = node.get("field", "")
        cmp_op = node.get("op", "==")
        value = node.get("value")
        if isinstance(value, str):
            return f'{field} {cmp_op} "{value}"'
        elif isinstance(value, bool):
            return f'{field} {cmp_op} {str(value).lower()}'
        else:
            return f'{field} {cmp_op} {value}'

    return str(node)
