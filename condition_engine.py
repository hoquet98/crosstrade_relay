"""
Condition Engine — evaluates conditions against indicator payloads.

Two modes:
1. JSON rule trees (simple form-based conditions)
2. Python scripts (advanced code-based conditions)

Used by AI Gate to provide programmable entry/exit gating without LLM calls.

JSON Rule format:
    {
        "op": "AND",
        "conditions": [
            {"field": "rsi14", "op": "<", "value": 70},
            {"field": "htf_bias", "op": "==", "value": "bullish"}
        ]
    }

Python Script format:
    score = 0
    if whale_buying and whale_conviction > 2: score += 3
    if vwap_long_ok: score += 2
    entry = score >= 5

Safety:
- JSON rules: No eval(), pure dict/value comparison
- Python scripts: exec() with restricted builtins (no imports, no file/network access)
  Only math, min, max, abs, round, len, range, True, False, None allowed
"""

import logging

logger = logging.getLogger("trade_relay")


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON SCRIPT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

# Safe builtins — no imports, no file access, no network
_SAFE_BUILTINS = {
    "True": True, "False": False, "None": None,
    "abs": abs, "min": min, "max": max, "round": round,
    "len": len, "range": range, "int": int, "float": float,
    "str": str, "bool": bool,
    "sum": sum, "sorted": sorted,
}


def evaluate_script(code: str, context: dict, mode: str = "entry") -> bool:
    """Evaluate a Python script against indicator context.

    The script has access to all indicator values as local variables.
    It must set `entry = True/False` or `exit = True/False` to signal its decision.

    Args:
        code: Python code string
        context: Flat dict of indicator values
        mode: "entry" or "exit" — determines which variable to read

    Returns:
        True if the script sets entry=True (or exit=True for exit mode)
        False otherwise or on error
    """
    if not code or not code.strip():
        return True  # No script = pass through

    # Check if it looks like a Python script (has if/else/score/=)
    stripped = code.strip()
    if not any(kw in stripped for kw in ['if ', 'score', 'entry', 'exit', '=']):
        # Might be a simple text condition — try the old parser
        return False

    # Build execution namespace with indicator values
    namespace = dict(_SAFE_BUILTINS)
    namespace["__builtins__"] = {}  # Block all default builtins

    # Inject all indicator values as variables
    for key, value in context.items():
        # Make safe variable names (replace hyphens, spaces)
        safe_key = key.replace("-", "_").replace(" ", "_")
        namespace[safe_key] = value

    # Initialize result variables
    namespace["entry"] = False
    namespace["exit"] = False

    try:
        exec(stripped, {"__builtins__": {}}, namespace)
    except Exception as e:
        logger.warning(f"Script evaluation error: {type(e).__name__}: {e}")
        return False

    # Read the result
    if mode == "exit":
        return bool(namespace.get("exit", False))
    else:
        return bool(namespace.get("entry", False))


def validate_script(code: str) -> dict:
    """Validate Python script syntax without executing it.

    Returns:
        {"valid": True} or {"valid": False, "error": "...", "line": N}
    """
    if not code or not code.strip():
        return {"valid": True}

    try:
        compile(code.strip(), "<condition_script>", "exec")
        return {"valid": True}
    except SyntaxError as e:
        return {
            "valid": False,
            "error": str(e.msg),
            "line": e.lineno,
            "offset": e.offset,
        }


def is_python_script(code) -> bool:
    """Detect if a condition string is a Python script vs JSON/text rules.

    Python scripts contain control flow, assignments, or multi-line logic.
    Simple text conditions are single-line field comparisons with AND/OR.
    """
    if not code or not isinstance(code, str):
        return False
    stripped = code.strip()
    # Python indicators: if/else, score, multi-line, indent, for/while
    return any(indicator in stripped for indicator in [
        '\nif ', '\n    ', 'score ', 'score=', 'entry =', 'exit =',
        'entry=', 'exit=', 'elif ', 'else:', 'for ', 'while ',
    ]) or stripped.startswith('if ') or stripped.startswith('score')


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED EVALUATION (handles both JSON rules and Python scripts)
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_conditions(rules, context: dict, mode: str = "entry") -> bool:
    """Evaluate conditions against an indicator context dict.

    Handles three formats:
    1. dict — JSON rule tree (from form builder)
    2. str (Python script) — if/else/score logic
    3. str (text rules) — "rsi14 < 70 AND htf_bias == bullish"

    Args:
        rules: JSON rule tree (dict), Python script (str), or text rules (str)
        context: Flat dict of indicator values (enriched payload)
        mode: "entry" or "exit" — for Python scripts

    Returns:
        True if conditions are met, False otherwise.
        Returns True if rules is None or empty (no conditions = pass).
    """
    if not rules:
        return True

    if isinstance(rules, str):
        stripped = rules.strip()
        if not stripped:
            return True

        # Check if it's a Python script
        if is_python_script(stripped):
            return evaluate_script(stripped, context, mode=mode)

        # Try to parse as JSON rule tree
        import json
        try:
            rules = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            # Try as simple text conditions
            parsed = parse_text_conditions(stripped)
            if parsed:
                return _evaluate_node(parsed, context)
            logger.warning(f"Invalid condition rules string: {stripped[:100]}")
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
