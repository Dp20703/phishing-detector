"""
URL feature extraction.

This single module is used by BOTH training (ml/train.py) and the API
(backend/main.py), so the model always sees features computed the same way.

Key design decisions
--------------------
* The training dataset stores URLs WITHOUT a scheme (no "https://") and
  people paste full links into the app. `normalize_url` removes the scheme,
  a leading "www." and the fragment, so both sides look identical.
* Features that were constant in the training data (is_https, has_at, has_ip)
  are NOT model features. They are still reported to the user as rule-based
  "signals" in `get_signals`, but the model never relies on them.
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract

# Offline-safe: use the suffix list bundled with the package and never make a
# network request at runtime (important on hosted/free-tier servers).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account",
    "update", "bank", "confirm", "webscr", "password",
]

# Well-known brands that phishers like to imitate.
BRANDS = [
    "paypal", "ebay", "amazon", "apple", "google", "microsoft", "facebook",
    "netflix", "instagram", "whatsapp", "dropbox", "linkedin", "twitter",
    "paytm", "sbi", "hdfc", "icici", "fedex", "dhl",
]

# Order matters: it is saved with the model and reused by the API.
FEATURE_NAMES = [
    "url_length",
    "host_length",
    "path_length",
    "path_depth",
    "num_dots",
    "num_hyphens",
    "host_hyphens",
    "num_digits",
    "host_digits",
    "digit_ratio",
    "num_subdomains",
    "num_params",
    "suspicious_in_domain",
    "suspicious_in_rest",
    "brand_mismatch",
    "entropy",
]


def normalize_url(url: str) -> str:
    """Strip scheme, 'www.', fragment and whitespace; ensure a path exists."""
    url = (url or "").strip()
    url = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", url)  # scheme
    url = url.split("#", 1)[0]                              # fragment
    url = re.sub(r"^www\d*\.", "", url, flags=re.IGNORECASE)
    if "/" not in url and "?" not in url:
        url += "/"
    return url


def _host_of(normalized: str) -> str:
    parsed = urlparse("http://" + normalized)
    return (parsed.hostname or "").lower()


def _registered_domain(host: str) -> str:
    ext = _EXTRACT(host)
    return f"{ext.domain}.{ext.suffix}".strip(".") if ext.suffix else ext.domain


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def registered_domain_of(url: str) -> str:
    """Public helper used by training to group URLs by site."""
    return _registered_domain(_host_of(normalize_url(url)))


def extract_features(url: str) -> dict:
    norm = normalize_url(url)
    parsed = urlparse("http://" + norm)
    host = (parsed.hostname or "").lower()
    ext = _EXTRACT(host)
    lower = norm.lower()

    reg_domain = _registered_domain(host)
    non_domain_part = lower.replace(reg_domain, "", 1) if reg_domain else lower
    brand_mismatch = int(any(b in non_domain_part for b in BRANDS)
                         and not any(b in reg_domain for b in BRANDS))

    n = max(len(norm), 1)
    num_digits = sum(c.isdigit() for c in norm)

    return {
        "url_length": len(norm),
        "host_length": len(host),
        "path_length": len(parsed.path),
        "path_depth": parsed.path.count("/"),
        "num_dots": norm.count("."),
        "num_hyphens": norm.count("-"),
        "host_hyphens": host.count("-"),
        "num_digits": num_digits,
        "host_digits": sum(c.isdigit() for c in host),
        "digit_ratio": num_digits / n,
        "num_subdomains": len([s for s in ext.subdomain.split(".") if s]),
        "num_params": norm.count("="),
        # A word like "bank" inside the site's own domain (hdfcbank.com) means
        # something different from the same word in a subdomain or path
        # (secure-update.xyz/bank/login), so they are counted separately.
        "suspicious_in_domain": sum(w in reg_domain for w in SUSPICIOUS_WORDS),
        "suspicious_in_rest": sum(w in non_domain_part for w in SUSPICIOUS_WORDS),
        "brand_mismatch": brand_mismatch,
        "entropy": _entropy(lower),
    }


def feature_vector(url: str) -> list:
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]


# --------------------------------------------------------------------------
# Human-readable, RULE-BASED signals (shown in the UI).
# These are simple checks on the URL text. They are NOT the model's internal
# reasoning, so the UI/README label them as "signals", not "explanations".
# --------------------------------------------------------------------------
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def get_signals(raw_url: str, max_items: int = 6) -> list:
    raw = (raw_url or "").strip()
    norm = normalize_url(raw)
    feats = extract_features(raw)
    host = _host_of(norm)
    lower = norm.lower()
    signals = []

    if _IPV4.match(host) or re.fullmatch(r"(0x[0-9a-f]+|\d+)", host or ""):
        signals.append("Uses an IP address instead of a domain name")
    if "@" in norm.split("/", 1)[0]:
        signals.append("Contains '@' before the host (can hide the real destination)")
    if raw.lower().startswith("http://"):
        signals.append("Uses plain HTTP (not HTTPS)")
    if feats["brand_mismatch"]:
        signals.append("Mentions a well-known brand but the site's domain is different")
    for word in SUSPICIOUS_WORDS:
        if word in lower:
            signals.append(f"Contains the word '{word}'")
    if feats["num_subdomains"] >= 3:
        signals.append("Has many subdomains")
    if feats["host_hyphens"] >= 2:
        signals.append("Domain contains several hyphens")
    if feats["host_digits"] >= 4:
        signals.append("Domain contains many digits")
    if feats["url_length"] >= 100:
        signals.append("URL is unusually long")
    if feats["num_params"] >= 3:
        signals.append("Has many query parameters")

    if not signals:
        signals.append("No obvious suspicious patterns in the URL text")
    return signals[:max_items]
