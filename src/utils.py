import re
import iso8601
import datetime
import tempfile
import os
from datetime import timedelta, timezone

_ZWSP = "\u200b"

def get_temp_path(filename):
    """Get a temporary file path that works across different environments."""
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, filename)

def extract_ids_from_curl(curl_string: str):
    """
    Pull course_id, section_id, class_id from the Referer in the cURL (if present).
    Fallback: try to find classes/<digits> anywhere in the cURL.
    """
    out = {"course_id": None, "section_id": None, "class_id": None, "referer": None}
    # Referer header
    ref_match = re.search(r"-H\s+'referer:\s*(.*?)'", curl_string, re.IGNORECASE)
    if ref_match:
        ref = ref_match.group(1)
        out["referer"] = ref
        m = re.search(r"/courses/(\d+)/sections/(\d+)/classes/(\d+)", ref)
        if m:
            out["course_id"], out["section_id"], out["class_id"] = m.group(1), m.group(2), m.group(3)

    if not out["class_id"]:
        m2 = re.search(r"/classes/(\d+)", curl_string)
        if m2:
            out["class_id"] = m2.group(1)
    return out

def derive_class_link_from_curl(curl_string: str, course_id: str = None, section_id: str = None, class_id: str = None):
    """
    If referer exists, return it; else build link if we have ids; else empty string.
    """
    ref_match = re.search(r"-H\s+'referer:\s*(.*?)'", curl_string, re.IGNORECASE)
    if ref_match:
        return ref_match.group(1)
    if course_id and section_id and class_id:
        return f"https://forum.minerva.edu/app/courses/{course_id}/sections/{section_id}/classes/{class_id}"
    if class_id:
        return f"https://forum.minerva.edu/app/classes/{class_id}"
    return ""

def _safe_date(date_str):
    # return YYYY-MM-DD from ISO8601, or ""
    if not date_str:
        return ""
    try:
        return date_str.split('T')[0]
    except Exception:
        return ""

def _fmt_mmss(seconds_float):
    if seconds_float is None:
        return ""
    seconds = max(0, int(seconds_float))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

def normalize_sentence_spacing(text: str) -> str:
    """
    Fixes missing spaces after sentence punctuation and collapses over-spacing.
    Handles cases like "taught CS51.So you've" -> "taught CS51. So you've"
    """
    if not text:
        return text
    # Ensure space after . ! ? when followed by letter/quote/number
    text = re.sub(r'([.!?])(?=[A-Za-z0-9"\'])', r'\1 ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    # Trim space before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    # Make sure quotes then letter also have space before if needed: already covered by first rule most times
    return text.strip()

def soft_break_long_token(s: str, every: int = 14) -> str:
    """
    Insert zero-width breaks into very long tokens to avoid PDF table overlap.
    Keeps spaces intact; only breaks long runs of non-space characters.
    """
    if not s:
        return s
    parts = []
    for token in re.split(r"(\s+)", s):
        if token.strip() == "":
            parts.append(token)
        else:
            # insert breaks every N characters
            chunks = [token[i:i+every] for i in range(0, len(token), every)]
            parts.append(_ZWSP.join(chunks))
    return "".join(parts)

def clean_curl(curl_string):
   headers = {}

   # Extract headers with -H flag
   header_matches = re.findall(r"-H ['\"](.*?): (.*?)['\"]", curl_string)
   for name, value in header_matches:
       headers[name] = value

   # Extract cookies with -b flag
   cookie_match = re.search(r"-b ['\"](.*?)['\"]", curl_string)
   if cookie_match:
       cookie_str = cookie_match.group(1)
       headers['Cookie'] = cookie_str

   # Ensure we look like XHR JSON
   headers.setdefault("accept", "application/json, text/javascript, */*; q=0.01")
   headers.setdefault("x-requested-with", "XMLHttpRequest")
   return headers
