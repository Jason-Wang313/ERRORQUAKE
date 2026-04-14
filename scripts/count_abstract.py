"""Count words in the abstract of main.tex."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
body = m.group(1)
# Remove LaTeX commands and math
clean = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
clean = re.sub(r"[{}$]", " ", clean)
clean = re.sub(r"\[[^\]]*\]", " ", clean)
words = [w for w in clean.split() if w and any(c.isalpha() or c.isdigit() for c in w)]
print(f"Abstract word count: {len(words)}")
print(f"First 12 words: {' '.join(words[:12])}")
print(f"Last 12 words: {' '.join(words[-12:])}")
