import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
import fitz  # PyMuPDF
import numpy as np
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError
from sentence_transformers import SentenceTransformer


# -------------------------
# Load .env (root)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# -------------------------
# Config
# -------------------------
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI is required (MongoDB Atlas).")

DB_NAME = os.getenv("DB_NAME", "lesson_search")
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")

PDF_DIR = Path(os.getenv("PDF_DIR", str(BASE_DIR / "data" / "pdfs")))
BATCH_INSERT = int(os.getenv("BATCH_INSERT", "500"))

LIMIT_PDFS = int(os.getenv("LIMIT_PDFS", "0"))   # 0 = no limit
LIMIT_PAGES = int(os.getenv("LIMIT_PAGES", "0")) # 0 = no limit


# -------------------------
# Section headers (跨頁狀態)
# -------------------------
SECTION_COMP_RE = re.compile(r"肆、\s*核心素養")
SECTION_LEARN_RE = re.compile(r"伍、\s*學習重點")

# ✅ 學習重點結束：遇到下一章（陸、... / 六、... / 6、...）
NEXT_SECTION_RE = re.compile(r"^(?:陸|六|6)\s*、")


# -------------------------
# Regex: codes
# -------------------------
PERF_CODE_RE = re.compile(r"\b\d{1,2}-[IVXⅠⅡⅢⅣⅤ]+-\d{1,2}\b")
CONTENT_CODE_RE = re.compile(r"\b[音視表美藝][EJ]-[IVXⅠⅡⅢⅣⅤ]+-\d{1,2}\b")
COMP_CODE_RE = re.compile(r"\b藝(?:-[EJ]|S-U)-[A-C]\d\b")

PAGE_NUM_RE = re.compile(r"^\d{1,3}$")


# ✅ 移除 raw 內重複前綴（避免 "學習內容 視E-Ⅲ-1:" 又出現一次）
ANY_CODE_PATTERN = (
    r"(?:"
    r"\d{1,2}-[IVXⅠⅡⅢⅣⅤ]+-\d{1,2}"
    r"|[音視表美藝][EJ]-[IVXⅠⅡⅢⅣⅤ]+-\d{1,2}"
    r"|藝(?:-[EJ]|S-U)-[A-C]\d"
    r")"
)
DUP_PREFIX_RE = re.compile(rf"(?:學習表現|學習內容|核心素養)\s*{ANY_CODE_PATTERN}\s*[:：]?\s*")


# -------------------------
# Helpers
# -------------------------
def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalize_for_e5(s: str, is_query: bool) -> str:
    return ("query: " if is_query else "passage: ") + (s or "").strip()


def clean_text_lines(text: str) -> List[str]:
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if PAGE_NUM_RE.fullmatch(ln):
            continue
        out.append(ln)
    return out


def iter_batches(items: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _strip_codes_and_noise(raw: str) -> str:
    raw = (raw or "").strip()
    # 先移除重複前綴
    raw = DUP_PREFIX_RE.sub("", raw)
    # 再移除各類 code
    raw = PERF_CODE_RE.sub("", raw)
    raw = CONTENT_CODE_RE.sub("", raw)
    raw = COMP_CODE_RE.sub("", raw)
    # normalize
    raw = re.sub(r"\s+", " ", raw).strip(" ：:，,")
    raw = raw.strip('\'"“”‘’「」『』()（）[]【】')
    return raw


def _find_first_code(line: str, mode: str) -> Optional[Tuple[str, str, int]]:
    """
    回傳 (type, code, end_index_of_code_in_line)
    mode:
      - "competency": 只找 COMP
      - "learning": 找 CONTENT + PERF，取最先出現的那個
    """
    matches = []

    if mode == "competency":
        m = COMP_CODE_RE.search(line)
        if not m:
            return None
        return ("core_competency_item", m.group(0), m.end())

    if mode == "learning":
        m_ct = CONTENT_CODE_RE.search(line)
        if m_ct:
            matches.append(("learning_content", m_ct.group(0), m_ct.start(), m_ct.end()))
        m_pf = PERF_CODE_RE.search(line)
        if m_pf:
            matches.append(("learning_performance", m_pf.group(0), m_pf.start(), m_pf.end()))

        if not matches:
            return None

        matches.sort(key=lambda x: x[2])  # 最早出現
        typ, code, _s, e = matches[0]
        return (typ, code, e)

    return None


# -------------------------
# Extract across whole doc (跨頁狀態機)
# -------------------------
def extract_items_from_doc(doc: fitz.Document) -> List[Dict[str, Any]]:
    """
    規則（跨頁）：
    - 遇到「肆、核心素養」→ mode="competency" 持續到遇到「伍、學習重點」
    - 遇到「伍、學習重點」→ mode="learning" 持續到遇到「陸、/六、/6、」
    - 重複的 (type, code, text) 直接跳過（跨整份 PDF）
    """
    out: List[Dict[str, Any]] = []
    seen = set()  # (type, code, text)

    mode: Optional[str] = None  # None | "competency" | "learning"
    cur_type: Optional[str] = None
    cur_code: Optional[str] = None
    cur_page: Optional[int] = None  # code 出現的頁碼
    cur_buf: List[str] = []

    def flush():
        nonlocal cur_type, cur_code, cur_buf, cur_page
        if not cur_type or not cur_code:
            cur_type, cur_code, cur_buf, cur_page = None, None, [], None
            return

        raw = _strip_codes_and_noise(" ".join(cur_buf))
        if raw:
            key = (cur_type, cur_code, raw)
            if key not in seen:
                seen.add(key)
                out.append({
                    "type": cur_type,
                    "code": cur_code,
                    "page": cur_page if cur_page is not None else 1,
                    "text": raw
                })

        cur_type, cur_code, cur_buf, cur_page = None, None, [], None

    total_pages = len(doc)
    for i in range(total_pages):
        page_no = i + 1
        lines = clean_text_lines(doc[i].get_text("text") or "")
        if not lines:
            continue

        for ln in lines:
            # 章節切換（跨頁）
            if SECTION_COMP_RE.search(ln):
                flush()
                mode = "competency"
                continue

            if SECTION_LEARN_RE.search(ln):
                flush()
                mode = "learning"
                continue

            # learning 結束：遇到下一章（你說的「六」那段）
            if mode == "learning" and NEXT_SECTION_RE.search(ln):
                flush()
                mode = None
                continue

            # 不在想抽的章節就不處理
            if mode is None:
                continue

            found = _find_first_code(ln, mode)
            if found:
                typ, code, end_idx = found
                flush()
                cur_type = typ
                cur_code = code
                cur_page = page_no

                rest = ln[end_idx:].strip(" ：:，,")
                if rest:
                    cur_buf.append(rest)
                continue

            # 沒遇到新 code：如果正在收集就加進去
            if cur_type and cur_code:
                cur_buf.append(ln)

    flush()
    return out


# -------------------------
# Main
# -------------------------
def main():
    if not PDF_DIR.exists():
        raise RuntimeError(f"PDF_DIR not found: {PDF_DIR}")

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    col_docs = db["docs"]
    col_items = db["items"]

    col_docs.create_index([("sha1", ASCENDING)], unique=True)
    col_items.create_index([("doc_sha1", ASCENDING), ("page", ASCENDING), ("type", ASCENDING)])
    col_items.create_index([("doc_sha1", ASCENDING), ("item_id", ASCENDING)], unique=True)

    model = SentenceTransformer(MODEL_NAME)

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if LIMIT_PDFS > 0:
        pdf_paths = pdf_paths[:LIMIT_PDFS]

    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}")
        return

    for pdf_path in pdf_paths:
        filename = pdf_path.name
        sha1 = file_sha1(pdf_path)

        if col_docs.find_one({"sha1": sha1}):
            print(f"Skip ingested: {filename}")
            continue

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        page_limit = min(total_pages, LIMIT_PAGES) if LIMIT_PAGES > 0 else total_pages

        # 如果你有 LIMIT_PAGES，這裡也要裁掉 doc（避免 state machine 掃到後面）
        if page_limit < total_pages:
            doc = fitz.open()  # new empty
            src = fitz.open(str(pdf_path))
            for i in range(page_limit):
                doc.insert_pdf(src, from_page=i, to_page=i)

        col_docs.insert_one({
            "sha1": sha1,
            "filename": filename,
            "pages": total_pages,
            "source": "CIRN",
        })

        extracted = extract_items_from_doc(doc)

        # 產 items（含 item_id、加前綴 text）
        counters_by_page = {}  # (page, type) -> idx

        items: List[Dict[str, Any]] = []
        for it in extracted:
            typ = it["type"]
            page_no = int(it["page"])

            idx = counters_by_page.get((page_no, typ), 0)
            counters_by_page[(page_no, typ)] = idx + 1

            prefix = {
                "learning_performance": "pf",
                "learning_content": "ct",
                "core_competency_item": "cc",
            }.get(typ, "it")

            item_id = f"p{page_no:04d}_{prefix}{idx:04d}"

            zh_prefix = {
                "learning_performance": "學習表現",
                "learning_content": "學習內容",
                "core_competency_item": "核心素養",
            }.get(typ, "項目")

            text = f"{zh_prefix} {it['code']}: {it['text']}"

            items.append({
                "doc_sha1": sha1,
                "filename": filename,
                "page": page_no,
                "item_id": item_id,
                "type": typ,
                "code": it["code"],
                "text": text,
            })

        # Embedding
        if items:
            texts = [normalize_for_e5(d["text"], is_query=False) for d in items]
            emb = model.encode(texts, normalize_embeddings=True, batch_size=32)
            emb = np.asarray(emb, dtype=np.float32)
            for d, v in zip(items, emb):
                d["embedding"] = v.tolist()

        # Insert batches
        inserted = 0
        for batch in iter_batches(items, BATCH_INSERT):
            if not batch:
                continue
            try:
                col_items.insert_many(batch, ordered=False)
                inserted += len(batch)
            except BulkWriteError as e:
                dup = sum(1 for we in e.details.get("writeErrors", []) if we.get("code") == 11000)
                ok = len(batch) - dup
                inserted += max(ok, 0)
                print(f"⚠️ BulkWriteError: duplicated={dup}, inserted={ok}")

        c_pf = sum(1 for x in items if x["type"] == "learning_performance")
        c_ct = sum(1 for x in items if x["type"] == "learning_content")
        c_cc = sum(1 for x in items if x["type"] == "core_competency_item")

        print(
            f"Ingested {filename}: items={inserted} "
            f"(performance={c_pf}, content={c_ct}, competency={c_cc}), "
            f"pages_scanned={page_limit}/{total_pages}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
