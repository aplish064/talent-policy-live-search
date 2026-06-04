from io import BytesIO
from dataclasses import dataclass, field
import re
from urllib.parse import urljoin, urlparse, urlunparse
from zipfile import BadZipFile

import fitz
from bs4 import BeautifulSoup
from docx import Document
from docx.opc.exceptions import PackageNotFoundError


DOCX_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
HTML_CONTENT_TYPES = {
    "application/xhtml+xml",
    "text/html",
}
PARSER_ERRORS = (BadZipFile, LookupError, PackageNotFoundError, fitz.FileDataError)
DATE_PATTERN = re.compile(r"(?<!\d)(20\d{2})[-年./]\s*(\d{1,2})[-月./]\s*(\d{1,2})")
POLICY_KEYWORDS = (
    "政策",
    "通知",
    "指南",
    "办法",
    "措施",
    "申报",
    "补贴",
    "津贴",
    "人才",
    "博士后",
    "资助",
    "奖励",
)
POLICY_SIGNALS = (
    "通知",
    "办法",
    "细则",
    "方案",
    "公告",
    "指南",
    "申报",
    "补贴",
    "津贴",
    "奖励",
    "认定",
    "入选",
    "申请",
    "支持",
    "计划",
    "名单",
    "名额",
    "标准",
    "流程",
    "条件",
)
TALENT_POLICY_SIGNALS = (
    "人才",
    "引进",
    "高层次",
    "博士后",
    "认定",
    "补贴",
    "资助",
    "奖励",
    "安家",
    "住房",
    "津贴",
    "落户",
    "职称",
)
GENERIC_MENU_TITLES = {
    "首页",
    "网站首页",
    "人才科",
    "人事科",
    "师资科",
    "劳资科",
    "综合科",
    "师德师风",
    "人才招聘",
    "服务指南",
    "下载专区",
    "政策文件",
    "政策法规",
    "通知公告",
    "新闻动态",
    "政务公开",
    "综合服务",
    "政策文件库",
    "政策解读",
    "解读回应",
}
APPLICATION_KEYWORDS = (
    "申报入口",
    "申报系统",
    "申请入口",
    "在线申报",
    "在线申请",
    "在线办理",
    "办理入口",
    "入口",
    "apply",
)
ATTACHMENT_KEYWORDS = ("附件", "下载", "材料", "表格", "申请表", "申报表")
ATTACHMENT_EXTENSIONS = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip", ".rar")


@dataclass(frozen=True)
class PageLink:
    text: str
    url: str
    snippet: str | None = None


@dataclass(frozen=True)
class PagePolicyItem:
    title: str
    url: str
    published_date: str | None = None
    snippet: str | None = None


@dataclass(frozen=True)
class PageExtraction:
    title: str | None
    fulltext: str
    headings: list[str] = field(default_factory=list)
    links: list[PageLink] = field(default_factory=list)
    policy_items: list[PagePolicyItem] = field(default_factory=list)
    application_links: list[PageLink] = field(default_factory=list)
    attachment_links: list[PageLink] = field(default_factory=list)
    date_candidates: list[str] = field(default_factory=list)


def extract_fulltext(content: bytes, content_type: str, max_chars: int) -> str:
    normalized_content_type, charset = _parse_content_type(content_type)

    try:
        if normalized_content_type in HTML_CONTENT_TYPES:
            return _extract_html_text(content, charset, max_chars)
        if normalized_content_type == "application/pdf":
            return _extract_pdf_text(content, max_chars)
        if normalized_content_type in DOCX_CONTENT_TYPES:
            return _extract_docx_text(content, max_chars)
        if normalized_content_type.startswith("text/"):
            return _normalize_and_truncate(_decode_text(content, charset), max_chars)
    except PARSER_ERRORS:
        return ""

    return ""


def extract_page_info(
    content: bytes,
    content_type: str,
    max_chars: int,
    base_url: str,
) -> PageExtraction:
    normalized_content_type, charset = _parse_content_type(content_type)

    if normalized_content_type in HTML_CONTENT_TYPES:
        try:
            return _extract_html_page_info(content, charset, max_chars, base_url)
        except PARSER_ERRORS:
            return PageExtraction(title=None, fulltext="")

    fulltext = extract_fulltext(content, content_type, max_chars)
    return PageExtraction(
        title=_first_line(fulltext),
        fulltext=fulltext,
        date_candidates=_date_candidates(fulltext),
    )


def _parse_content_type(content_type: str) -> tuple[str, str | None]:
    parts = content_type.split(";")
    normalized_content_type = parts[0].strip().lower()
    charset = None
    for parameter in parts[1:]:
        key, separator, value = parameter.partition("=")
        if separator and key.strip().lower() == "charset":
            charset = value.strip().strip("\"'") or None
            break

    return normalized_content_type, charset


def _decode_text(content: bytes, charset: str | None) -> str:
    try:
        return content.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def _extract_html_text(content: bytes, charset: str | None, max_chars: int) -> str:
    kwargs = {"from_encoding": charset} if charset else {}
    soup = BeautifulSoup(content, "lxml", **kwargs)
    for element in soup.find_all(["footer", "nav", "noscript", "script", "style"]):
        element.decompose()

    container = soup.body or soup
    return _normalize_and_truncate(container.get_text(separator="\n"), max_chars)


def _extract_html_page_info(
    content: bytes,
    charset: str | None,
    max_chars: int,
    base_url: str,
) -> PageExtraction:
    kwargs = {"from_encoding": charset} if charset else {}
    soup = BeautifulSoup(content, "lxml", **kwargs)
    for element in soup.find_all(["footer", "nav", "noscript", "script", "style"]):
        element.decompose()

    container = soup.body or soup
    fulltext = _normalize_and_truncate(container.get_text(separator="\n"), max_chars)
    links = _extract_page_links(container, base_url)
    policy_items = _extract_policy_items(container, base_url)
    return PageExtraction(
        title=_document_title(soup),
        fulltext=fulltext,
        headings=_dedupe_strings(
            [
                text
                for element in container.find_all(["h1", "h2", "h3"])
                if (text := _clean_text(element.get_text(" ")))
            ]
        )[:30],
        links=links,
        policy_items=policy_items,
        application_links=[link for link in links if _is_application_link(link)],
        attachment_links=[link for link in links if _is_attachment_link(link)],
        date_candidates=_date_candidates(fulltext),
    )


def _document_title(soup: BeautifulSoup) -> str | None:
    body = soup.body or soup
    for selector in ("h1", "h2", ".title"):
        element = body.select_one(selector)
        if element is not None and (text := _clean_text(element.get_text(" "))):
            return text
    if soup.title and (text := _clean_text(soup.title.get_text(" "))):
        return text
    return None


def _extract_page_links(container: BeautifulSoup, base_url: str) -> list[PageLink]:
    links: list[PageLink] = []
    for anchor in container.find_all("a"):
        href = anchor.get("href")
        text = _clean_text(anchor.get_text(" "))
        if not href or not text:
            continue
        url = _safe_join(base_url, str(href))
        if url is None:
            continue
        links.append(PageLink(text=text, url=url, snippet=_snippet(anchor)))
    return _dedupe_links(links)


def _extract_policy_items(container: BeautifulSoup, base_url: str) -> list[PagePolicyItem]:
    items: list[PagePolicyItem] = []
    for anchor in container.find_all("a"):
        href = anchor.get("href")
        title = _clean_text(anchor.get_text(" "))
        if not href or not title:
            continue
        url = _safe_join(base_url, str(href))
        if url is None:
            continue
        snippet = _snippet(anchor)
        published_date = _first_date(snippet or "")
        if not _looks_like_policy_item(title=title, snippet=snippet, published_date=published_date, url=url):
            continue
        items.append(
            PagePolicyItem(
                title=title,
                url=url,
                published_date=published_date,
                snippet=snippet,
            )
        )
    return _dedupe_policy_items(items)


def _looks_like_policy_item(
    title: str,
    snippet: str | None,
    published_date: str | None,
    url: str,
) -> bool:
    if _is_generic_menu_title(title) or _is_navigation_snippet(snippet):
        return False
    if _is_navigation_url(url):
        return False

    haystack = f"{title} {snippet or ''}"
    if any(keyword in haystack for keyword in POLICY_KEYWORDS):
        if not _has_talent_policy_signal(haystack):
            return False
        if _is_short_navigation_like(title, snippet):
            return False
        return True

    return published_date is not None and len(title) >= 8


def _has_talent_policy_signal(value: str) -> bool:
    return any(signal in value for signal in TALENT_POLICY_SIGNALS)


def _is_generic_menu_title(title: str) -> bool:
    normalized = " ".join(title.split())
    if not normalized:
        return False
    if normalized in GENERIC_MENU_TITLES:
        return True

    tokens = re.split(r"[|>｜/]", normalized)
    for token in tokens:
        cleaned = _clean_text(token)
        if cleaned in GENERIC_MENU_TITLES:
            return True

    if len(normalized) <= 8:
        return any(marker in normalized for marker in GENERIC_MENU_TITLES)
    return False


def _is_navigation_snippet(snippet: str | None) -> bool:
    if not snippet:
        return False
    normalized = _clean_text(snippet)
    if not normalized:
        return False
    if ">" in normalized and any(marker in normalized for marker in GENERIC_MENU_TITLES):
        return True
    if " | " in normalized and any(marker in normalized for marker in GENERIC_MENU_TITLES):
        return True
    if normalized in GENERIC_MENU_TITLES:
        return True
    return False


def _is_navigation_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if not path or path.endswith("/"):
        return len(path.strip("/").split("/")) <= 1
    return False


def _is_short_navigation_like(title: str, snippet: str | None) -> bool:
    if len(title) < 6:
        return True
    if len(title) < 10 and not any(keyword in (title + (snippet or "")) for keyword in POLICY_SIGNALS):
        return True
    return False


def _is_application_link(link: PageLink) -> bool:
    haystack = f"{link.text} {link.url}".casefold()
    return any(keyword.casefold() in haystack for keyword in APPLICATION_KEYWORDS)


def _is_attachment_link(link: PageLink) -> bool:
    path = urlparse(link.url).path.casefold()
    haystack = f"{link.text} {link.url}".casefold()
    return path.endswith(ATTACHMENT_EXTENSIONS) or any(
        keyword.casefold() in haystack for keyword in ATTACHMENT_KEYWORDS
    )


def _snippet(anchor) -> str | None:
    parent = anchor.find_parent(["li", "tr", "p", "div"]) or anchor
    return _clean_text(parent.get_text(" "))


def _date_candidates(text: str) -> list[str]:
    return _dedupe_strings([_format_date(match) for match in DATE_PATTERN.finditer(text)])


def _first_date(text: str) -> str | None:
    match = DATE_PATTERN.search(text)
    if match is None:
        return None
    return _format_date(match)


def _format_date(match: re.Match[str]) -> str:
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _safe_join(base_url: str, href: str) -> str | None:
    if "\\" in href or _has_control_chars(href):
        return None
    parsed_href = urlparse(href)
    if parsed_href.scheme and parsed_href.scheme.lower() not in {"http", "https"}:
        return None
    joined = urljoin(base_url, href)
    parsed = urlparse(joined)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    if "@" in parsed.netloc:
        return None
    return urlunparse(parsed._replace(fragment=""))


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = _clean_text(line)
        if stripped:
            return stripped
    return None


def _dedupe_links(links: list[PageLink]) -> list[PageLink]:
    seen: set[tuple[str, str]] = set()
    deduped: list[PageLink] = []
    for link in links:
        key = (link.text, link.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)
    return deduped


def _dedupe_policy_items(items: list[PagePolicyItem]) -> list[PagePolicyItem]:
    seen: set[tuple[str, str]] = set()
    deduped: list[PagePolicyItem] = []
    for item in items:
        key = (item.title, item.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _has_control_chars(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def _extract_pdf_text(content: bytes, max_chars: int) -> str:
    accumulator = _TextAccumulator(max_chars)
    with fitz.open(stream=content, filetype="pdf") as document:
        for page in document:
            if accumulator.add(page.get_text("text")):
                break

    return accumulator.text


def _extract_docx_text(content: bytes, max_chars: int) -> str:
    accumulator = _TextAccumulator(max_chars)
    document = Document(BytesIO(content))
    for paragraph in document.paragraphs:
        if accumulator.add(paragraph.text):
            break

    return accumulator.text


def _normalize_and_truncate(text: str, max_chars: int) -> str:
    accumulator = _TextAccumulator(max_chars)
    accumulator.add(text)
    return accumulator.text


class _TextAccumulator:
    def __init__(self, max_chars: int) -> None:
        self._max_chars = max(max_chars, 0)
        self._parts: list[str] = []
        self._length = 0

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def add(self, text: str) -> bool:
        if self._length >= self._max_chars:
            return True

        for raw_line in text.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue
            if self._append_line(line):
                return True

        return self._length >= self._max_chars

    def _append_line(self, line: str) -> bool:
        if self._parts:
            if self._append_text("\n"):
                return True

        return self._append_text(line)

    def _append_text(self, text: str) -> bool:
        remaining = self._max_chars - self._length
        if remaining <= 0:
            return True

        fragment = text[:remaining]
        self._parts.append(fragment)
        self._length += len(fragment)
        return len(fragment) < len(text) or self._length >= self._max_chars
