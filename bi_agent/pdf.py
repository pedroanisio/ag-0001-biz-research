"""Typeset the Markdown report as a PDF with reportlab.

The PDF is a second rendering of ``report.md``, not a second report: it reads the Markdown that
:mod:`bi_agent.report` produced (a small, known subset: headings, paragraphs, bullets, pipe
tables, block quotes, bold and italic) and lays it out with a cover page, a table of contents,
PDF bookmarks, a running header naming the current section, classification tags and citation
groups that link to their row in the Sources table. Tables too wide for the page (offerings,
competitors) are set as one card per row. The cover repeats headline facts from the Company
Snapshot and counts the report's own classification labels; nothing else is added.

The built-in Helvetica family covers the Latin-1 / cp1252 range, which is every character the
five report languages need; anything outside it is transliterated or replaced.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from .i18n import Translator

# --------------------------------------------------------------------------- look

INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#616E7C")
ACCENT = colors.HexColor("#1D4E89")
ACCENT_SOFT = colors.HexColor("#E8EEF6")
RULE = colors.HexColor("#CBD2D9")
ZEBRA = colors.HexColor("#F5F7FA")
NOTE_BG = colors.HexColor("#FFF8E6")
NOTE_EDGE = colors.HexColor("#E0A800")

# One colour per classification, so a reader can see at a glance how a claim is grounded.
CLASS_COLORS: dict[str, str] = {
    "verified_fact": "#1E7B45",
    "company_claim": "#1D4E89",
    "third_party_claim": "#7B4BA8",
    "analytical_inference": "#B35C00",
    "unknown": "#7B8794",
}

# The same hue mixed with white, as the background of a classification tag.
CLASS_TINTS: dict[str, str] = {
    "verified_fact": "#E3F1E8",
    "company_claim": "#E4ECF6",
    "third_party_claim": "#EFE7F6",
    "analytical_inference": "#F8EBDD",
    "unknown": "#EDEFF2",
}
CHIP_BG = "#EDEFF2"

PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
MARGIN_TOP = 22 * mm
MARGIN_BOTTOM = 20 * mm
BODY_W = PAGE_W - 2 * MARGIN_X

FONT, BOLD, ITALIC, BOLD_ITALIC = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("body", fontName=FONT, fontSize=9.5, leading=13.6, textColor=INK, alignment=TA_LEFT,
                          spaceAfter=5)
    return {
        "body": base,
        "h1": ParagraphStyle("h1", parent=base, fontName=BOLD, fontSize=15, leading=19, textColor=ACCENT,
                             spaceBefore=6, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base, fontName=BOLD, fontSize=11.5, leading=15, textColor=INK,
                             spaceBefore=10, spaceAfter=5),
        "label": ParagraphStyle("label", parent=base, fontName=BOLD, fontSize=9.5, leading=13, textColor=ACCENT,
                                spaceBefore=6, spaceAfter=3),
        "bullet": ParagraphStyle("bullet", parent=base, spaceAfter=2.5),
        "note": ParagraphStyle("note", parent=base, fontSize=9, leading=12.5, textColor=INK, spaceAfter=0),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8, leading=10.2, spaceAfter=0),
        "cell_small": ParagraphStyle("cell_small", parent=base, fontSize=6.8, leading=8.6, spaceAfter=0),
        "head": ParagraphStyle("head", parent=base, fontName=BOLD, fontSize=8, leading=10, textColor=colors.white,
                               spaceAfter=0),
        "head_small": ParagraphStyle("head_small", parent=base, fontName=BOLD, fontSize=6.8, leading=8.6,
                                     textColor=colors.white, spaceAfter=0),
        "toc1": ParagraphStyle("toc1", parent=base, fontSize=10, leading=15, leftIndent=0, spaceAfter=0),
        "toc_title": ParagraphStyle("toc_title", parent=base, fontName=BOLD, fontSize=15, leading=19,
                                    textColor=ACCENT, spaceAfter=12),
        "cover_kicker": ParagraphStyle("cover_kicker", parent=base, fontName=BOLD, fontSize=10, leading=13,
                                       textColor=colors.HexColor("#BCCCDC")),
        "cover_title": ParagraphStyle("cover_title", parent=base, fontName=BOLD, fontSize=30, leading=35,
                                      textColor=colors.white),
        "cover_meta": ParagraphStyle("cover_meta", parent=base, fontSize=10, leading=15, textColor=INK),
        "cover_key": ParagraphStyle("cover_key", parent=base, fontSize=8.5, leading=12, textColor=MUTED),
        "cover_section": ParagraphStyle("cover_section", parent=base, fontName=BOLD, fontSize=8, leading=11,
                                        textColor=ACCENT, spaceBefore=0, spaceAfter=3),
        "fact_label": ParagraphStyle("fact_label", parent=base, fontName=BOLD, fontSize=7.5, leading=10,
                                     textColor=MUTED, spaceAfter=0),
        "fact": ParagraphStyle("fact", parent=base, fontSize=8.5, leading=11.5, spaceAfter=0),
        "card_title": ParagraphStyle("card_title", parent=base, fontSize=9.5, leading=13, textColor=INK, spaceAfter=0),
        "card_label": ParagraphStyle("card_label", parent=base, fontName=BOLD, fontSize=7.5, leading=10.2,
                                     textColor=MUTED, spaceAfter=0),
        "caption": ParagraphStyle("caption", parent=base, fontSize=8, leading=11, textColor=MUTED, spaceAfter=4),
    }


# --------------------------------------------------------------------------- text

_CP1252_FALLBACK = {"→": "->", "←": "<-", "≈": "~", "≥": ">=", "≤": "<=", "≠": "!=", "×": "x", "−": "-",
                    "‑": "-", " ": " ", " ": " ", " ": " ", "✓": "v", "★": "*"}


def _encodable(text: str) -> str:
    """Keep what Helvetica (cp1252) can draw; transliterate or drop the rest."""
    out = []
    for ch in text:
        try:
            ch.encode("cp1252")
            out.append(ch)
            continue
        except UnicodeEncodeError:
            pass
        if ch in _CP1252_FALLBACK:
            out.append(_CP1252_FALLBACK[ch])
            continue
        folded = unicodedata.normalize("NFKD", ch).encode("cp1252", "ignore").decode("cp1252")
        out.append(folded or "?")
    return "".join(out)


def _nobreak(text: str) -> str:
    """Escaped text that never wraps inside (a tag reads as one unit)."""
    return escape(_encodable(text)).replace(" ", "&nbsp;")


_CITE = re.compile(r"\[(E\d{3,})\]")
_CITE_RUN = re.compile(r"\[E\d{3,}\](?:\s*\[E\d{3,}\])*")  # "[E001][E003] [E061]" is one citation group
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CLASS = re.compile(r"\*\(([^()*]+)\)\*")
_ITALIC = re.compile(r"(?<![\w*])[*_]([^*_\n]+?)[*_](?![\w*])")
_URL = re.compile(r"(?<![\"'=>])(https?://[^\s<|]+)")


class Inline:
    """Markdown inline markup to reportlab paragraph markup."""

    def __init__(self, t: Translator, targets: set[str] | frozenset[str] = frozenset()) -> None:
        self.class_by_label = {t(k).lower(): k for k in CLASS_COLORS}
        self.targets = targets  # evidence ids with a row (a link destination) in the Sources table

    def __call__(self, text: str, *, links: bool = True) -> str:
        s = escape(_encodable(text.replace("\\|", "|")))
        urls: list[str] = []

        def stash(m: re.Match) -> str:  # keep URLs away from the emphasis patterns
            urls.append(m.group(1))
            return f"\x00{len(urls) - 1}\x00"

        s = _URL.sub(stash, s)
        s = _CLASS.sub(self._class_tag, s)
        s = _BOLD.sub(r"<b>\1</b>", s)
        s = _ITALIC.sub(r"<i>\1</i>", s)
        s = re.sub("\x00(\\d+)\x00", lambda m: self._url(urls[int(m.group(1))], links), s)
        s = s.replace("\n", "<br/>")
        return _CITE_RUN.sub(self._cites, s)

    def _cites(self, m: re.Match) -> str:
        """One compact bracket per run of citations, each id linked to its Sources row."""
        ids = sorted(dict.fromkeys(_CITE.findall(m.group(0))), key=lambda i: int(i[1:]))
        linked = [f'<a href="#{i}" color="#1D4E89">{i}</a>' if i in self.targets else i for i in ids]
        return f'<font size="-1.5" color="#616E7C">[{", ".join(linked)}]</font>'

    def tag(self, key: str, label: str) -> str:
        """A classification as a small tinted tag."""
        return (f'<span backColor="{CLASS_TINTS[key]}" color="{CLASS_COLORS[key]}"><font size="-2.5">'
                f'<b>&nbsp;{_nobreak(label.upper())}&nbsp;</b></font></span>')

    def basis(self, text: str) -> str:
        """A table "Basis" cell ("Company claim [E001] [E002]") as a tag plus its citations."""
        for label, key in sorted(self.class_by_label.items(), key=lambda kv: -len(kv[0])):
            if text.lower().startswith(label):
                return self.tag(key, text[:len(label)]) + " " + self(text[len(label):].strip())
        return self(text)

    @staticmethod
    def _url(url: str, link: bool) -> str:
        return f'<a href="{url}" color="#1D4E89">{url}</a>' if link else url

    def _class_tag(self, m: re.Match) -> str:
        key = self.class_by_label.get(m.group(1).strip().lower())
        if key is None:
            return f"<i>({m.group(1)})</i>"
        return self.tag(key, m.group(1).strip())


# --------------------------------------------------------------------------- markdown blocks


@dataclass
class Block:
    kind: str  # title, h1, h2, label, para, bullets, table, quote
    text: str = ""
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


def _split_row(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip() for c in cells]


def parse_markdown(md: str) -> list[Block]:
    """Split the report Markdown into blocks. Only the subset report.py writes is recognised."""
    blocks: list[Block] = []
    lines = md.splitlines()
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            hard = any(p.endswith("  ") for p in para[:-1])  # Markdown hard line breaks
            text = "\n".join(p.rstrip() for p in para) if hard else " ".join(p.strip() for p in para)
            blocks.append(Block("para", text=text.strip()))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("# "):
            flush()
            blocks.append(Block("title", text=stripped[2:].strip()))
        elif stripped.startswith("## "):
            flush()
            blocks.append(Block("h1", text=stripped[3:].strip()))
        elif stripped.startswith("### "):
            flush()
            blocks.append(Block("h2", text=stripped[4:].strip()))
        elif stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = _split_row(lines[i])
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in row if c):
                    rows.append(row)
                i += 1
            blocks.append(Block("table", rows=rows))
            continue
        elif stripped.startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            blocks.append(Block("bullets", items=items))
            continue
        elif stripped.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(Block("quote", text=" ".join(quote)))
            continue
        elif re.fullmatch(r"\*\*[^*]+\*\*", stripped) and not para:
            blocks.append(Block("label", text=stripped[2:-2]))
        else:
            para.append(line.rstrip("\n"))
        i += 1
    flush()
    return blocks


# --------------------------------------------------------------------------- flowables


class SectionHeading(Paragraph):
    """A numbered section heading that registers itself in the TOC and the PDF outline."""

    def __init__(self, text: str, style: ParagraphStyle, key: str) -> None:
        super().__init__(text, style)
        self.key = key
        self.plain = re.sub(r"<[^>]+>", "", text)

    def draw(self) -> None:
        # The running header shows the first section that starts on a page, else the one carried over.
        self.canv._bi_last = self.plain
        if not getattr(self.canv, "_bi_set", False):
            self.canv._bi_section, self.canv._bi_set = self.plain, True
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.plain, self.key, level=0, closed=False)
        # A rule under the heading, full body width.
        self.canv.setStrokeColor(RULE)
        self.canv.setLineWidth(0.6)
        self.canv.line(0, -3, self._width_available, -3)
        super().draw()

    def wrap(self, availWidth: float, availHeight: float):  # noqa: N803 - reportlab API
        self._width_available = availWidth
        return super().wrap(availWidth, availHeight)


class Anchor(Flowable):
    """Zero-size named destination, the target of a citation link."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def wrap(self, *_args):
        return 0, 0

    def draw(self) -> None:
        self.canv.bookmarkHorizontal(self.name, 0, 0)


class _Doc(BaseDocTemplate):
    def afterFlowable(self, flowable: Flowable) -> None:  # noqa: N802 - reportlab API
        if isinstance(flowable, SectionHeading):
            self.notify("TOCEntry", (0, flowable.plain, self.page, flowable.key))


def _numbered_canvas(page_label: str, header: str):
    """A canvas that knows the total page count, for "page X of Y" footers."""

    class NumberedCanvas(Canvas):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._saved: list[dict] = []

        def showPage(self) -> None:  # noqa: N802 - reportlab API
            self._saved.append(dict(self.__dict__))
            self._bi_section, self._bi_set = getattr(self, "_bi_last", ""), False  # carried to the next page
            self._startPage()

        def save(self) -> None:
            total = len(self._saved)
            for state in self._saved:
                self.__dict__.update(state)
                if self._pageNumber > 1:
                    self._chrome(total, state.get("_bi_section", ""))
                super().showPage()
            super().save()

        def _chrome(self, total: int, section: str) -> None:
            self.saveState()
            self.setFont(FONT, 7.5)
            self.setFillColor(MUTED)
            self.drawString(MARGIN_X, PAGE_H - 12 * mm, header)
            if section:
                self.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 12 * mm,
                                     section if len(section) <= 60 else section[:57] + "...")
            self.setStrokeColor(RULE)
            self.setLineWidth(0.5)
            self.line(MARGIN_X, PAGE_H - 13.8 * mm, PAGE_W - MARGIN_X, PAGE_H - 13.8 * mm)
            self.line(MARGIN_X, 13.5 * mm, PAGE_W - MARGIN_X, 13.5 * mm)
            self.drawRightString(PAGE_W - MARGIN_X, 9.5 * mm, page_label.format(n=self._pageNumber, total=total))
            self.restoreState()

    return NumberedCanvas


def _cover_background(canvas: Canvas, _doc: BaseDocTemplate) -> None:
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H * 0.52, PAGE_W, PAGE_H * 0.48, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#163D6C"))
    canvas.rect(0, PAGE_H * 0.52, PAGE_W, 4 * mm, stroke=0, fill=1)
    canvas.restoreState()


# --------------------------------------------------------------------------- tables


def _col_widths(rows: list[list[str]], total: float, ncols: int, font_size: float) -> list[float]:
    """Give each column room for its longest word (so dates and ids never break), then share the
    rest by how much text the column holds."""
    char_w = font_size * 0.64  # Helvetica caps and digits run ~0.56-0.72 em
    padding = 10
    floors, weights = [], []
    for c in range(ncols):
        texts = [re.sub(r"\*|\[E\d+\]", "", r[c]) if c < len(r) else "" for r in rows]
        words = [len(w) for text in texts for w in text.split() if not w.startswith("http")]
        floor = min(max(words, default=4) * char_w + padding, 34 * mm)
        if sum(x.startswith("http") for x in texts) > len(texts) / 2:  # a URL column: wraps mid-word
            floor = max(floor, 48 * mm)
        floors.append(floor)
        lengths = sorted(len(x) for x in texts)
        weights.append(max(lengths[int(len(lengths) * 0.8)], 4) ** 0.85)
    excess = sum(floors) - total
    if excess > 0:  # too many columns: take the room from the wide ones, never from ids or dates
        wide = [max(f - 25 * mm, 0) for f in floors]
        floors = [f - excess * w / sum(wide) for f, w in zip(floors, wide)] if sum(wide) > excess else floors
    spare = max(total - sum(floors), 0)
    widths = [f + spare * w / sum(weights) for f, w in zip(floors, weights)]
    scale = total / sum(widths)
    return [w * scale for w in widths]


def _table(block: Block, st: dict[str, ParagraphStyle], inline: Inline, anchors: bool) -> Table:
    ncols = max(len(r) for r in block.rows)
    small = ncols >= 6
    cell, head = (st["cell_small"], st["head_small"]) if small else (st["cell"], st["head"])
    data = []
    for r, row in enumerate(block.rows):
        row = row + [""] * (ncols - len(row))
        if r == 0:
            data.append([Paragraph(inline(c, links=False), head) for c in row])
            continue
        cells: list = []
        for c, text in enumerate(row):
            para = Paragraph(inline(text), cell)
            if anchors and c == 0 and re.fullmatch(r"E\d{3,}", text):
                cells.append([Anchor(text), para])
            else:
                cells.append(para)
        data.append(cells)
    widths = _col_widths(block.rows, BODY_W, ncols, cell.fontSize)
    if ncols == 2:  # field/value tables: a narrow label column
        widths = [BODY_W * 0.27, BODY_W * 0.73]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    if ncols == 2:
        style.append(("FONTNAME", (0, 1), (0, -1), BOLD))
        style.append(("BACKGROUND", (0, 1), (0, -1), ACCENT_SOFT))
    table.setStyle(TableStyle(style))
    return table


# --------------------------------------------------------------------------- cards, sources, cover data


def _chip(text: str) -> str:
    return (f'<span backColor="{CHIP_BG}" color="#3E4C59"><font size="-2.5">&nbsp;{_nobreak(text)}'
            f'&nbsp;</font></span>')


def _column(rows: list[list[str]], t: Translator, key: str) -> int | None:
    header = [h.strip().lower() for h in rows[0]]
    label = t(key).lower()
    return header.index(label) if label in header else None


def _cards(block: Block, st: dict[str, ParagraphStyle], inline: Inline, t: Translator) -> list[Flowable]:
    """A wide table (offerings, competitors) as one card per row: the first column is the card's
    title, type/category and basis become tags beside it, every other column a labelled line.
    Eight narrow columns cannot hold sentences without breaking words; a card can."""
    head = block.rows[0]
    chips = [c for c in (_column(block.rows, t, "type"), _column(block.rows, t, "category")) if c is not None]
    basis = _column(block.rows, t, "basis")
    out: list[Flowable] = []
    for row in block.rows[1:]:
        row = row + [""] * (len(head) - len(row))
        if row[0].strip() in ("", "—"):  # the "nothing found" placeholder row
            out.append(Paragraph(f"<i>{inline(row[-1])}</i>", st["body"]))
            continue
        tags = "&nbsp;".join(_chip(row[c]) for c in chips if row[c].strip() not in ("", "—"))
        title = f"<b>{inline(row[0], links=False)}</b>"
        if tags:
            title += "&nbsp;&nbsp;" + tags
        if basis is not None and row[basis].strip():
            title += "&nbsp;&nbsp;" + inline.basis(row[basis])
        data: list[list] = [[Paragraph(title, st["card_title"]), ""]]
        for c, name in enumerate(head):
            if c == 0 or c in chips or c == basis or not row[c].strip():
                continue
            data.append([Paragraph(inline(name, links=False), st["card_label"]), Paragraph(inline(row[c]), st["cell"])])
        card = Table(data, colWidths=[BODY_W * 0.22, BODY_W * 0.78], hAlign="LEFT")
        card.setStyle(TableStyle([
            ("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (-1, 0), ACCENT_SOFT),
            ("BOX", (0, 0), (-1, -1), 0.5, RULE), ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        out += [KeepTogether([card]), Spacer(1, 6)]
    return out


def _lean_sources(block: Block, t: Translator) -> tuple[Block, str | None]:
    """The Sources table without columns that say nothing: Published when no row has a date,
    Accessed when every row has the same date (said once in a caption instead), and Type folded
    into Source kind ("news · third party")."""
    rows = [r[:] for r in block.rows]
    caption = None
    drop: set[int] = set()
    published = _column(rows, t, "published")
    if published is not None and all(r[published].strip() in ("", t("n_a")) for r in rows[1:]):
        drop.add(published)
    accessed = _column(rows, t, "accessed")
    if accessed is not None and rows[1:] and len({r[accessed] for r in rows[1:]}) == 1:
        drop.add(accessed)
        caption = t("accessed_all", date=rows[1][accessed])
    kind, kind_of = _column(rows, t, "type"), _column(rows, t, "source_kind")
    if kind is not None and kind_of is not None:
        for r in rows[1:]:
            r[kind_of] = f"{r[kind_of]} · {r[kind]}" if r[kind_of].strip() not in ("", "—") else r[kind]
        drop.add(kind)
    return Block("table", rows=[[c for i, c in enumerate(r) if i not in drop] for r in rows]), caption


_STRIP = re.compile(r"\*\([^()*]+\)\*|\[E\d{3,}\]|\*\*")


def _plain(text: str, limit: int = 150) -> str:
    text = " ".join(_STRIP.sub("", text.replace("\\|", "|")).split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _key_facts(blocks: list[Block], t: Translator) -> list[tuple[str, str]]:
    """Headline facts for the cover, taken from the Company Snapshot table."""
    heading = f"2. {t('s2')}"
    at = next((i for i, b in enumerate(blocks) if b.kind == "h1" and b.text == heading), None)
    table = next((b for b in blocks[at:] if b.kind == "table"), None) if at is not None else None
    if table is None:
        return []
    values = {r[0].strip(): r[1] for r in table.rows[1:] if len(r) > 1}
    facts = []
    for key in ("website_subject", "headquarters", "founded", "ownership", "industry", "business_model"):
        value = values.get(t(key), "")
        if value and value.strip() != t("unknown"):
            facts.append((t(key), _plain(value)))
    return facts


def _evidence_counts(blocks: list[Block], inline: Inline) -> dict[str, int]:
    """How many claims in the report carry each classification (inline tags and Basis cells)."""
    counts = dict.fromkeys(CLASS_COLORS, 0)
    labels = sorted(inline.class_by_label.items(), key=lambda kv: -len(kv[0]))
    texts = [b.text for b in blocks] + [x for b in blocks for x in b.items]
    cells = [c for b in blocks if b.kind == "table" for r in b.rows[1:] for c in r]
    for text in texts + cells:
        for m in _CLASS.finditer(text):
            key = inline.class_by_label.get(m.group(1).strip().lower())
            if key:
                counts[key] += 1
    for cell in cells:
        low = cell.strip().lower()
        key = next((k for label, k in labels if low.startswith(label) and "[e" in low), None)
        if key:
            counts[key] += 1
    return counts


class EvidenceBar(Flowable):
    """A stacked bar: the share of the report's claims in each classification."""

    def __init__(self, counts: dict[str, int], width: float, height: float = 4.5 * mm) -> None:
        super().__init__()
        self.counts, self.width, self.height = counts, width, height

    def wrap(self, *_args):
        return self.width, self.height

    def draw(self) -> None:
        total = sum(self.counts.values()) or 1
        x = 0.0
        for key, n in self.counts.items():
            w = self.width * n / total
            if w <= 0:
                continue
            self.canv.setFillColor(colors.HexColor(CLASS_COLORS[key]))
            self.canv.rect(x, 0, w, self.height, stroke=0, fill=1)
            x += w


# --------------------------------------------------------------------------- document


def _cover(blocks: list[Block], st: dict[str, ParagraphStyle], inline: Inline, t: Translator,
           facts: list[tuple[str, str]] | None = None, counts: dict[str, int] | None = None) -> list[Flowable]:
    title = next((b.text for b in blocks if b.kind == "title"), t("title", name=""))
    kicker, _, company = title.partition(":")
    story: list[Flowable] = [Spacer(1, PAGE_H * 0.14)]
    if company.strip():
        story += [Paragraph(escape(_encodable(kicker.strip().upper())), st["cover_kicker"]), Spacer(1, 4 * mm),
                  Paragraph(escape(_encodable(company.strip())), st["cover_title"])]
    else:
        story.append(Paragraph(escape(_encodable(title)), st["cover_title"]))
    story.append(Spacer(1, PAGE_H * (0.2 if facts else 0.26)))
    for b in blocks:
        if b.kind == "para":
            story.append(Paragraph(inline(b.text), st["cover_meta"] if "\n" in b.text else st["cover_key"]))
            story.append(Spacer(1, 3 * mm))
        elif b.kind == "quote":
            story.append(_note(b.text, st, inline))
    width = BODY_W - 12 * mm
    if facts:
        rows = [[Paragraph(escape(_encodable(k.upper())), st["fact_label"]), Paragraph(escape(_encodable(v)), st["fact"])]
                for k, v in facts]
        panel = Table(rows, colWidths=[width * 0.26, width * 0.74], hAlign="LEFT")
        panel.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story += [Spacer(1, 3 * mm), Paragraph(escape(_encodable(t("key_facts").upper())), st["cover_section"]),
                  panel, Spacer(1, 5 * mm)]
    total = sum((counts or {}).values())
    if total:
        story += [Paragraph(escape(_encodable(t("evidence_profile", n=total).upper())), st["cover_section"]),
                  EvidenceBar(counts, width), Spacer(1, 2 * mm)]
    legend = "&nbsp;&nbsp;&nbsp;&nbsp;".join(
        f'<font color="{CLASS_COLORS[k]}" size="11">&#9632;</font>&nbsp;{_nobreak(t(k))}'
        + (f"&nbsp;<b>{counts[k]}</b>" if total else "")
        for k in CLASS_COLORS)
    story += [Spacer(1, 1 * mm), Paragraph(legend, st["cover_key"])]
    return story


def _note(text: str, st: dict[str, ParagraphStyle], inline: Inline) -> Table:
    box = Table([[Paragraph(inline(text), st["note"])]], colWidths=[BODY_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NOTE_BG), ("LINEBEFORE", (0, 0), (0, -1), 2.5, NOTE_EDGE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return box


def build_story(md: str, lang: str) -> tuple[list[Flowable], str]:
    """The flowables for ``md`` and the running-header text."""
    t = Translator(lang)
    st = _styles()
    blocks = parse_markdown(md)
    first_section = next((i for i, b in enumerate(blocks) if b.kind == "h1"), len(blocks))
    sources_heading = f"21. {t('s21')}"
    at = next((i for i, b in enumerate(blocks) if b.kind == "h1" and b.text == sources_heading), None)
    targets = {r[0] for b in blocks[at:] if b.kind == "table" for r in b.rows[1:]
               if r and re.fullmatch(r"E\d{3,}", r[0])} if at is not None else set()
    inline = Inline(t, targets)

    story: list[Flowable] = _cover(blocks[:first_section], st, inline, t,
                                   facts=_key_facts(blocks, t), counts=_evidence_counts(blocks[first_section:], inline))
    story += [NextPageTemplate("body"), PageBreak()]
    toc = TableOfContents(levelStyles=[st["toc1"]], dotsMinLevel=0)
    story += [Paragraph(escape(_encodable(t("contents"))), st["toc_title"]), toc, PageBreak()]

    in_sources = False
    pending_label: Flowable | None = None
    for n, b in enumerate(blocks[first_section:]):
        if b.kind == "h1":
            in_sources = b.text == sources_heading
            story += [CondPageBreak(40 * mm), SectionHeading(inline(b.text, links=False), st["h1"], f"s{n}")]
            continue
        if b.kind == "h2":
            story += [CondPageBreak(25 * mm), Paragraph(inline(b.text, links=False), st["h2"])]
            continue
        if b.kind == "label":
            pending_label = Paragraph(inline(b.text, links=False), st["label"])
            continue
        if b.kind == "para":
            flow: Flowable = Paragraph(inline(b.text), st["body"])
        elif b.kind == "bullets":
            flow = ListFlowable(
                [ListItem(Paragraph(inline(x), st["bullet"]), leftIndent=11, value="circle") for x in b.items],
                bulletType="bullet", start="•", bulletFontSize=7, bulletColor=ACCENT, leftIndent=11,
                bulletOffsetY=-1)
        elif b.kind == "quote":
            flow = _note(b.text, st, inline)
        elif b.kind == "table" and b.rows and not in_sources and max(len(r) for r in b.rows) >= 7:
            if pending_label is not None:
                story.append(pending_label)
                pending_label = None
            story += _cards(b, st, inline, t)
            continue
        elif b.kind == "table" and b.rows:
            if in_sources:
                b, caption = _lean_sources(b, t)
                if caption:
                    story.append(Paragraph(escape(_encodable(caption)), st["caption"]))
            flow = _table(b, st, inline, anchors=in_sources)
        else:
            continue
        if pending_label is not None:  # keep a label with the first thing under it
            story.append(KeepTogether([pending_label, flow]) if b.kind != "table" else pending_label)
            if b.kind == "table":
                story.append(flow)
            pending_label = None
        else:
            story.append(flow)
        story.append(Spacer(1, 2))
    if pending_label is not None:
        story.append(pending_label)

    title = next((b.text for b in blocks if b.kind == "title"), "")
    return story, _encodable(title)


def render_pdf(md: str, path: Path, *, lang: str = "en") -> Path:
    """Typeset the report Markdown ``md`` into ``path``."""
    t = Translator(lang)
    story, header = build_story(md, lang)
    doc = _Doc(str(path), pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X, topMargin=MARGIN_TOP,
               bottomMargin=MARGIN_BOTTOM, title=header, author="bi-agent", subject=header)
    body = Frame(MARGIN_X, MARGIN_BOTTOM, BODY_W, PAGE_H - MARGIN_TOP - MARGIN_BOTTOM, id="body",
                 leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    cover = Frame(MARGIN_X + 6 * mm, MARGIN_BOTTOM, BODY_W - 12 * mm, PAGE_H - 2 * MARGIN_BOTTOM, id="cover",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate("cover", [cover], onPage=_cover_background),
                          PageTemplate("body", [body])])
    doc.multiBuild(story, canvasmaker=_numbered_canvas(_encodable(t("page_of")), header))
    return path
