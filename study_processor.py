import os
import re
import uuid
import joblib
import pymupdf as fitz

from difflib import SequenceMatcher


# PATHS

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "revision_model.pkl"
)

GENERATED_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "generated"
)

os.makedirs(
    GENERATED_FOLDER,
    exist_ok=True
)


# MODEL

revision_model = None


def load_revision_model():

    global revision_model

    if revision_model is not None:
        return revision_model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "revision_model.pkl was not found. "
            "Run train_model.py first."
        )

    revision_model = joblib.load(
        MODEL_PATH
    )

    return revision_model


# GENERAL TEXT CLEANING

def clean_text(text):

    if not text:
        return ""

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove date/time headers
    text = re.sub(
        r"\b\d{1,2}/\d{1,2}/\d{4},?\s*"
        r"\d{1,2}:\d{2}\b",
        " ",
        text
    )

    # Remove page indicators like 1/3
    text = re.sub(
        r"\b\d+\s*/\s*\d+\b",
        " ",
        text
    )

    # Remove ebook header
    text = re.sub(
        r"\bMicroelectronic Circuits\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Rejoin words split across lines
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# NORMALISE PDF BLOCK TEXT

def normalise_block_text(text):

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# REAL FIGURE CAPTION DETECTION

def detect_real_figure_caption(text):
    """
    ACCEPTS:

        Figure 5.12 The relative levels...
        Fig. 5.12 The relative levels...
        Figure 5.13 The iD-vDS characteristics...

    REJECTS:

        Refer to Fig. 5.12...
        As Figure 5.12 shows...
        Figure 5.13 shows a set of...
        Figure 5.11(a) shows the circuit symbol...
    """

    text = normalise_block_text(text)

    if not text:
        return None

    # Caption must START with Figure or Fig.
    match = re.match(
        r"^(?:Figure|Fig\.?)\s*"
        r"(\d+(?:\.\d+)*(?:[A-Za-z])?)"
        r"\s*(.*)$",
        text,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    figure_number = (
        match.group(1).strip()
    )

    remainder = (
        match.group(2).strip()
    )

    # Remove subfigure marker temporarily.
    #
    # Example:
    #
    # Figure 5.11(a) shows...
    #
    # becomes:
    #
    # shows...
    #
    # allowing us to identify it as BODY TEXT.

    prose_test = re.sub(
        r"^\s*\([a-zA-Z0-9]+\)\s*",
        "",
        remainder
    ).strip().lower()

    prose_starts = [
        "shows ",
        "show ",
        "illustrates ",
        "illustrates how ",
        "demonstrates ",
        "demonstrates how ",
        "indicates ",
        "indicates that ",
        "indicates how ",
        "depicts ",
        "presents ",
        "gives ",
        "is shown ",
        "can be seen "
    ]

    for phrase in prose_starts:

        if prose_test.startswith(
            phrase
        ):
            return None

    # Avoid huge body-text blocks
    if len(remainder) > 500:
        return None

    if len(remainder) < 3:
        return None

    return {
        "number": figure_number,
        "caption": remainder,
        "full_caption": (
            f"Figure {figure_number}"
            f" — {remainder}"
        )
    }


# GET TEXT BLOCKS WITH PDF COORDINATES

def get_page_blocks(page):

    raw_blocks = page.get_text(
        "blocks"
    )

    blocks = []

    for raw_block in raw_blocks:

        x0 = raw_block[0]
        y0 = raw_block[1]
        x1 = raw_block[2]
        y1 = raw_block[3]

        text = normalise_block_text(
            raw_block[4]
        )

        if not text:
            continue

        blocks.append({
            "rect": fitz.Rect(
                x0,
                y0,
                x1,
                y1
            ),
            "text": text
        })

    blocks.sort(
        key=lambda block: (
            block["rect"].y0,
            block["rect"].x0
        )
    )

    return blocks


# FIND REAL FIGURE CAPTIONS

def find_real_figure_captions(page):

    blocks = get_page_blocks(
        page
    )

    captions = []

    for index, block in enumerate(
        blocks
    ):

        detected = (
            detect_real_figure_caption(
                block["text"]
            )
        )

        if not detected:
            continue

        captions.append({
            "number":
                detected["number"],

            "caption":
                detected["caption"],

            "full_caption":
                detected["full_caption"],

            "rect":
                block["rect"],

            "block_index":
                index
        })

    return captions, blocks


# BODY TEXT DETECTION

def is_body_text_block(text):

    text = normalise_block_text(
        text
    )

    if not text:
        return False

    # Genuine figure caption isn't body text
    if detect_real_figure_caption(
        text
    ):
        return False

    # Section heading
    if re.match(
        r"^\d+(?:\.\d+)+\s+",
        text
    ):
        return False

    lower = text.lower()

    # Page headers
    if lower == "microelectronic circuits":
        return False

    if "http://" in lower:
        return False

    if "https://" in lower:
        return False

    word_count = len(
        text.split()
    )

    # Normal textbook paragraph
    if word_count >= 12:
        return True

    if (
        word_count >= 7
        and text.endswith(
            (
                ".",
                "?",
                "!"
            )
        )
    ):
        return True

    return False


# GET ALL GRAPHICAL OBJECTS ABOVE CAPTION

def get_visual_objects_above_caption(
    page,
    caption_rect
):
    """
    Find graphical objects above the genuine caption.

    Includes:
    - vector lines
    - curves
    - circuit symbols
    - graph axes
    - arrows
    - embedded images
    """

    objects = []

    # VECTOR GRAPHICS

    try:

        drawings = page.get_drawings()

        for drawing in drawings:

            rect = drawing.get(
                "rect"
            )

            if rect is None:
                continue

            # Must be above caption
            if rect.y1 >= caption_rect.y0:
                continue

            # Ignore microscopic marks
            if (
                rect.width < 2
                and
                rect.height < 2
            ):
                continue

            objects.append(
                fitz.Rect(rect)
            )

    except Exception as error:

        print(
            "Drawing detection error:",
            error
        )

    # EMBEDDED IMAGES

    try:

        page_dict = page.get_text(
            "dict"
        )

        for block in page_dict[
            "blocks"
        ]:

            if block.get(
                "type"
            ) != 1:
                continue

            bbox = block.get(
                "bbox"
            )

            if not bbox:
                continue

            rect = fitz.Rect(
                bbox
            )

            if rect.y1 >= caption_rect.y0:
                continue

            objects.append(
                rect
            )

    except Exception as error:

        print(
            "Image detection error:",
            error
        )

    return objects


# GROUP GRAPHICAL OBJECTS

def group_visual_objects(objects):
    """
    A PDF diagram may actually consist of hundreds of
    independent lines/arrows/curves.

    Group nearby objects so that a whole circuit diagram or
    graph becomes one candidate figure.
    """

    if not objects:
        return []

    objects = sorted(
        objects,
        key=lambda rect: (
            rect.y0,
            rect.x0
        )
    )

    groups = []

    for rect in objects:

        added = False

        for index, group in enumerate(
            groups
        ):

            expanded_group = fitz.Rect(
                group.x0 - 35,
                group.y0 - 35,
                group.x1 + 35,
                group.y1 + 35
            )

            if expanded_group.intersects(
                rect
            ):

                groups[index] = (
                    group | rect
                )

                added = True

                break

        if not added:

            groups.append(
                fitz.Rect(rect)
            )

    # Repeat merging because joining one group can cause
    # another previously separate group to become connected.

    changed = True

    while changed:

        changed = False

        new_groups = []

        while groups:

            current = groups.pop(0)

            remaining = []

            for other in groups:

                expanded = fitz.Rect(
                    current.x0 - 45,
                    current.y0 - 45,
                    current.x1 + 45,
                    current.y1 + 45
                )

                if expanded.intersects(
                    other
                ):

                    current = (
                        current | other
                    )

                    changed = True

                else:

                    remaining.append(
                        other
                    )

            new_groups.append(
                current
            )

            groups = remaining

        groups = new_groups

    return groups


# SCORE VISUAL GROUP

def score_visual_group(
    group,
    caption_rect,
    page_rect
):

    if group.y1 > caption_rect.y0:
        return -999999

    distance = (
        caption_rect.y0
        - group.y1
    )

    width_ratio = (
        group.width
        / page_rect.width
    )

    height_ratio = (
        group.height
        / page_rect.height
    )

    area_ratio = (
        group.get_area()
        / page_rect.get_area()
    )

    # Ignore tiny decorations
    if (
        width_ratio < 0.08
        and
        height_ratio < 0.05
    ):
        return -999999

    score = 0

    # Prefer large graphical areas
    score += (
        area_ratio * 500
    )

    # Prefer diagram close to caption
    score -= (
        distance * 0.15
    )

    # Prefer reasonably wide diagrams
    score += (
        width_ratio * 50
    )

    return score


# FIND COMPLETE VISUAL

def find_complete_visual(
    page,
    caption
):
    """
    Find the WHOLE diagram associated with a caption.

    For a multi-part figure such as:

        (a)       (b)       (c)

    nearby graphical groups are combined so all three parts
    are included.
    """

    caption_rect = caption[
        "rect"
    ]

    page_rect = page.rect

    objects = (
        get_visual_objects_above_caption(
            page,
            caption_rect
        )
    )

    if not objects:
        return None

    groups = group_visual_objects(
        objects
    )

    if not groups:
        return None

    possible_groups = []

    for group in groups:

        distance = (
            caption_rect.y0
            - group.y1
        )

        if distance < 0:
            continue

        # Ignore graphics far above caption
        if distance > 280:
            continue

        score = score_visual_group(
            group,
            caption_rect,
            page_rect
        )

        if score > -999999:

            possible_groups.append(
                (
                    score,
                    group
                )
            )

    if not possible_groups:
        return None

    possible_groups.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best = fitz.Rect(
        possible_groups[0][1]
    )

    # Expand to nearby graphical groups belonging to the
    # SAME figure.
    #
    # This is particularly important for figures containing
    # several circuit symbols.

    changed = True

    while changed:

        changed = False

        for group in groups:

            # Must be above caption
            if group.y1 > caption_rect.y0:
                continue

            # Horizontal separation
            if group.x1 < best.x0:

                horizontal_gap = (
                    best.x0
                    - group.x1
                )

            elif best.x1 < group.x0:

                horizontal_gap = (
                    group.x0
                    - best.x1
                )

            else:

                horizontal_gap = 0

            # Vertical separation
            if group.y1 < best.y0:

                vertical_gap = (
                    best.y0
                    - group.y1
                )

            elif best.y1 < group.y0:

                vertical_gap = (
                    group.y0
                    - best.y1
                )

            else:

                vertical_gap = 0

            # Join nearby pieces.
            #
            # Larger horizontal allowance is deliberate:
            # multi-part textbook diagrams may have spaces
            # between (a), (b), and (c).
            if (
                horizontal_gap <= 110
                and
                vertical_gap <= 90
            ):

                old_best = fitz.Rect(
                    best
                )

                best = (
                    best | group
                )

                if (
                    best.x0 != old_best.x0
                    or
                    best.y0 != old_best.y0
                    or
                    best.x1 != old_best.x1
                    or
                    best.y1 != old_best.y1
                ):
                    changed = True

    return best


# INCLUDE TEXT LABELS INSIDE DIAGRAM

def expand_for_diagram_labels(
    page,
    visual_rect,
    caption_rect
):
    """
    Important:

    Diagram labels such as

        G
        D
        S
        vGS
        vDS
        Saturation
        Triode
        (a)
        (b)
        (c)

    may be stored as PDF TEXT rather than vector graphics.

    Include nearby short text blocks without including
    ordinary textbook paragraphs.
    """

    blocks = get_page_blocks(
        page
    )

    expanded = fitz.Rect(
        visual_rect
    )

    changed = True

    while changed:

        changed = False

        nearby_area = fitz.Rect(
            expanded.x0 - 50,
            expanded.y0 - 40,
            expanded.x1 + 50,
            expanded.y1 + 40
        )

        for block in blocks:

            rect = block[
                "rect"
            ]

            # Caption must remain outside image
            if rect.y0 >= caption_rect.y0:
                continue

            if not nearby_area.intersects(
                rect
            ):
                continue

            text = block[
                "text"
            ]

            # Do NOT absorb paragraphs
            if is_body_text_block(
                text
            ):
                continue

            old = fitz.Rect(
                expanded
            )

            expanded = (
                expanded | rect
            )

            if (
                expanded.x0 != old.x0
                or
                expanded.y0 != old.y0
                or
                expanded.x1 != old.x1
                or
                expanded.y1 != old.y1
            ):
                changed = True

    return expanded


# FALLBACK FIGURE REGION

def fallback_figure_region(
    page,
    caption,
    blocks
):
    """
    Used if PyMuPDF can't identify the diagram's graphics.

    Capture the region between the preceding paragraph
    and the genuine caption.
    """

    caption_rect = caption[
        "rect"
    ]

    page_rect = page.rect

    body_blocks = []

    for block in blocks:

        rect = block[
            "rect"
        ]

        if rect.y1 >= caption_rect.y0:
            continue

        if is_body_text_block(
            block["text"]
        ):

            body_blocks.append(
                block
            )

    if body_blocks:

        nearest_body = max(
            body_blocks,
            key=lambda item:
                item["rect"].y1
        )

        top = (
            nearest_body[
                "rect"
            ].y1
            + 8
        )

    else:

        top = max(
            page_rect.y0 + 20,
            caption_rect.y0 - 350
        )

    bottom = (
        caption_rect.y0 - 5
    )

    return fitz.Rect(
        page_rect.x0 + 25,
        top,
        page_rect.x1 - 25,
        bottom
    )


# GET WHOLE FIGURE REGION

def get_actual_figure_region(
    page,
    caption,
    blocks
):
    """
    Return the WHOLE diagram.

    This deliberately does NOT crop to one individual
    curve, symbol, arrow or circuit element.
    """

    caption_rect = caption[
        "rect"
    ]

    page_rect = page.rect

    visual = find_complete_visual(
        page,
        caption
    )

    # No graphics detected -> use fallback.

    if visual is None:

        return fallback_figure_region(
            page,
            caption,
            blocks
        )

    # Add labels belonging to diagram.

    visual = (
        expand_for_diagram_labels(
            page,
            visual,
            caption_rect
        )
    )

    # Add breathing room around complete diagram.

    horizontal_padding = 22
    vertical_padding = 18

    left = max(
        page_rect.x0,
        visual.x0
        - horizontal_padding
    )

    right = min(
        page_rect.x1,
        visual.x1
        + horizontal_padding
    )

    top = max(
        page_rect.y0,
        visual.y0
        - vertical_padding
    )

    bottom = min(
        caption_rect.y0 - 4,
        visual.y1
        + vertical_padding
    )

    # Don't allow a weird narrow crop.

    minimum_width = (
        page_rect.width * 0.45
    )

    current_width = (
        right - left
    )

    if current_width < minimum_width:

        centre = (
            left + right
        ) / 2

        left = max(
            page_rect.x0,
            centre
            - minimum_width / 2
        )

        right = min(
            page_rect.x1,
            centre
            + minimum_width / 2
        )

    return fitz.Rect(
        left,
        top,
        right,
        bottom
    )


# SAVE FIGURE IMAGE

def save_figure_image(
    page,
    figure_number,
    region
):

    safe_number = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        figure_number
    )

    unique_id = (
        uuid.uuid4().hex[:8]
    )

    filename = (
        f"figure_"
        f"{safe_number}_"
        f"{unique_id}.png"
    )

    filepath = os.path.join(
        GENERATED_FOLDER,
        filename
    )

    # Higher resolution for equations,
    # graph labels and circuit symbols.
    matrix = fitz.Matrix(
        2.5,
        2.5
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        clip=region,
        alpha=False
    )

    pixmap.save(
        filepath
    )

    return (
        "generated/"
        + filename
    )


# EXTRACT FIGURES FROM PAGE

def extract_figures_from_page(
    page,
    page_number
):

    captions, blocks = (
        find_real_figure_captions(
            page
        )
    )

    figures = []

    seen_numbers = set()

    for caption in captions:

        figure_number = (
            caption["number"]
        )

        # Avoid duplicates
        if figure_number in seen_numbers:
            continue

        region = (
            get_actual_figure_region(
                page,
                caption,
                blocks
            )
        )

        try:

            image_path = (
                save_figure_image(
                    page,
                    figure_number,
                    region
                )
            )

        except Exception as error:

            print(
                f"Could not extract "
                f"Figure {figure_number}: "
                f"{error}"
            )

            continue

        figures.append({
            "number":
                figure_number,

            "caption":
                caption["caption"],

            "full_caption":
                caption[
                    "full_caption"
                ],

            "page":
                page_number,

            "image":
                image_path
        })

        seen_numbers.add(
            figure_number
        )

    return figures


# MATH EXTRACTION: character geometry + LaTeX + image fallback

MATH_RELATIONS = set('=<>≤≥≈≠∝')
MATH_OPERATORS = set('+−-×÷∑∫√±')
MATH_GREEK = {
    'π': r'\pi', 'θ': r'\theta', 'λ': r'\lambda',
    'μ': r'\mu', 'Δ': r'\Delta', 'δ': r'\delta',
    'Ω': r'\Omega', 'ω': r'\omega', 'σ': r'\sigma',
    'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma',
}
MATH_UNICODE = {
    '−': '-', '×': r'\times ', '÷': r'\div ',
    '≤': r'\leq ', '≥': r'\geq ', '≈': r'\approx ',
    '≠': r'\neq ', '∝': r'\propto ', '±': r'\pm ',
    '∞': r'\infty ', '∑': r'\sum ', '∫': r'\int ',
    '²': '^{2}', '³': '^{3}', '⁴': '^{4}',
    '′': r'\prime ',
}


def math_text_to_latex(value):
    """Convert supported Unicode maths; never guess an absent operator."""
    for char, command in {**MATH_UNICODE, **MATH_GREEK}.items():
        value = value.replace(char, command)
    return re.sub(r'\s+', ' ', value).strip()


def raw_math_line(line):
    """Use character positions to identify small raised/lowered runs."""
    chars = []
    for span in line.get('spans', []):
        for ch in span.get('chars', []):
            c = ch.get('c', '')
            if c:
                chars.append((c, fitz.Rect(ch['bbox']), float(span.get('size', 0))))
    if not chars:
        return ''
    chars.sort(key=lambda item: item[1].x0)
    # Font sizes, unlike glyph box heights, are relatively stable.
    sizes = sorted(size for c, r, size in chars if c.strip() and size > 0)
    if not sizes:
        return ''
    base_size = sizes[len(sizes) // 2]
    regular = [r for c, r, size in chars if c.strip() and size >= base_size * .88]
    baseline = sorted(r.y1 for r in regular)[len(regular)//2] if regular else 0
    result, script, buffer, previous = [], None, [], None

    def flush():
        nonlocal script, buffer
        if buffer:
            content = ''.join(buffer)
            result.append(('_{' if script == 'sub' else '^{') + content + '}' if script else content)
        script, buffer = None, []

    for c, rect, size in chars:
        gap = rect.x0 - previous.x1 if previous else 0
        if c.isspace() or gap > base_size * .55:
            flush()
            if result and not result[-1].endswith(' '):
                result.append(' ')
        if c.isspace():
            previous = rect
            continue
        small = size < base_size * .86
        mode = None
        if small and rect.y1 < baseline - base_size * .12:
            mode = 'super'
        elif small and rect.y1 > baseline + base_size * .08:
            mode = 'sub'
        if mode != script:
            flush()
        if mode:
            script = mode
            buffer.append(c)
        else:
            result.append(c)
        previous = rect
    flush()
    return re.sub(r'\s+', ' ', ''.join(result)).strip()


def equation_line_candidate(value):
    """Reject prose; require a short mathematical relation/expression."""
    if not value or len(value) > 145:
        return False
    if re.match(r'^(?:figure|fig\.?|table)\s*\d', value, re.I) or re.match(r'^\d+(?:\.\d+)+\s+', value):
        return False
    words = re.findall(r'[A-Za-z]{3,}', re.sub(r'[_^]\{[^}]*\}', '', value))
    if len(words) > 5:
        return False
    relations = sum(c in MATH_RELATIONS for c in value)
    operators = sum(c in MATH_OPERATORS for c in value)
    scripts = len(re.findall(r'[_^]\{', value))
    return bool(relations or (operators and scripts) or (scripts >= 2 and len(words) <= 2))


def save_equation_crop(page, page_number, rect):
    """Original source pixels are used when a faithful LaTeX parse is unavailable."""
    rect = fitz.Rect(rect) & page.rect
    if rect.is_empty or rect.width < 8 or rect.height < 4:
        return None
    filename = f'equation_p{page_number}_{uuid.uuid4().hex[:10]}.png'
    page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False).save(
        os.path.join(GENERATED_FOLDER, filename)
    )
    return 'generated/' + filename


def extract_equations_from_page(page, page_number):
    """Text equations, plus image fallback for formula-like non-text bands.

    A PDF may draw an equation as vector paths or an embedded image.
    Those cannot be truthfully converted to LaTeX from characters alone.
    """
    raw = page.get_text('rawdict')
    equations, seen_rects = [], []
    for block in raw.get('blocks', []):
        if block.get('type') != 0:
            continue
        for line in block.get('lines', []):
            value = raw_math_line(line)
            if not equation_line_candidate(value):
                continue
            rect = fitz.Rect(line['bbox'])
            latex = math_text_to_latex(value)
            # A long expression containing ambiguous PDF glyphs is safer
            # as a screenshot than as an incorrectly inferred formula.
            suspicious = any(c in latex for c in ('�', '□', '\ufffd'))
            image = save_equation_crop(page, page_number,
                                       fitz.Rect(rect.x0-6, rect.y0-5, rect.x1+6, rect.y1+5))
            equations.append({
                'equation': '' if suspicious else latex,
                'raw': value, 'page': page_number,
                'rect': tuple(rect), 'image': image,
                'render_as': 'image' if suspicious else 'latex',
            })
            seen_rects.append(rect)

    # A conservative fallback for unextractable displayed equations:
    # locate graphic-only bands immediately after prose that introduces
    # an equation. Do not classify arbitrary figures as equations.
    blocks = get_page_blocks(page)
    drawings = [fitz.Rect(d['rect']) for d in page.get_drawings()
                if d.get('rect') is not None]
    for i, block in enumerate(blocks):
        prose = block['text'].lower()
        if not re.search(r'(?:given by|determined by|expressed as|equal to|namely[, :]?\s*$|following equation|written as)', prose):
            continue
        if len(prose.split()) < 7:
            continue
        top = block['rect'].y1 + 2
        below = [b for b in blocks if b['rect'].y0 > top + 4 and b['rect'].y0 < top + 105]
        bottom = min((b['rect'].y0 for b in below), default=min(top + 90, page.rect.y1 - 15))
        bottom = min(bottom, top + 90)
        if bottom - top < 10:
            continue
        # Restrict to a plausible narrow equation band, not a diagram.
        region = fitz.Rect(page.rect.x0+32, top, page.rect.x1-32, bottom)
        relevant = [d for d in drawings if d.intersects(region) and d.width > 1 and d.height > 0]
        image_blocks = [fitz.Rect(b['bbox']) for b in raw.get('blocks', [])
                        if b.get('type') == 1 and fitz.Rect(b['bbox']).intersects(region)]
        objects = relevant + image_blocks
        if not objects:
            continue
        united = fitz.Rect(objects[0])
        for rect in objects[1:]:
            united |= rect
        if united.height > 55 or united.width > page.rect.width * .85:
            continue
        if any(united.intersects(existing) for existing in seen_rects):
            continue
        clip = fitz.Rect(max(page.rect.x0, united.x0-12), max(top, united.y0-8),
                         min(page.rect.x1, united.x1+12), min(bottom, united.y1+8))
        image = save_equation_crop(page, page_number, clip)
        if image:
            equations.append({
                'equation': '', 'raw': '[non-text equation]',
                'page': page_number, 'rect': tuple(clip),
                'image': image, 'render_as': 'image',
            })
            seen_rects.append(clip)
    equations.sort(key=lambda item: (item['rect'][1], item['rect'][0]))
    return equations


def normalise_equation_for_comparison(equation):
    return re.sub(r'\s+', '', equation).lower()


def remove_duplicate_equations(equations):
    result, seen = [], set()
    for eq in equations:
        # Do not collapse distinct image-only equations with empty LaTeX.
        key = normalise_equation_for_comparison(eq['equation']) or (eq['page'], eq['rect'])
        if key not in seen:
            seen.add(key)
            result.append(eq)
    return result


def get_equation_context(page, equation):
    rect = fitz.Rect(equation['rect'])
    preceding = [(rect.y0-b['rect'].y1, b['text']) for b in get_page_blocks(page)
                 if b['rect'].y1 <= rect.y0 and rect.y0-b['rect'].y1 < 150
                 and is_body_text_block(b['text'])]
    if not preceding:
        return ''
    preceding.sort()
    sentences = split_into_sentences(preceding[0][1])
    return sentences[-1] if sentences else ''


def get_equation_section(page, equation):
    rect = fitz.Rect(equation['rect'])
    headings = [(b['rect'].y1, b['text']) for b in get_page_blocks(page)
                if b['rect'].y1 <= rect.y0 and is_heading(b['text'])]
    return clean_heading(max(headings)[1]) if headings else None


def select_relevant_equations(equations, selected):
    pages = {item['page'] for item in selected}
    return remove_duplicate_equations([eq for eq in equations if eq['page'] in pages])


def extract_pdf(pdf_bytes):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    figures = []

    equations = []

    for page_index in range(
        len(document)
    ):

        page = document[
            page_index
        ]

        page_number = (
            page_index + 1
        )

        # NORMAL TEXT

        raw_text = page.get_text(
            "text"
        )

        cleaned_text = clean_text(
            raw_text
        )

        if cleaned_text:

            pages.append({
                "page":
                    page_number,

                "text":
                    cleaned_text
            })

        # FIGURES

        page_figures = (
            extract_figures_from_page(
                page,
                page_number
            )
        )

        figures.extend(
            page_figures
        )

        # EQUATIONS

        page_equations = (
            extract_equations_from_page(
                page,
                page_number
            )
        )

        print(
            f"\nPAGE {page_number} EQUATIONS:"
        )

        for eq in page_equations:
            print(
                "  RAW:",
                eq["raw"]
            )

            print(
                "  LATEX:",
                eq["equation"]
            )

        for equation in page_equations:

            equation["context"] = (
                get_equation_context(
                    page,
                    equation
                )
            )

            equation["section"] = (
                get_equation_section(
                    page,
                    equation
                )
            )

        equations.extend(
            page_equations
        )

    document.close()

    if not pages:

        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be a scanned/image-only PDF."
        )

    return (
        pages,
        figures,
        remove_duplicate_equations(
            equations
        )
    )


# HEADING DETECTION

def is_heading(line):

    line = line.strip()

    if not line:
        return False

    # Examples:
    # 5.2 Current-Voltage Characteristics
    # 5.2.1 Circuit Symbol

    if re.match(
        r"^\d+(?:\.\d+)+\s+[A-Z]",
        line
    ):
        return True

    words = line.split()

    if 1 < len(words) <= 10:

        if line.endswith(
            (
                ".",
                "?",
                "!",
                ";"
            )
        ):
            return False

        capitalised = sum(
            1
            for word in words
            if word[:1].isupper()
        )

        if capitalised >= max(
            2,
            len(words) * 0.6
        ):
            return True

    return False


# SPLIT PAGE INTO SECTIONS

def split_page_into_sections(
    page_text
):

    lines = page_text.splitlines()

    sections = []

    current_heading = None
    current_text = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if is_heading(
            line
        ):

            if current_text:

                sections.append({
                    "heading":
                        current_heading,

                    "text":
                        " ".join(
                            current_text
                        )
                })

                current_text = []

            current_heading = line

        else:

            current_text.append(
                line
            )

    if current_text:

        sections.append({
            "heading":
                current_heading,

            "text":
                " ".join(
                    current_text
                )
        })

    if (
        not sections
        and page_text.strip()
    ):

        sections.append({
            "heading": None,
            "text": page_text.strip()
        })

    return sections


# SENTENCE SPLITTING

def split_into_sentences(text):

    protected = text

    protected = protected.replace(
        "Fig.",
        "Fig<PERIOD>"
    )

    protected = protected.replace(
        "Eq.",
        "Eq<PERIOD>"
    )

    protected = protected.replace(
        "e.g.",
        "eg<PERIOD>"
    )

    protected = protected.replace(
        "i.e.",
        "ie<PERIOD>"
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        protected
    )

    result = []

    for sentence in sentences:

        sentence = sentence.replace(
            "<PERIOD>",
            "."
        )

        sentence = re.sub(
            r"\s+",
            " ",
            sentence
        ).strip()

        if sentence:
            result.append(
                sentence
            )

    return result


# JUNK FILTER

def is_junk_sentence(sentence):

    text = sentence.strip()

    lower = text.lower()

    if len(text) < 25:
        return True

    if len(text) > 650:
        return True

    if "http://" in lower:
        return True

    if "https://" in lower:
        return True

    if "www." in lower:
        return True

    if "oupereader" in lower:
        return True

    # Figure captions aren't revision bullets.
    if re.match(
        r"^(figure|fig\.?)\s+\d+",
        lower
    ):
        return True

    if re.match(
        r"^table\s+\d+",
        lower
    ):
        return True

    filler_phrases = [
        "in this chapter we",
        "in the next chapter",
        "will be discussed in chapter",
        "will be discussed later",
        "as will be seen later",
        "we will see later",
        "we shall see later",
        "as discussed previously",
        "as mentioned previously",
        "shown in the accompanying diagram",
        "the following figure",
        "the following table",
        "you should memorize",
        "for a graphical reminder",
        "the next section",
        "this section we present"
    ]

    if any(
        phrase in lower
        for phrase in filler_phrases
    ):
        return True

    letters = sum(
        character.isalpha()
        for character in text
    )

    if letters < 15:
        return True

    return False


# REVISION BONUS

def revision_bonus(sentence):

    lower = sentence.lower()

    score = 0.0

    definition_patterns = [
        " is defined as ",
        " is the ",
        " refers to ",
        " means ",
        " is called ",
        " is known as "
    ]

    if any(
        pattern in f" {lower} "
        for pattern in definition_patterns
    ):
        score += 0.08

    relationship_words = [
        "therefore",
        "because",
        "causes",
        "results in",
        "depends on",
        "determined by",
        "proportional to",
        "inversely proportional",
        "increases",
        "decreases",
        "greater than",
        "less than",
        "independent of",
        "equal to"
    ]

    relationship_matches = sum(
        phrase in lower
        for phrase in relationship_words
    )

    score += min(
        relationship_matches * 0.025,
        0.12
    )

    important_phrases = [
        "cutoff",
        "triode",
        "saturation",
        "threshold voltage",
        "overdrive voltage",
        "drain current",
        "gate voltage",
        "source voltage",
        "current source",
        "voltage-controlled",
        "amplifier",
        "switch",
        "mosfet",
        "transistor"
    ]

    technical_matches = sum(
        phrase in lower
        for phrase in important_phrases
    )

    score += min(
        technical_matches * 0.015,
        0.10
    )

    if re.search(
        r"[=<>≤≥∝]",
        sentence
    ):
        score += 0.06

    return score


# ANALYSE DOCUMENT WITH YOUR TRAINED MODEL

def analyse_document(pages):

    model = load_revision_model()

    results = []

    global_order = 0

    for page in pages:

        sections = (
            split_page_into_sections(
                page["text"]
            )
        )

        for section in sections:

            sentences = (
                split_into_sentences(
                    section["text"]
                )
            )

            usable_sentences = [
                sentence
                for sentence in sentences
                if not is_junk_sentence(
                    sentence
                )
            ]

            if not usable_sentences:
                continue

            probabilities = (
                model.predict_proba(
                    usable_sentences
                )[:, 1]
            )

            for (
                sentence,
                probability
            ) in zip(
                usable_sentences,
                probabilities
            ):

                final_score = (
                    float(
                        probability
                    )
                    + revision_bonus(
                        sentence
                    )
                )

                final_score = max(
                    0.0,
                    min(
                        1.0,
                        final_score
                    )
                )

                results.append({
                    "sentence":
                        sentence,

                    "page":
                        page["page"],

                    "heading":
                        section[
                            "heading"
                        ],

                    "ml_score":
                        float(
                            probability
                        ),

                    "score":
                        final_score,

                    "order":
                        global_order
                })

                global_order += 1

    return results


# DUPLICATE REMOVAL

def normalise_similarity(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def sentence_similarity(
    a,
    b
):

    a = normalise_similarity(
        a
    )

    b = normalise_similarity(
        b
    )

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def remove_redundancy(items):

    ranked = sorted(
        items,
        key=lambda item:
            item["score"],
        reverse=True
    )

    selected = []

    for candidate in ranked:

        duplicate = False

        for existing in selected:

            similarity = (
                sentence_similarity(
                    candidate[
                        "sentence"
                    ],
                    existing[
                        "sentence"
                    ]
                )
            )

            if similarity >= 0.72:

                duplicate = True

                break

        if not duplicate:

            selected.append(
                candidate
            )

    return selected


# SELECT REVISION MATERIAL

def select_revision_content(
    analysed,
    maximum=28
):

    if not analysed:
        return []

    selected = [
        item
        for item in analysed
        if item["score"] >= 0.60
    ]

    minimum = min(
        8,
        len(analysed)
    )

    if len(selected) < minimum:

        selected = sorted(
            analysed,
            key=lambda item:
                item["score"],
            reverse=True
        )[:minimum]

    selected = remove_redundancy(
        selected
    )

    selected = sorted(
        selected,
        key=lambda item:
            item["score"],
        reverse=True
    )[:maximum]

    selected = sorted(
        selected,
        key=lambda item:
            item["order"]
    )

    return selected


# SIMPLIFY SENTENCES

def simplify_sentence(sentence):

    text = sentence.strip()

    lead_ins = [
        r"^notice that\s+",
        r"^note that\s+",
        r"^as table [\d.]+ shows,\s*",
        r"^as figure [\d.]+ shows,\s*",
        r"^as fig\.?\s*[\d.]+ shows,\s*",
        r"^finally,\s*",
        r"^in effect,\s*",
        r"^thus,\s*"
    ]

    for pattern in lead_ins:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(
        r"\(see\s+(?:fig\.|figure|table)"
        r"\s*[\d.]+\)",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if text:

        text = (
            text[0].upper()
            + text[1:]
        )

    return text


# HEADING CLEANUP

def clean_heading(heading):

    if not heading:
        return None

    heading = re.sub(
        r"^\d+(?:\.\d+)+\s*",
        "",
        heading
    )

    heading = heading.strip()

    return (
        heading
        or None
    )


# GROUP NOTES

def group_notes(selected):

    groups = []

    current_key = None
    current_group = None

    for item in selected:

        heading = clean_heading(
            item["heading"]
        )

        key = (
            heading
            or f"Page {item['page']}"
        )

        if key != current_key:

            current_group = {
                "heading":
                    key,

                "items":
                    []
            }

            groups.append(
                current_group
            )

            current_key = key

        current_group[
            "items"
        ].append(
            item
        )

    return groups


# FIGURE RELEVANCE

def figure_is_relevant(
    figure,
    selected
):

    return any(
        item["page"]
        == figure["page"]

        for item in selected
    )


def select_figures(
    figures,
    selected
):

    relevant = []

    seen = set()

    for figure in figures:

        number = figure[
            "number"
        ]

        if number in seen:
            continue

        if figure_is_relevant(
            figure,
            selected
        ):

            relevant.append(
                figure
            )

            seen.add(
                number
            )

    return relevant


# REVISION NOTES

def generate_notes(
    selected,
    figures,
    equations
):

    groups = group_notes(
        selected
    )

    note_sections = []

    for group in groups:

        bullets = []

        pages = set()

        for item in group[
            "items"
        ]:

            sentence = (
                simplify_sentence(
                    item["sentence"]
                )
            )

            if sentence:

                bullets.append(
                    sentence
                )

                pages.add(
                    item["page"]
                )

        if bullets:

            note_sections.append({
                "heading":
                    group["heading"],

                "bullets":
                    bullets,

                "pages":
                    sorted(pages)
            })

    relevant_figures = (
        select_figures(
            figures,
            selected
        )
    )

    relevant_equations = (
        select_relevant_equations(
            equations,
            selected
        )
    )

    return {
        "type":
            "notes",

        "title":
            "Revision Notes",

        "sections":
            note_sections,

        "figures":
            relevant_figures,

        "equations":
            relevant_equations
    }


# DEFINITION EXTRACTION

def extract_definition(sentence):
    """Extract term/meaning pairs from explicit, colon and implicit definitions."""
    text = re.sub(r"\s+", " ", sentence).strip()
    if not text:
        return None

    colon = re.match(r"^([^:]{2,85}):\s*(.{5,})$", text)
    if colon:
        term, meaning = (part.strip() for part in colon.groups())
        if (1 <= len(term.split()) <= 10
                and not re.search(r"[.!?]", term)
                and not re.match(r"^(?:example|examples|e\.g\.|note|figure|fig\.|table)\b", term, re.I)):
            return term, meaning

    patterns = [
        r"^(.{2,80}?)\s+is defined as\s+(.+)$",
        r"^(.{2,80}?)\s+refers to\s+(.+)$",
        r"^(.{2,80}?)\s+is the\s+(.+)$",
        r"^(.{2,80}?)\s+means\s+(.+)$",
        r"^(.{2,80}?)\s+is called\s+(.+)$",
        r"^(.{2,80}?)\s+is known as\s+(.+)$",
        r"^(.{2,80}?)\s+can be described as\s+(.+)$",
        r"^(.{2,80}?)\s+is a type of\s+(.+)$",
        r"^(.{2,80}?)\s+is an?\s+(.+)$",
        r"^(.{2,80}?)\s+are\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, re.I)
        if match:
            term, meaning = (part.strip(" :,-") for part in match.groups())
            if (1 <= len(term.split()) <= 8 and len(meaning) >= 10
                    and not re.search(r"[.!?]", term)
                    and not re.match(r"^(?:there|this|that|it|these|those|we|they)\b", term, re.I)):
                return term, meaning

    # Examples: "A capacitor stores energy in an electric field."
    match = re.match(
        r"^((?:An?|The)\s+[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,4})\s+"
        r"(stores|measures|converts|represents|describes|consists of|contains|"
        r"provides|allows|controls|opposes|detects|indicates)\s+(.{10,})$",
        text, re.I
    )
    if match:
        term, verb, meaning = match.groups()
        return term, f"{verb} {meaning}"
    return None


# CONCEPT QUESTIONS

def make_concept_question(sentence):

    lower = sentence.lower()

    if " states that " in lower:

        parts = re.split(
            r"\s+states that\s+",
            sentence,
            maxsplit=1,
            flags=re.IGNORECASE
        )

        if len(parts) == 2:

            return (
                f"What does "
                f"{parts[0].strip()} state?",
                parts[1].strip()
            )

    if " causes " in lower:

        parts = re.split(
            r"\s+causes\s+",
            sentence,
            maxsplit=1,
            flags=re.IGNORECASE
        )

        if len(parts) == 2:

            return (
                f"What does "
                f"{parts[0].strip()} cause?",
                parts[1].strip()
            )

    if " results in " in lower:

        parts = re.split(
            r"\s+results in\s+",
            sentence,
            maxsplit=1,
            flags=re.IGNORECASE
        )

        if len(parts) == 2:

            return (
                f"What does "
                f"{parts[0].strip()} "
                f"result in?",
                parts[1].strip()
            )

    if " determined by " in lower:

        parts = re.split(
            r"\s+determined by\s+",
            sentence,
            maxsplit=1,
            flags=re.IGNORECASE
        )

        if len(parts) == 2:

            return (
                f"What determines "
                f"{parts[0].strip()}?",
                parts[1].strip()
            )

    if (
        "amplifier" in lower
        and
        "saturation" in lower
    ):

        return (
            "Which MOSFET region is "
            "normally used for amplification?",
            sentence
        )

    if (
        "switch" in lower
        and
        (
            "triode" in lower
            or
            "cutoff" in lower
        )
    ):

        return (
            "Which MOSFET regions are "
            "useful for switching?",
            sentence
        )

    return None


# FLASHCARDS

def generate_flashcards(selected):
    """Create term-front/definition-back cards, preserving source examples."""
    cards = []
    used = set()
    ranked = sorted(selected, key=lambda item: item["score"], reverse=True)

    for item in ranked:
        sentence = simplify_sentence(item["sentence"])
        definition = extract_definition(sentence)

        if definition:
            term, answer = definition
            front = re.sub(r"^(?:a|an|the)\s+", "", term, flags=re.I).strip(" .:;?!")
        else:
            concept = make_concept_question(sentence)
            if concept:
                front, answer = concept
            else:
                # Use the source heading rather than inventing a definition.
                front = clean_heading(item.get("heading"))
                if not front or len(sentence.split()) < 7:
                    continue
                answer = sentence

        if not front or not answer:
            continue
        key = (front.casefold(), answer.casefold())
        if key in used:
            continue
        used.add(key)
        cards.append({
            "question": front,
            "answer": answer,
            "source_page": item["page"]
        })
        if len(cards) >= 15:
            break

    return {"type": "flashcards", "title": "Flashcards", "flashcards": cards}


# QUIZ

def generate_quiz(selected):

    questions = []

    used = set()

    ranked = sorted(
        selected,
        key=lambda item:
            item["score"],
        reverse=True
    )

    for item in ranked:

        sentence = (
            simplify_sentence(
                item["sentence"]
            )
        )

        definition = (
            extract_definition(
                sentence
            )
        )

        if definition:

            term, answer = (
                definition
            )

            question = (
                f"Define {term}."
            )

        else:

            concept = (
                make_concept_question(
                    sentence
                )
            )

            if not concept:
                # Retain informative sentences as answers instead of
                # silently discarding all material that lacks a pattern.
                if len(sentence.split()) < 7:
                    continue
                heading = clean_heading(item.get("heading"))
                if not heading:
                    continue
                question = f"What is an important point about {heading}?"
                answer = sentence
            else:
                question, answer = concept

        key = question.lower()

        if key in used:
            continue

        used.add(
            key
        )

        questions.append({
            "question":
                question,

            "answer":
                answer,

            "source_page":
                item["page"]
        })

        if len(
            questions
        ) >= 12:
            break

    return {
        "type":
            "quiz",

        "title":
            "Practice Quiz",

        "questions":
            questions
    }


# MAIN FUNCTION CALLED BY APP.PY

def generate_study_material(
    pdf_bytes,
    mode
):

    if mode not in (
        "notes",
        "flashcards",
        "quiz"
    ):

        raise ValueError(
            "Mode must be notes, "
            "flashcards or quiz."
        )

    # 1. Extract PDF text + figures + equations

    pages, figures, equations = extract_pdf(
        pdf_bytes
    )

    # 2. Score content using YOUR trained model

    analysed = analyse_document(
        pages
    )

    # 3. Select useful revision content

    selected = (
        select_revision_content(
            analysed
        )
    )

    # 4. Generate requested output

    if mode == "notes":

        return generate_notes(
            selected,
            figures,
            equations
        )

    if mode == "flashcards":

        return generate_flashcards(
            selected
        )

    return generate_quiz(
        selected
    )

