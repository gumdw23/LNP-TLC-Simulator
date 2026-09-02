"""Shared visual design system, ported from the LNP Lab Platform's main
Portal (apps/lab_portal/lib/theme.py in that repository) so this app and
the Portal read as one product family rather than three unrelated tools.

Token VALUES are copied verbatim, not reinterpreted -- if the Portal's own
values ever change, this file should be updated to match rather than
drifting its own direction. See that repository's docs/25_DESIGN_SYSTEM.md
for the full reasoning (reference services, why these specific tokens, why
this is not a copy of Toss/Linear/Stripe).
"""

import streamlit as st

ACCENT = "#3B5BDB"
ACCENT_HOVER = "#3151C4"
ACCENT_SOFT = "#EEF1FE"
ACCENT_SOFT_LINE = "#D7DEFB"
TEAL = "#12A594"
TEAL_SOFT = "#E4F7F4"
AMBER = "#C77D14"
AMBER_SOFT = "#FBF0DD"
DANGER = "#D8434E"
DANGER_SOFT = "#FBEAEB"

INK = "#1A1D29"
INK_2 = "#4A4F63"
INK_3 = "#7A8094"
LINE = "#E4E6ED"
LINE_SOFT = "#EEF0F4"
SURFACE = "#FFFFFF"
CANVAS = "#F6F7FB"

FONT_DISPLAY = "'Manrope', 'Pretendard', system-ui, sans-serif"
FONT_BODY = "'Inter', 'Pretendard', system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', 'D2Coding', monospace"

_GOOGLE_FONTS_HREF = (
    "https://fonts.googleapis.com/css2?"
    "family=Manrope:wght@500;700;800&"
    "family=Inter:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@500;600&"
    "display=swap"
)

_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_GOOGLE_FONTS_HREF}" rel="stylesheet">
<style>
:root {{
    --accent: {ACCENT};
    --accent-hover: {ACCENT_HOVER};
    --accent-soft: {ACCENT_SOFT};
    --accent-soft-line: {ACCENT_SOFT_LINE};
    --teal: {TEAL};
    --teal-soft: {TEAL_SOFT};
    --amber: {AMBER};
    --amber-soft: {AMBER_SOFT};
    --danger: {DANGER};
    --danger-soft: {DANGER_SOFT};
    --ink: {INK};
    --ink-2: {INK_2};
    --ink-3: {INK_3};
    --line: {LINE};
    --line-soft: {LINE_SOFT};
    --surface: {SURFACE};
    --canvas: {CANVAS};
}}
html, body, [class*="css"] {{ font-family: {FONT_BODY}; }}
h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {{
    font-family: {FONT_DISPLAY} !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
    color: var(--ink);
}}
[data-testid="stAppViewContainer"] {{ background: var(--canvas); }}
[data-testid="stHeader"] {{ background: transparent; }}
.stButton > button, .stFormSubmitButton > button, .stLinkButton > a {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: {FONT_BODY} !important;
    border: 1.5px solid var(--line) !important;
    color: var(--ink-2) !important;
    background: var(--surface) !important;
    transition: border-color .15s ease, color .15s ease;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stLinkButton > a:hover {{
    border-color: var(--ink-3) !important;
    color: var(--ink) !important;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
}}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
    color: #fff !important;
}}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div, .stDateInput input {{
    border-radius: 7px !important;
    border: 1.5px solid var(--line) !important;
    font-family: {FONT_BODY} !important;
}}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 10px !important;
    border-color: var(--line) !important;
    background: var(--surface);
}}
[data-testid="stMetricValue"] {{
    font-family: {FONT_MONO} !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
}}
[data-testid="stMetricLabel"] {{ color: var(--ink-3) !important; font-size: 0.8rem !important; }}
code, .stCode, [data-testid="stMarkdownContainer"] code {{
    font-family: {FONT_MONO} !important;
    background: #F1F2F6 !important;
    color: var(--ink) !important;
    border-radius: 5px !important;
}}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--ink-3) !important; }}
[data-testid="stNotificationContentSuccess"] {{ background: var(--teal-soft) !important; border-color: var(--teal) !important; }}
[data-testid="stNotificationContentInfo"] {{ background: var(--accent-soft) !important; border-color: var(--accent-soft-line) !important; }}
[data-testid="stNotificationContentWarning"] {{ background: var(--amber-soft) !important; border-color: var(--amber) !important; }}
[data-testid="stNotificationContentError"] {{ background: var(--danger-soft) !important; border-color: var(--danger) !important; }}
.stTabs [data-baseweb="tab"] {{ font-family: {FONT_BODY} !important; font-weight: 600 !important; color: var(--ink-3) !important; }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; }}
</style>
"""


def inject_theme():
    """Load the shared fonts and component styling. Call once, near the
    top of app.py, right after st.set_page_config().

    st.html(), not st.markdown(..., unsafe_allow_html=True): found live
    2026-09-03 -- the CSS's own curly braces get caught by Streamlit's
    markdown renderer, which printed the entire stylesheet as literal
    on-page text instead of a <style> block. st.html() renders raw HTML
    directly, no markdown parsing pass to collide with."""
    st.html(_CSS)
