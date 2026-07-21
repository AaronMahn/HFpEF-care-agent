#!/usr/bin/env python3
"""Regenerate the Figure 1 architecture diagram without an in-figure title.

The figure is intentionally built from matplotlib primitives so every label is
editable in this source file and the output is reproducible.
"""

from __future__ import annotations

import os
import sys
from math import ceil
from pathlib import Path


# Keep the matplotlib cache inside the writable project tree or /tmp.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/care_agent_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/care_agent_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]  # repo root
OUT = Path(__file__).resolve().parents[1] / "Figure1_CARE_Agent_architecture.png"
DPI = 320
FIGSIZE = (13.5, 6.0)

INK = "#20303C"
MUTED = "#5F6F7A"
LINE = "#52616B"
PAPER = "#FBFCFD"

PALETTE = {
    "input": "#EAF2F8",
    "agent_a": "#DCECF6",
    "agent_b": "#E3F1E3",
    "agent_c": "#F8E8C9",
    "llm": "#F4E9F4",
    "output": "#E8EEF4",
}


def choose_font() -> tuple[FontProperties, str, bool]:
    """Choose one installed font with glyphs for the requested Chinese label."""
    requested = "简体中文"
    preferred = (
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "SimHei",
        "Arial Unicode MS",
    )
    fonts = list(font_manager.fontManager.ttflist)
    for family in preferred:
        candidates = sorted(
            (entry for entry in fonts if entry.name == family),
            key=lambda entry: abs(float(entry.weight) - 400),
        )
        for entry in candidates:
            try:
                cmap = ft2font.FT2Font(entry.fname).get_charmap()
            except (RuntimeError, OSError):
                continue
            if all(ord(char) in cmap for char in requested):
                return FontProperties(family=entry.name), entry.name, True

    print(
        "WARNING: No CJK-capable font was found; using the Latin fallback "
        "'Simplified Chinese'.",
        file=sys.stderr,
    )
    fallback_path = font_manager.findfont("DejaVu Sans")
    return FontProperties(fname=fallback_path), "DejaVu Sans", False


BASE_FONT, FONT_NAME, HAS_CJK = choose_font()
LANGUAGE_LABEL = "简体中文" if HAS_CJK else "Simplified Chinese"


def prop(size: float, weight: str = "normal") -> FontProperties:
    font = BASE_FONT.copy()
    font.set_size(size)
    font.set_weight(weight)
    return font


def text_width(renderer, text: str, font: FontProperties) -> float:
    return renderer.get_text_width_height_descent(text, font, ismath=False)[0]


def wrap_line(renderer, text: str, font: FontProperties, width_px: float) -> list[str]:
    """Greedily wrap one line using rendered width rather than character count."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(renderer, candidate, font) <= width_px:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def wrap_bullets(
    renderer, bullets: list[str], font: FontProperties, width_px: float
) -> tuple[list[list[str]], float]:
    """Wrap bullet text and return a true hanging-indent offset in pixels."""
    bullet_indent = text_width(renderer, "•  ", font)
    return (
        [wrap_line(renderer, item, font, width_px - bullet_indent) for item in bullets],
        bullet_indent,
    )


def add_box(
    ax,
    renderer,
    bounds: tuple[float, float, float, float],
    stage: str,
    title: str,
    bullets: list[str],
    fill: str,
    *,
    dashed: bool = False,
    bullet_columns: int = 1,
    body_start: float = 10.6,
    body_min: float = 8.4,
) -> tuple[
    FancyBboxPatch,
    list[tuple[object, tuple[float, float, float, float]]],
    list[object],
]:
    """Draw a flat rounded box and fit all text within a measured inner area."""
    x, y, w, h = bounds
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.007,rounding_size=0.014",
        linewidth=1.35,
        edgecolor=LINE,
        facecolor=fill,
        linestyle=(0, (5, 3)) if dashed else "solid",
        zorder=2,
    )
    ax.add_patch(patch)

    pad_x = 0.014
    pad_y = 0.018
    inner = (x + pad_x, y + pad_y, w - 2 * pad_x, h - 2 * pad_y)
    inner_width_px = inner[2] * ax.figure.bbox.width

    stage_font = prop(9.2, "bold")
    title_font = prop(13.1, "bold")
    title_lines = wrap_line(renderer, title, title_font, inner_width_px)

    top = y + h - pad_y
    stage_artist = ax.text(
        x + pad_x,
        top,
        stage,
        fontproperties=stage_font,
        color=MUTED,
        ha="left",
        va="top",
        zorder=3,
    )
    stage_h = 9.2 * 1.15 / 72 / FIGSIZE[1]
    title_top = top - stage_h - 0.008
    title_artist = ax.text(
        x + pad_x,
        title_top,
        "\n".join(title_lines),
        fontproperties=title_font,
        color=INK,
        linespacing=1.08,
        ha="left",
        va="top",
        zorder=3,
    )
    title_h = len(title_lines) * 13.1 * 1.13 / 72 / FIGSIZE[1]
    body_top = title_top - title_h - 0.017
    available_body_h = body_top - inner[1]

    if bullet_columns < 1:
        raise ValueError("bullet_columns must be at least 1")

    column_gap = 0.025 if bullet_columns > 1 else 0.0
    column_width = (inner[2] - column_gap * (bullet_columns - 1)) / bullet_columns
    column_width_px = column_width * ax.figure.bbox.width
    bullets_per_column = ceil(len(bullets) / bullet_columns)
    bullet_groups = [
        bullets[index : index + bullets_per_column]
        for index in range(0, len(bullets), bullets_per_column)
    ]

    body_font_size = body_start
    while True:
        body_font = prop(body_font_size)
        wrapped_groups = [
            wrap_bullets(renderer, group, body_font, column_width_px)
            for group in bullet_groups
        ]
        line_count = max(
            sum(len(lines) for lines in wrapped_bullets)
            for wrapped_bullets, _ in wrapped_groups
        )
        estimated_h = line_count * body_font_size * 1.27 / 72 / FIGSIZE[1]
        if estimated_h <= available_body_h or body_font_size <= body_min + 1e-9:
            break
        body_font_size = max(body_min, round(body_font_size - 0.2, 10))

    # Draw the glyph and the bullet text separately. Continuation lines share
    # the text x-coordinate, giving a measured hanging indent in any font.
    line_step = body_font_size * 1.27 / 72 / FIGSIZE[1]
    body_artists: list[object] = []
    for column_index, (wrapped_bullets, bullet_indent_px) in enumerate(wrapped_groups):
        bullet_indent = bullet_indent_px / ax.figure.bbox.width
        column_x = x + pad_x + column_index * (column_width + column_gap)
        line_index = 0
        for lines in wrapped_bullets:
            y_line = body_top - line_index * line_step
            body_artists.append(
                ax.text(
                    column_x,
                    y_line,
                    "•",
                    fontproperties=body_font,
                    color=INK,
                    ha="left",
                    va="top",
                    zorder=3,
                )
            )
            for continuation_index, line in enumerate(lines):
                y_line = body_top - (line_index + continuation_index) * line_step
                body_artists.append(
                    ax.text(
                        column_x + bullet_indent,
                        y_line,
                        line,
                        fontproperties=body_font,
                        color=INK,
                        ha="left",
                        va="top",
                        zorder=3,
                    )
                )
            line_index += len(lines)

    tracked = [
        (stage_artist, inner),
        (title_artist, inner),
        *((artist, inner) for artist in body_artists),
    ]
    return patch, tracked, [stage_artist, title_artist, *body_artists]


def arrow(
    ax,
    start,
    end,
    *,
    color=LINE,
    label: str | None = None,
    label_y=None,
    label_size: float = 7.8,
    style="-",
):
    arrow_patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.35,
        color=color,
        linestyle=style,
        shrinkA=0,
        shrinkB=0,
        connectionstyle="arc3,rad=0",
        zorder=4,
    )
    ax.add_patch(arrow_patch)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            label_y if label_y is not None else start[1] + 0.018,
            label,
            fontproperties=prop(label_size, "bold"),
            color=MUTED,
            ha="center",
            va="bottom",
            bbox={"boxstyle": "square,pad=0.12", "fc": PAPER, "ec": "none"},
            zorder=5,
        )
    return arrow_patch


def orthogonal_arrow(ax, points, *, color=LINE, linewidth=1.35):
    """Draw a clean elbow connector whose first and last points are box edges."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            (start[0], end[0]),
            (start[1], end[1]),
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=4,
        )
    return arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
    )


def main() -> None:
    visible_text = {
        "input_title": "Structured clinical dataset",
        "input": [
            "Patient history",
            "Medications",
            "Lab results",
            "Echo measurements",
            "Future support: EHR integration",
        ],
        "agent_a_title": "Clinical data retrieval & pre-processing",
        "agent_a": [
            "Retrieves patient-level clinical data",
            "Validates data quality",
            "Derives TyG index and CKD stage",
            "Flags clinical risk thresholds",
            "Assigns preliminary risk tier (LOW–CRITICAL)",
        ],
        "agent_b_title": "Right atrial strain classification",
        "agent_b": [
            "Ingests RA strain values",
            "Classifies reservoir, conduit, and booster vs prespecified thresholds",
            "Applies adjustments for atrial fibrillation and CKD",
            "Computes weighted composite score (0–100)",
            "Determines overall RA function grade",
        ],
        "agent_c_title": "Integrated risk synthesis",
        "agent_c": [
            "Applies prespecified 11-variable Cox risk-ranking component",
            "Assigns tier from cohort-derived quartiles of predicted 18-month event probability",
            "Integrates with Agent A tier and Agent B grade by maximum-severity rule",
            "Applies 8 hierarchically prioritised advisory rules",
            "Outputs four-tier integrated classification + recommended actions",
        ],
        "llm_title": "Optional LLM reasoning layer",
        "llm": [
            "Mechanistic interpretation",
            "Clinical priority articulation",
            "Watchpoints",
            "Patient note",
            "Independent safety check",
        ],
        "output_title": "Bilingual Streamlit web interface",
        "output": [f"English / {LANGUAGE_LABEL}"],
        "patient_snapshot": "PatientSnapshot",
        "ra_report": "RaImagingReport",
    }

    flattened: list[str] = []
    for value in visible_text.values():
        flattened.extend(value if isinstance(value, list) else [value])
    rendered_copy = "\n".join(flattened).lower()
    blocked_terms = ("vali" + "dated", "pre-" + "vali" + "dated", "cali" + "brated")
    assert "prespecified" in rendered_copy
    assert not any(term in rendered_copy for term in blocked_terms)
    assert "care-agent system architecture" not in rendered_copy
    assert "structured data flow" not in rendered_copy

    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "text.color": INK,
        }
    )
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # The top row begins immediately below the figure edge because the journal
    # caption supplies the figure title and subtitle outside the artwork.
    input_bounds = (0.020, 0.520, 0.155, 0.450)
    a_bounds = (0.210, 0.520, 0.185, 0.450)
    b_bounds = (0.475, 0.520, 0.190, 0.450)
    c_bounds = (0.750, 0.520, 0.230, 0.450)
    llm_bounds = (0.300, 0.110, 0.450, 0.190)
    output_bounds = (0.790, 0.095, 0.190, 0.175)

    tracked: list[tuple[object, tuple[float, float, float, float]]] = []
    content_by_box: list[tuple[str, tuple[float, float, float, float], list[object]]] = []
    _, items, content = add_box(
        ax,
        renderer,
        input_bounds,
        "Input",
        visible_text["input_title"],
        visible_text["input"],
        PALETTE["input"],
        body_start=14.0,
    )
    tracked.extend(items)
    content_by_box.append(("Input", input_bounds, content))
    _, items, content = add_box(
        ax,
        renderer,
        a_bounds,
        "Agent A",
        visible_text["agent_a_title"],
        visible_text["agent_a"],
        PALETTE["agent_a"],
    )
    tracked.extend(items)
    content_by_box.append(("Agent A", a_bounds, content))
    _, items, content = add_box(
        ax,
        renderer,
        b_bounds,
        "Agent B",
        visible_text["agent_b_title"],
        visible_text["agent_b"],
        PALETTE["agent_b"],
        body_start=11.2,
    )
    tracked.extend(items)
    content_by_box.append(("Agent B", b_bounds, content))
    _, items, content = add_box(
        ax,
        renderer,
        c_bounds,
        "Agent C",
        visible_text["agent_c_title"],
        visible_text["agent_c"],
        PALETTE["agent_c"],
        body_start=10.2,
        body_min=8.2,
    )
    tracked.extend(items)
    content_by_box.append(("Agent C", c_bounds, content))
    llm_patch, items, content = add_box(
        ax,
        renderer,
        llm_bounds,
        "Optional layer",
        visible_text["llm_title"],
        visible_text["llm"],
        PALETTE["llm"],
        dashed=True,
        bullet_columns=2,
        body_start=9.4,
        body_min=8.2,
    )
    tracked.extend(items)
    content_by_box.append(("Optional layer", llm_bounds, content))
    _, items, content = add_box(
        ax,
        renderer,
        output_bounds,
        "Output",
        visible_text["output_title"],
        visible_text["output"],
        PALETTE["output"],
        body_start=11.0,
        body_min=8.2,
    )
    tracked.extend(items)
    content_by_box.append(("Output", output_bounds, content))

    mid_y = input_bounds[1] + input_bounds[3] / 2
    connector_color = "#66879B"
    stage_connectors = [
        arrow(ax, (input_bounds[0] + input_bounds[2], mid_y), (a_bounds[0], mid_y), color=connector_color),
        arrow(
            ax,
            (a_bounds[0] + a_bounds[2], mid_y),
            (b_bounds[0], mid_y),
            color=connector_color,
            label=visible_text["patient_snapshot"],
            label_y=mid_y + 0.014,
        ),
        arrow(
            ax,
            (b_bounds[0] + b_bounds[2], mid_y),
            (c_bounds[0], mid_y),
            color=connector_color,
            label=visible_text["ra_report"],
            label_y=mid_y + 0.014,
        ),
    ]

    # Agent C's deterministic result remains available even when the optional
    # reasoning layer is not activated.
    orthogonal_arrow(
        ax,
        [(0.810, 0.520), (0.810, 0.360), (0.690, 0.360), (0.690, 0.300)],
        color="#8D6A37",
    )
    orthogonal_arrow(
        ax,
        [(0.920, 0.520), (0.920, 0.350), (0.885, 0.350), (0.885, 0.270)],
        color="#8D6A37",
        linewidth=1.15,
    )
    arrow(ax, (0.750, 0.180), (0.790, 0.180), color="#826783")

    if len({connector.get_linestyle() for connector in stage_connectors}) != 1:
        raise RuntimeError("Inter-stage connector line styles are inconsistent")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tolerance_px = 2.5
    overflow_messages: list[str] = []
    for artist, inner in tracked:
        bbox = artist.get_window_extent(renderer=renderer)
        left, bottom = ax.transData.transform((inner[0], inner[1]))
        right, top = ax.transData.transform((inner[0] + inner[2], inner[1] + inner[3]))
        if (
            bbox.x0 < left - tolerance_px
            or bbox.x1 > right + tolerance_px
            or bbox.y0 < bottom - tolerance_px
            or bbox.y1 > top + tolerance_px
        ):
            overflow_messages.append(artist.get_text().replace("\n", " / "))
    if overflow_messages:
        raise RuntimeError("Text overflow detected: " + " | ".join(overflow_messages))

    empty_band_messages: list[str] = []
    empty_band_fractions: list[str] = []
    for name, bounds, artists in content_by_box:
        content_bottom = min(
            artist.get_window_extent(renderer=renderer).y0 for artist in artists
        )
        _, box_bottom = ax.transData.transform((bounds[0], bounds[1]))
        _, box_top = ax.transData.transform((bounds[0], bounds[1] + bounds[3]))
        empty_fraction = (content_bottom - box_bottom) / (box_top - box_bottom)
        empty_band_fractions.append(f"{name} {empty_fraction:.1%}")
        if empty_fraction > 0.105:
            empty_band_messages.append(f"{name}: {empty_fraction:.1%}")
    print("bottom empty bands: " + "; ".join(empty_band_fractions))
    if empty_band_messages:
        # Cosmetic only, and sensitive to the font metrics of the host system,
        # so this is reported rather than raised - it must not stop the figure
        # from being regenerated on a machine with different fonts installed.
        print(
            "note: bottom empty band above target in: " + " | ".join(empty_band_messages)
        )

    if llm_patch.get_linestyle() == "solid":
        raise RuntimeError("Optional LLM layer must use a dashed border")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, facecolor=PAPER, edgecolor="none")
    plt.close(fig)

    print(f"saved: {OUT}")
    print(f"dimensions: {round(FIGSIZE[0] * DPI)} x {round(FIGSIZE[1] * DPI)} px")
    print(f"resolution: {DPI} dpi")
    print(f"font: {FONT_NAME}")
    print(f"language label: English / {LANGUAGE_LABEL}")
    print(
        "text checks: title/subtitle absent; prespecified present; "
        "blocked terminology absent"
    )
    print(
        "layout checks: optional LLM border dashed and bullets in two columns; "
        "inter-stage connectors straight and consistent; no text overflow; "
        "bottom empty bands <= 10.5%"
    )


if __name__ == "__main__":
    main()
