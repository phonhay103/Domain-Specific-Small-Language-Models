"""Dark-mode optimized, soothing UI component library and visual theme for Rich.

Designed for high readability, low eye fatigue in dark terminals, clear typography,
and functional composability.
"""

import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Eye-friendly Dark-mode Color Constants (Catppuccin Macchiato & Tokyo Night)
# ---------------------------------------------------------------------------
COLOR_PRIMARY = "#89b4fa"  # Soft Sky Blue
COLOR_SECONDARY = "#89dceb"  # Soft Aqua / Cyan
COLOR_HEADER = "#b4befe"  # Soft Lavender
COLOR_TEXT = "#cdd6f4"  # Main text (calm light blue-grey)
COLOR_MUTED = "#6c7086"  # Subtle grey
COLOR_DIM = "#9399b2"  # Secondary readable grey
COLOR_SUCCESS = "#a6e3a1"  # Pastel Sage Green
COLOR_WARNING = "#fab387"  # Soft Amber / Peach
COLOR_GOLD = "#f9e2af"  # Warm sand
COLOR_ERROR = "#f38ba8"  # Soft Coral
COLOR_BORDER = "#45475a"  # Low-contrast frame border
COLOR_CARD_BORDER = "#585b70"  # Medium border

# Styles for direct Table column / element styling
STYLE_INDEX = "#a6adc8"
STYLE_NUMBER = "#fab387"
STYLE_PRIMARY = "#89b4fa"
STYLE_SECONDARY = "#89dceb"
STYLE_SUCCESS = "#a6e3a1"
STYLE_WARNING = "#fab387"
STYLE_TEXT = "#cdd6f4"
STYLE_MUTED = "#6c7086"

EYE_FRIENDLY_THEME = Theme(
    {
        "primary": COLOR_PRIMARY,
        "secondary": COLOR_SECONDARY,
        "header": f"bold {COLOR_HEADER}",
        "success": COLOR_SUCCESS,
        "warning": COLOR_WARNING,
        "gold": COLOR_GOLD,
        "error": COLOR_ERROR,
        "text": COLOR_TEXT,
        "muted": COLOR_MUTED,
        "dim": COLOR_DIM,
        "border": COLOR_BORDER,
    }
)

console = Console(theme=EYE_FRIENDLY_THEME)

DEFAULT_STEP_DELAY = 0.4


def pause(delay: float = DEFAULT_STEP_DELAY) -> None:
    """Non-intrusive step delay for animated pace."""
    time.sleep(delay)


# ---------------------------------------------------------------------------
# Functional UI Renderers
# ---------------------------------------------------------------------------
def render_banner(
    title: str,
    subtitle: str = "Domain-Specific Small Language Models",
    metadata: Mapping[str, Any] | None = None,
    icon: str = "🚀",
) -> None:
    """Render a clean, calm banner header at the start of script execution."""
    meta_lines: list[str] = []
    if metadata:
        meta_items = [f"[muted]{k}:[/muted] [text]{v}[/text]" for k, v in metadata.items()]
        meta_lines.append("  •  ".join(meta_items))

    content = f"[header]{icon} {title}[/header]\n[muted]{subtitle}[/muted]"
    if meta_lines:
        content += f"\n[dim]{'─' * 48}[/dim]\n" + "\n".join(meta_lines)

    panel = Panel(
        content,
        box=box.ROUNDED,
        border_style=COLOR_CARD_BORDER,
        padding=(1, 2),
    )
    console.print(panel)
    pause()


def render_step(step_number: int, title: str, icon: str = "◆") -> None:
    """Render a soothing, low-glare step delimiter rule."""
    console.rule(
        f"[dim]{icon}[/dim] [primary]Step {step_number}:[/primary] [text]{title}[/text]",
        style=COLOR_BORDER,
    )
    pause()


def create_table(
    title: str,
    columns: Sequence[tuple[str, str, str]],  # (Header, Style, Justify)
    rows: Sequence[Sequence[Any]],
    show_lines: bool = False,
) -> Table:
    """Construct an eye-friendly, low-contrast table."""
    table = Table(
        title=f"[header]{title}[/header]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style=f"bold {COLOR_PRIMARY}",
        show_lines=show_lines,
        padding=(0, 1),
    )
    for name, style, justify in columns:
        table.add_column(name, style=style, justify=justify)  # type: ignore[arg-type]

    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    return table


def render_card(
    title: str,
    content: str,
    border_style: str = COLOR_CARD_BORDER,
    icon: str = "ℹ️",
) -> None:
    """Render a soft result or info card."""
    panel = Panel(
        content,
        title=f"[dim]{icon}[/dim] [header]{title}[/header]",
        box=box.ROUNDED,
        border_style=border_style,
        padding=(0, 2),
    )
    console.print(panel)
    pause()


def render_takeaways(
    points: Sequence[tuple[str, str]],  # (Title, Description)
    title: str = "Educational Insights & Takeaways",
    icon: str = "💡",
) -> None:
    """Render structured key takeaways in a gentle sand/amber tone."""
    formatted_points = [
        f"[gold]•[/gold] [primary]{pt_title}:[/primary] [text]{pt_desc}[/text]" for pt_title, pt_desc in points
    ]
    content = "\n".join(formatted_points)
    panel = Panel(
        content,
        title=f"[gold]{icon} {title}[/gold]",
        box=box.ROUNDED,
        border_style=COLOR_CARD_BORDER,
        padding=(1, 2),
    )
    console.print(panel)


def render_code_block(code: str, language: str = "python", title: str = "Generated Code") -> None:
    """Render syntax highlighted code inside an eye-friendly panel."""
    syntax = Syntax(code.strip(), language, theme="monokai", line_numbers=True)
    panel = Panel(
        syntax,
        title=f"[secondary]⚡ {title}[/secondary]",
        box=box.ROUNDED,
        border_style=COLOR_CARD_BORDER,
    )
    console.print(panel)
    pause()


@contextmanager
def status_spinner(message: str) -> Generator[None, None, None]:
    """Provide a quiet, smooth spinner for async/heavy processing."""
    with console.status(f"[dim]⠋[/dim] [primary]{message}[/primary]", spinner="dots"):
        yield
