"""Minimal SVG builder library for code-based SVG generation.

The model writes Python code using these functions, we execute it, get SVG.
"""


class SVGBuilder:
    """Simple API for building SVGs programmatically."""

    def __init__(self, width: int = 200, height: int = 200):
        self.width = width
        self.height = height
        self.elements: list[str] = []

    def background(self, color: str = "white") -> None:
        """Fill the background with a color."""
        self.elements.insert(0, f'<rect width="{self.width}" height="{self.height}" fill="{color}"/>')

    def rect(self, x: float, y: float, w: float = 0, h: float = 0, fill: str = "black", stroke: str = "none", stroke_width: float = 1, rx: float = 0, width: float = 0, height: float = 0) -> None:
        """Draw a rectangle. Accepts w/h or width/height."""
        w = w or width
        h = h or height
        attrs = f'x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"'
        if stroke != "none":
            attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        if rx > 0:
            attrs += f' rx="{rx}"'
        self.elements.append(f"<rect {attrs}/>")

    def circle(self, cx: float, cy: float, r: float, fill: str = "black", stroke: str = "none", stroke_width: float = 1) -> None:
        """Draw a circle."""
        attrs = f'cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"'
        if stroke != "none":
            attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        self.elements.append(f"<circle {attrs}/>")

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, fill: str = "black", stroke: str = "none", stroke_width: float = 1) -> None:
        """Draw an ellipse."""
        attrs = f'cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}"'
        if stroke != "none":
            attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        self.elements.append(f"<ellipse {attrs}/>")

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "black", stroke_width: float = 2) -> None:
        """Draw a line."""
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{stroke_width}"/>')

    def polygon(self, points: list[tuple[float, float]], fill: str = "black", stroke: str = "none", stroke_width: float = 1) -> None:
        """Draw a polygon from a list of (x, y) points."""
        pts = " ".join(f"{x},{y}" for x, y in points)
        attrs = f'points="{pts}" fill="{fill}"'
        if stroke != "none":
            attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        self.elements.append(f"<polygon {attrs}/>")

    def polyline(self, points: list[tuple[float, float]], fill: str = "none", stroke: str = "black", stroke_width: float = 2) -> None:
        """Draw a polyline (open path) from a list of (x, y) points."""
        pts = " ".join(f"{x},{y}" for x, y in points)
        attrs = f'points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"'
        self.elements.append(f"<polyline {attrs}/>")

    def path(self, d: str, fill: str = "black", stroke: str = "none", stroke_width: float = 1) -> None:
        """Draw a path using SVG path commands (M, L, C, Q, A, Z)."""
        attrs = f'd="{d}" fill="{fill}"'
        if stroke != "none":
            attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        self.elements.append(f"<path {attrs}/>")

    def text(self, x: float, y: float, content: str, font_size: float = 16, fill: str = "black", font_family: str = "Arial", anchor: str = "start") -> None:
        """Draw text."""
        self.elements.append(
            f'<text x="{x}" y="{y}" font-size="{font_size}" fill="{fill}" '
            f'font-family="{font_family}" text-anchor="{anchor}">{content}</text>'
        )

    def arrow(self, x: float, y: float, width: float = 40, height: float = 20, fill: str = "black", direction: str = "right", **kwargs) -> None:
        """Draw an arrow using a polygon."""
        w, h = width, height
        if direction == "right":
            self.polygon([(x, y + h*0.25), (x + w*0.6, y + h*0.25), (x + w*0.6, y), (x + w, y + h*0.5), (x + w*0.6, y + h), (x + w*0.6, y + h*0.75), (x, y + h*0.75)], fill=fill)
        elif direction == "left":
            self.polygon([(x + w, y + h*0.25), (x + w*0.4, y + h*0.25), (x + w*0.4, y), (x, y + h*0.5), (x + w*0.4, y + h), (x + w*0.4, y + h*0.75), (x + w, y + h*0.75)], fill=fill)
        elif direction == "up":
            self.polygon([(x + w*0.25, y + h), (x + w*0.25, y + h*0.4), (x, y + h*0.4), (x + w*0.5, y), (x + w, y + h*0.4), (x + w*0.75, y + h*0.4), (x + w*0.75, y + h)], fill=fill)
        else:  # down
            self.polygon([(x + w*0.25, y), (x + w*0.25, y + h*0.6), (x, y + h*0.6), (x + w*0.5, y + h), (x + w, y + h*0.6), (x + w*0.75, y + h*0.6), (x + w*0.75, y)], fill=fill)

    def arc(self, cx: float, cy: float, r: float, start_angle: float = 0, end_angle: float = 360, fill: str = "black", **kwargs) -> None:
        """Draw an arc/pie slice using a path."""
        import math
        sa = math.radians(start_angle)
        ea = math.radians(end_angle)
        x1 = cx + r * math.cos(sa)
        y1 = cy + r * math.sin(sa)
        x2 = cx + r * math.cos(ea)
        y2 = cy + r * math.sin(ea)
        large = 1 if (end_angle - start_angle) > 180 else 0
        d = f"M {cx} {cy} L {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f} Z"
        self.path(d, fill=fill)

    def star(self, cx: float, cy: float, r: float, points: int = 5, fill: str = "gold", **kwargs) -> None:
        """Draw a star."""
        import math
        pts = []
        for i in range(points * 2):
            angle = math.radians(i * 180 / points - 90)
            radius = r if i % 2 == 0 else r * 0.4
            pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        self.polygon(pts, fill=fill)

    def __getattr__(self, name):
        """Catch-all for methods the model invents — silently ignore."""
        def noop(*args, **kwargs):
            pass
        return noop

    def group(self, transform: str = "") -> "SVGGroup":
        """Create a group with an optional transform."""
        return SVGGroup(self, transform)

    def render(self) -> str:
        """Render to SVG string."""
        inner = "\n  ".join(self.elements)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'width="256" height="256">\n  {inner}\n</svg>'
        )


class SVGGroup:
    """Group element for transforms."""

    def __init__(self, parent: SVGBuilder, transform: str = ""):
        self.parent = parent
        self.transform = transform
        self.elements: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        inner = "\n    ".join(self.elements)
        if self.transform:
            self.parent.elements.append(f'<g transform="{self.transform}">\n    {inner}\n  </g>')
        else:
            self.parent.elements.append(f"<g>\n    {inner}\n  </g>")


# Convenience function
def create_svg(width: int = 200, height: int = 200) -> SVGBuilder:
    """Create a new SVG builder."""
    return SVGBuilder(width, height)
