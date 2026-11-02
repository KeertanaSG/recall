"""Every prompt and long message lives here as a Jinja template, so wording is edited in one place.

Values passed to a template are inserted as plain text, so saved notes can't act as template code.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_templates = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    undefined=StrictUndefined,  # A missing value should fail, not send the model a broken prompt.
    autoescape=False,  # Prompts are plain text, not HTML.
)


def render(name: str, **values) -> str:
    return _templates.get_template(f"{name}.j2").render(**values)
