"""Every prompt and long message, as named blocks in Jinja templates grouped by subject.

    render("search.plan", question=...)  is the `plan` block of search.j2

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
    file, block = name.split(".")
    template = _templates.get_template(f"{file}.j2")
    return "".join(template.blocks[block](template.new_context(values)))
