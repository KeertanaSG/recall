from pathlib import Path

import jinja2
import pytest

import recall.prompts
from recall.prompts import render

STATIC = sorted(path.stem for path in Path(recall.prompts.__file__).parent.glob("*_system.j2"))


def test_system_prompts_render_without_values():
    assert len(STATIC) == 5
    assert all(render(name) for name in STATIC)


def test_a_missing_value_fails_loudly():
    with pytest.raises(jinja2.UndefinedError):
        render("search_plan", subjects="movies")


def test_saved_text_is_never_run_as_template_code():
    text = "{{ 7 * 7 }} {% if x %} {# hidden #}"
    assert text in render("turn", context="c", text=text)
