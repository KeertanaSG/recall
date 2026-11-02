import jinja2
import pytest

from recall.prompts import render

SYSTEM_PROMPTS = [
    "agent.system",
    "search.plan_system",
    "search.grade_system",
    "tables.design_system",
    "nudges.judge_system",
]


def test_system_prompts_render_without_values():
    assert all(render(name) for name in SYSTEM_PROMPTS)


def test_a_missing_value_fails_loudly():
    with pytest.raises(jinja2.UndefinedError):
        render("search.plan", subjects="movies")


def test_saved_text_is_never_run_as_template_code():
    text = "{{ 7 * 7 }} {% if x %} {# hidden #}"
    assert text in render("agent.turn", context="c", text=text)
