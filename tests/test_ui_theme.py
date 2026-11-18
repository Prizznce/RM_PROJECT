from unittest.mock import MagicMock, patch
import pytest
from src.app_ui.theme import (
    render_kpi_card,
    render_research_master_ribbon,
    render_hero_ticker,
    compute_efficiency_rating,
    get_efficiency_badge_html,
    COLOR_CYAN,
    COLOR_DORMANT,
    COLOR_AWAKE,
    COLOR_ORACLE,
)


def test_render_kpi_card_no_delta():
    """Verify KPI card renders subtext properly without CommonMark 4-space indentation code-block bugs."""
    with patch("streamlit.markdown") as mock_st_markdown:
        render_kpi_card(
            label="Grid Base Capacity",
            value="50,000 MW",
            subtext="Synchronous Scale (REE / POSOCO)",
            border_color=COLOR_CYAN,
        )
        assert mock_st_markdown.called
        args, kwargs = mock_st_markdown.call_args
        html = args[0]
        assert kwargs.get("unsafe_allow_html") is True

        # Check content
        assert "Grid Base Capacity" in html
        assert "50,000 MW" in html
        assert "Synchronous Scale (REE / POSOCO)" in html
        assert 'class="kpi-subtext"' in html

        # Crucial: Must NOT contain multi-line indented code block patterns
        for line in html.split("\n"):
            assert not line.startswith("    "), f"Indented line detected that would trigger code block: {line}"


def test_render_kpi_card_with_delta():
    """Verify KPI card renders delta class and subtext together correctly."""
    with patch("streamlit.markdown") as mock_st_markdown:
        render_kpi_card(
            label="CGT-v3 Acc",
            value="85.85%",
            delta="+0.71% vs HT",
            delta_color="positive",
            subtext="Champion Gating",
            border_color=COLOR_DORMANT,
        )
        assert mock_st_markdown.called
        args, _ = mock_st_markdown.call_args
        html = args[0]

        assert "delta-positive" in html
        assert "+0.71% vs HT" in html
        assert "Champion Gating" in html


def test_render_kpi_card_positional_resilience():
    """Verify legacy positional arguments are automatically mapped to subtext and border_color."""
    with patch("streamlit.markdown") as mock_st_markdown:
        render_kpi_card("OMS-MAB Duty (ρ)", "61.9%", "Wilson et al. (2024)", COLOR_AWAKE)
        assert mock_st_markdown.called
        args, _ = mock_st_markdown.call_args
        html = args[0]

        assert "OMS-MAB Duty (ρ)" in html
        assert "61.9%" in html
        assert "Wilson et al. (2024)" in html
        assert COLOR_AWAKE in html
        # Should not treat hex color as delta class
        assert f"delta-{COLOR_AWAKE}" not in html


def test_render_research_master_ribbon():
    """Verify master ribbon renders without 4-space indentation and highlights champion findings."""
    with patch("streamlit.markdown") as mock_st_markdown:
        render_research_master_ribbon()
        assert mock_st_markdown.called
        args, kwargs = mock_st_markdown.call_args
        html = args[0]
        assert kwargs.get("unsafe_allow_html") is True
        assert "👑 CHAMPION: CGT-v3 (8.8 / 10)" in html
        assert "85.85% (Europe)" in html
        assert "87.30% (India)" in html


def test_efficiency_rating_integrity():
    """Ensure Oracle is strictly pinned to 9.0 and CGT-v3 to 8.8."""
    oracle_score = compute_efficiency_rating(93.09, 876.9, model_name="Oracle Switch")
    assert oracle_score == 9.0

    cgt3_score = compute_efficiency_rating(85.85, 876.9, model_name="CGT-v3 (Proposed Champion)")
    assert cgt3_score == 8.8

    cgt2_score = compute_efficiency_rating(85.49, 416.1, model_name="CGT-v2 (Adaptive Dual-Gate)")
    assert cgt2_score == 8.5

    badge = get_efficiency_badge_html(cgt3_score)
    assert "8.8 / 10" in badge
    assert "DEPLOYABLE" in badge
