"""Writing a figure: a redraw writes the same bytes, and a raster is left alone."""

from __future__ import annotations

import pytest


class TestWritingAFigure:
    """`save_figure` is where every figure of the repository is written."""

    def test_a_raster_format_is_written_unchanged(self, tmp_path):
        import matplotlib.pyplot as plt

        from aas_shortcut.define_figure_theme import apply_theme, save_figure

        apply_theme(plt)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = save_figure(fig, tmp_path / "probe.png")
        plt.close(fig)

        assert path.read_bytes().startswith(b"\x89PNG")

    @pytest.mark.parametrize("extension", ["svg", "pdf", "png"])
    def test_redrawing_writes_the_same_bytes(self, tmp_path, extension):
        """A run manifest's digest must change only when the figure does."""
        import matplotlib.pyplot as plt

        from aas_shortcut.define_figure_theme import apply_theme, save_figure

        apply_theme(plt)
        written = []
        for attempt in range(2):
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], clip_on=True)
            written.append(save_figure(fig, tmp_path / f"{attempt}.{extension}"))
            plt.close(fig)

        assert written[0].read_bytes() == written[1].read_bytes()
