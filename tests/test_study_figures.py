from moe_pilot.study_analysis import write_table
from moe_pilot.study_figures import build_study_figures


def test_repeated_figure_exports_preserve_source_and_output_hashes(tmp_path):
    write_table(
        tmp_path / "tables" / "model_summary.csv",
        [
            {
                "model": "olmoe-0924-step10000",
                "metric": "accuracy",
                "estimate": 0.5,
                "ci_low": 0.3,
                "ci_high": 0.7,
            },
            {
                "model": "olmoe-0924-step10000",
                "metric": "choice_nll",
                "estimate": 1.0,
                "ci_low": 0.7,
                "ci_high": 1.3,
            },
        ],
    )
    first = build_study_figures(tmp_path)
    second = build_study_figures(tmp_path)
    assert first == second
    assert set(first["checkpoint_comparison"]["files"]) == {
        "checkpoint_comparison.png",
        "checkpoint_comparison.svg",
        "checkpoint_comparison.pdf",
    }
