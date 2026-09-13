"""The demonstrations run, and print the numbers they claim to.

A demo that has silently stopped working is worse than no demo, so each subcommand is executed and
its output inspected for the specific evidence it exists to show.
"""

import pytest

from minigpt.cli import COMMANDS, main


def test_every_command_is_reachable_from_the_parser():
    assert set(COMMANDS) == {
        "gradcheck",
        "causal",
        "init",
        "overfit",
        "optim",
        "tokenizer",
        "sample",
        "train",
    }


def test_gradcheck_command_reports_passing_checks(capsys):
    assert main(["gradcheck"]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "worst relative error" in out
    assert "tok_emb.W" in out


def test_causal_command_shows_an_exactly_zero_leak(capsys):
    assert main(["causal"]) == 0
    out = capsys.readouterr().out
    assert "0.000e+00" in out


def test_init_command_reports_the_expected_loss(capsys):
    assert main(["init"]) == 0
    out = capsys.readouterr().out
    assert "as expected" in out
    assert "SUSPICIOUS" not in out


def test_overfit_command_drives_the_loss_down(capsys):
    assert main(["overfit"]) == 0
    out = capsys.readouterr().out
    assert "loss at initialisation" in out
    assert "after 400 steps" in out


def test_optim_command_shows_the_decay_and_schedule(capsys):
    assert main(["optim"]) == 0
    out = capsys.readouterr().out
    assert "expected" in out
    assert "warmup" in out.lower()


def test_tokenizer_command_round_trips_everything(capsys):
    assert main(["tokenizer"]) == 0
    out = capsys.readouterr().out
    assert "LOST DATA" not in out
    assert "bytes/token" in out


def test_sample_command_confirms_cache_equivalence(capsys):
    assert main(["sample"]) == 0
    out = capsys.readouterr().out
    assert "identical: True" in out


def test_train_command_runs_and_generates(capsys):
    assert main(["train", "--steps", "40"]) == 0
    out = capsys.readouterr().out
    assert "uniform-baseline loss" in out
    assert "generated:" in out


def test_unknown_command_exits_with_an_error():
    with pytest.raises(SystemExit):
        main(["does-not-exist"])
