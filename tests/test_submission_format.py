"""Q5 -- Tests for the submission format utilities, independent of real data.
Catching a format bug here is much cheaper than burning a rate-limited
Codabench submission attempt (MIND: 1/day; EB-NeRD: 5/day) to discover it.
"""
from __future__ import annotations

import zipfile

import pytest

from src.submission.format import (
    format_line,
    scores_to_ranks,
    validate_submission,
    write_submission_file,
    zip_submission,
)


def test_scores_to_ranks_matches_guideline_example():
    # From the EB-NeRD guidelines: article_ids_inview order [9798759, 9798604, 9777339, 9798829],
    # and 9798829 (last in the list) should be rank 1 -> expected ranks [3,2,4,1]... wait, actual
    # example: ranks[139350] = [3,2,4,1] means: 1st article rank 3, 2nd article rank 2,
    # 3rd article rank 4, 4th article rank 1 (i.e. 4th article is most likely clicked).
    scores = [1.0, 2.0, 0.5, 3.0]  # 4th (index 3) is highest -> rank 1
    ranks = scores_to_ranks(scores)
    assert ranks == [3, 2, 4, 1]


def test_scores_to_ranks_is_a_valid_permutation():
    scores = [0.1, 0.9, 0.1, 0.5, 0.1]
    ranks = scores_to_ranks(scores)
    assert sorted(ranks) == [1, 2, 3, 4, 5]


def test_scores_to_ranks_ties_broken_by_position_deterministically():
    scores = [0.5, 0.5, 0.5]
    assert scores_to_ranks(scores) == [1, 2, 3]


def test_format_line_no_spaces_inside_brackets():
    line = format_line("24481", [4, 1, 3, 2])
    assert line == "24481 [4,1,3,2]"
    assert " " not in line.split(" ", 1)[1]  # no spaces after the impression_id


def test_validate_submission_passes_on_correct_input():
    rows = [("1", [2, 1]), ("2", [1, 3, 2])]
    validate_submission(rows, expected_ids_in_order=["1", "2"])  # should not raise


def test_validate_submission_catches_wrong_order():
    rows = [("2", [1]), ("1", [1])]
    with pytest.raises(AssertionError):
        validate_submission(rows, expected_ids_in_order=["1", "2"])


def test_validate_submission_catches_missing_impression():
    rows = [("1", [1])]
    with pytest.raises(AssertionError):
        validate_submission(rows, expected_ids_in_order=["1", "2"])


def test_validate_submission_catches_invalid_rank_permutation():
    rows = [("1", [1, 1])]  # not a valid permutation of 1..2
    with pytest.raises(AssertionError):
        validate_submission(rows, expected_ids_in_order=["1"])


def test_zip_submission_has_no_parent_folder_or_macosx(tmp_path):
    txt_path = tmp_path / "predictions.txt"
    write_submission_file([("1", [1, 2]), ("2", [2, 1])], txt_path)

    zip_path = tmp_path / "out.zip"
    zip_submission(txt_path, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert names == ["predictions.txt"], f"zip must contain exactly one root-level file, got {names}"
