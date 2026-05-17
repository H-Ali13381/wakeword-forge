from __future__ import annotations

from wakeword_forge.update_check import UpdateRecommendation, check_for_updates, recommendation_from_compare_payload


def test_compare_payload_recommends_update_when_github_branch_has_new_commits():
    recommendation = recommendation_from_compare_payload(
        {
            "ahead_by": 3,
            "behind_by": 0,
            "html_url": "https://github.com/H-Ali13381/wakeword-forge/compare/old...main",
            "commits": [{"sha": "newest-sha"}],
        },
        local_sha="old-sha",
        branch="main",
    )

    assert recommendation.needs_update is True
    assert recommendation.status == "update_available"
    assert recommendation.remote_ahead_by == 3
    assert recommendation.local_ahead_by == 0
    assert recommendation.remote_ref == "newest-sha"
    assert "3 commits ahead" in recommendation.message
    assert "git pull --ff-only origin main" == recommendation.update_command
    assert recommendation.detail_url.endswith("old...main")


def test_compare_payload_reports_current_when_remote_has_no_new_commits():
    recommendation = recommendation_from_compare_payload(
        {"ahead_by": 0, "behind_by": 0, "commits": []},
        local_sha="same-sha",
        branch="main",
    )

    assert recommendation.needs_update is False
    assert recommendation.status == "current"
    assert recommendation.remote_ahead_by == 0
    assert "up to date" in recommendation.message.lower()


def test_compare_payload_warns_carefully_when_checkout_has_local_commits_too():
    recommendation = recommendation_from_compare_payload(
        {"ahead_by": 2, "behind_by": 1, "commits": [{"sha": "remote-sha"}]},
        local_sha="local-sha",
        branch="main",
    )

    assert recommendation.needs_update is True
    assert recommendation.status == "update_available"
    assert recommendation.remote_ahead_by == 2
    assert recommendation.local_ahead_by == 1
    assert "also has 1 local commit" in recommendation.message
    assert "pull carefully" in recommendation.message


def test_check_for_updates_reports_unknown_without_a_git_checkout(tmp_path):
    recommendation = check_for_updates(
        repository_path=tmp_path,
        version_fetcher=lambda _repo, _branch, _timeout: None,
    )

    assert isinstance(recommendation, UpdateRecommendation)
    assert recommendation.needs_update is False
    assert recommendation.status == "unknown"
    assert "Git checkout" in recommendation.message


def test_check_for_updates_falls_back_to_github_version_without_a_git_checkout(tmp_path):
    recommendation = check_for_updates(
        repository_path=tmp_path,
        local_version="0.1.0",
        version_fetcher=lambda _repo, _branch, _timeout: "0.2.0",
    )

    assert recommendation.needs_update is True
    assert recommendation.status == "update_available"
    assert "0.2.0" in recommendation.message
    assert "0.1.0" in recommendation.message
    assert "pip install --upgrade" in recommendation.update_command


def test_check_for_updates_stays_current_when_non_git_version_matches_github(tmp_path):
    recommendation = check_for_updates(
        repository_path=tmp_path,
        local_version="0.2.0",
        version_fetcher=lambda _repo, _branch, _timeout: "0.2.0",
    )

    assert recommendation.needs_update is False
    assert recommendation.status == "current"
    assert "version 0.2.0" in recommendation.message
