"""GitHub update checks for wakeword-forge checkouts."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from wakeword_forge import __version__ as LOCAL_VERSION

DEFAULT_GITHUB_REPO = "H-Ali13381/wakeword-forge"
DEFAULT_BRANCH = "main"
GITHUB_API_ROOT = "https://api.github.com"
GITHUB_RAW_ROOT = "https://raw.githubusercontent.com"
GITHUB_REPO_URL = f"https://github.com/{DEFAULT_GITHUB_REPO}"
VERSION_FILE_PATH = "wakeword_forge/__init__.py"
DEFAULT_UPDATE_COMMAND = f"git pull --ff-only origin {DEFAULT_BRANCH}"
PYPI_UPDATE_COMMAND = (
    f'pip install --upgrade "wakeword-forge[ui,tts] @ git+https://github.com/{DEFAULT_GITHUB_REPO}.git"'
)

UpdateStatus = Literal["current", "update_available", "unknown"]
GitRunner = Callable[[Sequence[str], Path], str | None]
CompareFetcher = Callable[[str, str, str, float], Mapping[str, Any]]
VersionFetcher = Callable[[str, str, float], str | None]


@dataclass(frozen=True)
class UpdateRecommendation:
    """User-facing result from comparing this checkout to GitHub."""

    status: UpdateStatus
    message: str
    update_command: str
    repo_url: str
    local_ref: str | None = None
    remote_ref: str | None = None
    remote_ahead_by: int = 0
    local_ahead_by: int = 0
    detail_url: str | None = None
    checked: bool = False

    @property
    def needs_update(self) -> bool:
        return self.status == "update_available"


def _plural(count: int, noun: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix}"


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _compare_url(repo: str, local_sha: str, branch: str) -> str:
    return f"https://github.com/{repo}/compare/{local_sha}...{branch}"


def _update_command(branch: str) -> str:
    return f"git pull --ff-only origin {branch}"


def _latest_commit_sha(payload: Mapping[str, Any]) -> str | None:
    commits = payload.get("commits")
    if isinstance(commits, list) and commits:
        last_commit = commits[-1]
        if isinstance(last_commit, Mapping):
            sha = last_commit.get("sha")
            if isinstance(sha, str) and sha:
                return sha
    return None


def recommendation_from_compare_payload(
    payload: Mapping[str, Any],
    *,
    local_sha: str,
    repo: str = DEFAULT_GITHUB_REPO,
    branch: str = DEFAULT_BRANCH,
) -> UpdateRecommendation:
    """Translate a GitHub compare API payload into an update recommendation.

    The compare request is shaped as ``local_sha...branch``. In that direction,
    ``ahead_by`` means the GitHub branch contains commits missing locally.
    """

    remote_ahead_by = _as_int(payload.get("ahead_by"))
    local_ahead_by = _as_int(payload.get("behind_by"))
    detail_url = str(payload.get("html_url") or _compare_url(repo, local_sha, branch))
    command = _update_command(branch)
    repo_url = f"https://github.com/{repo}"
    remote_ref = _latest_commit_sha(payload)

    if remote_ahead_by > 0:
        message = (
            f"Update available: GitHub {branch} is {_plural(remote_ahead_by, 'commit')} "
            "ahead of this checkout."
        )
        if local_ahead_by > 0:
            message += (
                f" This checkout also has {_plural(local_ahead_by, 'local commit')}; "
                "pull carefully or rebase."
            )
        return UpdateRecommendation(
            status="update_available",
            message=message,
            update_command=command,
            repo_url=repo_url,
            local_ref=local_sha,
            remote_ref=remote_ref,
            remote_ahead_by=remote_ahead_by,
            local_ahead_by=local_ahead_by,
            detail_url=detail_url,
            checked=True,
        )

    if local_ahead_by > 0:
        message = (
            f"No GitHub update is needed. This checkout has {_plural(local_ahead_by, 'local commit')} "
            f"not on GitHub {branch}."
        )
    else:
        message = f"wakeword-forge is up to date with GitHub {branch}."

    return UpdateRecommendation(
        status="current",
        message=message,
        update_command=command,
        repo_url=repo_url,
        local_ref=local_sha,
        remote_ref=remote_ref or local_sha,
        remote_ahead_by=remote_ahead_by,
        local_ahead_by=local_ahead_by,
        detail_url=detail_url,
        checked=True,
    )


def parse_version_from_text(text: str) -> str | None:
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else None


def _version_parts(version: str) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", version)]
    return tuple(parts or [0])


def _is_remote_version_newer(local_version: str, remote_version: str) -> bool:
    return _version_parts(remote_version) > _version_parts(local_version)


def fetch_github_version(
    repo: str = DEFAULT_GITHUB_REPO,
    branch: str = DEFAULT_BRANCH,
    timeout_seconds: float = 2.0,
) -> str | None:
    raw_url = f"{GITHUB_RAW_ROOT}/{repo}/{branch}/{VERSION_FILE_PATH}"
    request = urllib.request.Request(
        raw_url,
        headers={"User-Agent": "wakeword-forge-update-check"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return parse_version_from_text(response.read().decode("utf-8"))


def recommendation_from_versions(
    *,
    local_version: str,
    remote_version: str,
    repo: str = DEFAULT_GITHUB_REPO,
    branch: str = DEFAULT_BRANCH,
) -> UpdateRecommendation:
    repo_url = f"https://github.com/{repo}"
    if _is_remote_version_newer(local_version, remote_version):
        return UpdateRecommendation(
            status="update_available",
            message=(
                f"Update available: GitHub {branch} reports version {remote_version}; "
                f"this install is {local_version}."
            ),
            update_command=PYPI_UPDATE_COMMAND.replace(DEFAULT_GITHUB_REPO, repo),
            repo_url=repo_url,
            local_ref=local_version,
            remote_ref=remote_version,
            remote_ahead_by=1,
            detail_url=repo_url,
            checked=True,
        )

    return UpdateRecommendation(
        status="current",
        message=f"wakeword-forge version {local_version} matches GitHub {branch}.",
        update_command=PYPI_UPDATE_COMMAND.replace(DEFAULT_GITHUB_REPO, repo),
        repo_url=repo_url,
        local_ref=local_version,
        remote_ref=remote_version,
        detail_url=repo_url,
        checked=True,
    )


def _run_git(args: Sequence[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    return output or None


def _repository_root(repository_path: Path | str | None, git_runner: GitRunner) -> Path | None:
    start = Path(repository_path) if repository_path is not None else Path(__file__).resolve()
    if start.is_file():
        start = start.parent
    root = git_runner(["rev-parse", "--show-toplevel"], start)
    return Path(root) if root else None


def _local_git_sha(repository_path: Path | str | None, git_runner: GitRunner) -> str | None:
    root = _repository_root(repository_path, git_runner)
    if root is None:
        return None
    return git_runner(["rev-parse", "HEAD"], root)


def fetch_github_compare(
    local_sha: str,
    repo: str = DEFAULT_GITHUB_REPO,
    branch: str = DEFAULT_BRANCH,
    timeout_seconds: float = 2.0,
) -> Mapping[str, Any]:
    """Fetch GitHub's compare payload for ``local_sha...branch``."""

    api_url = f"{GITHUB_API_ROOT}/repos/{repo}/compare/{local_sha}...{branch}"
    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "wakeword-forge-update-check",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _unknown_recommendation(message: str, *, repo: str, branch: str) -> UpdateRecommendation:
    return UpdateRecommendation(
        status="unknown",
        message=message,
        update_command=_update_command(branch),
        repo_url=f"https://github.com/{repo}",
        detail_url=f"https://github.com/{repo}",
    )


def check_for_updates(
    *,
    repository_path: Path | str | None = None,
    repo: str = DEFAULT_GITHUB_REPO,
    branch: str = DEFAULT_BRANCH,
    timeout_seconds: float = 2.0,
    git_runner: GitRunner | None = None,
    compare_fetcher: CompareFetcher | None = None,
    local_version: str = LOCAL_VERSION,
    version_fetcher: VersionFetcher | None = None,
) -> UpdateRecommendation:
    """Compare this checkout with GitHub and return a safe user recommendation.

    The function never raises for missing git, offline users, rate limits, or
    GitHub API errors. The dashboard can call it opportunistically on startup.
    """

    runner = git_runner or _run_git
    local_sha = _local_git_sha(repository_path, runner)
    if local_sha is None:
        fetch_version = version_fetcher or fetch_github_version
        try:
            remote_version = fetch_version(repo, branch, timeout_seconds)
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            remote_version = None
        if remote_version:
            return recommendation_from_versions(
                local_version=local_version,
                remote_version=remote_version,
                repo=repo,
                branch=branch,
            )
        return _unknown_recommendation(
            "Update check needs a Git checkout; compare manually with the GitHub repo.",
            repo=repo,
            branch=branch,
        )

    fetcher = compare_fetcher or fetch_github_compare
    try:
        payload = fetcher(local_sha, repo, branch, timeout_seconds)
    except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        return _unknown_recommendation(
            f"Could not check GitHub for updates: {exc}",
            repo=repo,
            branch=branch,
        )

    return recommendation_from_compare_payload(payload, local_sha=local_sha, repo=repo, branch=branch)
