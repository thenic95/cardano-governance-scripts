#!/usr/bin/env python3
"""Analyze Cardano governance voting participation using Blockfrost.

This script fetches all governance actions and their votes from the Blockfrost
API, then computes participation metrics for DReps, the Constitutional
Committee (CC) and Stake Pool Operators (SPOs). It prints a compact summary
table along with simple trend analytics, and can optionally export the raw
results to CSV or JSON for downstream analysis.

Prerequisites:
  * Install dependencies: `pip install requests`
  * Export your Blockfrost project id:
        export BLOCKFROST_PROJECT_ID=mainnetXXXXXXXXXXXX
    (or pass it with --project-id)

Blockfrost API coverage for CIP-1694 governance is still evolving. The script
attempts to handle minor schema variations gracefully, but you may need to
update the normalisation helpers below if Blockfrost adjusts field names.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

DEFAULT_BASE_URL = "https://cardano-mainnet.blockfrost.io/api/v0"
ENV_PROJECT_IDS = ("BLOCKFROST_PROJECT_ID", "BLOCKFROST_API_KEY", "BLOCKFROST_PROJECT_ID_MAINNET")
DEFAULT_PAGE_SIZE = 100
DEFAULT_SLEEP_SECONDS = 0.2  # Play nicely with rate limits.
GOVERNANCE_ACTION_ENDPOINT_CANDIDATES = (
    "/governance/actions",
    "/governance/action_proposals",
    "/governance/action-proposals",
    "/governance/governance-actions",
    "/governance/proposals",
    "/gov/actions",
    "/gov/action_proposals",
    "/gov/action-proposals",
    "/gov/governance-actions",
)
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DREP_CACHE_PATH = CACHE_DIR / "drep_records.json"

RoleKey = str


class BlockfrostAPIError(RuntimeError):
    """Raised when Blockfrost responds with an error."""


@dataclass
class RoleParticipation:
    """Participation metrics captured for a single constituency."""

    voters: int = 0
    total_votes: int = 0
    votes_by_choice: Dict[str, int] = field(default_factory=dict)
    voting_power: int = 0
    eligible_voters: Optional[int] = None
    participation_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        non_voters = None
        if self.eligible_voters is not None:
            non_voters = max(self.eligible_voters - self.voters, 0)
        return {
            "unique_voters": self.voters,
            "total_votes": self.total_votes,
            "votes_by_choice": self.votes_by_choice,
            "voting_power_lovelace": self.voting_power,
            "participation_ratio": self.participation_ratio,
            "eligible_voters": self.eligible_voters,
            "non_voters": non_voters,
        }


@dataclass
class DrepRecord:
    """Minimal registration metadata for a DRep used to compute baselines."""

    identifier: str
    active_epoch: Optional[int]
    last_active_epoch: Optional[int]
    expired: bool
    retired: bool


@dataclass
class ActionParticipation:
    """Participation metrics tied to a specific governance action."""

    identifier: str
    index: Optional[int]
    action_type: str
    title: str
    status: Optional[str]
    created_epoch: Optional[int]
    voting_start_epoch: Optional[int]
    voting_end_epoch: Optional[int]
    enacted_epoch: Optional[int]
    anchor_hash: Optional[str]
    tallies: Dict[RoleKey, RoleParticipation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_identifier": self.identifier,
            "action_index": self.index,
            "action_type": self.action_type,
            "title": self.title,
            "status": self.status,
            "created_epoch": self.created_epoch,
            "voting_start_epoch": self.voting_start_epoch,
            "voting_end_epoch": self.voting_end_epoch,
            "enacted_epoch": self.enacted_epoch,
            "anchor_hash": self.anchor_hash,
            "tallies": {role: tally.to_dict() for role, tally in self.tallies.items()},
        }


class BlockfrostClient:
    """Small helper for interacting with the Blockfrost REST API."""

    def __init__(
        self,
        project_id: str,
        base_url: str = DEFAULT_BASE_URL,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        timeout: int = 30,
        max_retries: int = 5,
    ) -> None:
        if not project_id:
            raise ValueError("A Blockfrost project id is required.")

        self._base_url = base_url.rstrip("/")
        self._sleep_seconds = sleep_seconds
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"project_id": project_id})

    def close(self) -> None:
        self._session.close()

    def get(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        """Perform a GET request with retry/backoff for rate limits."""
        url = f"{self._base_url}{path}"
        params = params or {}
        for attempt in range(1, self._max_retries + 1):
            response = self._session.get(url, params=params, timeout=self._timeout)
            if response.status_code == 429:
                wait_for = self._retry_after_seconds(response, fallback=attempt)
                logging.debug("Rate limited on %s (429). Sleeping for %.2fs", path, wait_for)
                time.sleep(wait_for)
                continue

            if response.status_code >= 400:
                message = self._format_error_message(response)
                raise BlockfrostAPIError(message)

            if self._sleep_seconds:
                time.sleep(self._sleep_seconds)
            return response.json()

        raise BlockfrostAPIError(f"Failed to fetch {path} after {self._max_retries} attempts.")

    def paginate(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: Optional[int] = None,
    ) -> Iterable[Any]:
        params = params.copy() if params else {}
        page = 1
        while True:
            params_with_paging = dict(params)
            params_with_paging.setdefault("count", page_size)
            params_with_paging["page"] = page
            logging.debug("Fetching %s page %d params=%s", path, page, params_with_paging)
            page_items = self.get(path, params=params_with_paging)
            if isinstance(page_items, dict):
                # Some endpoints respond with dicts; yield the dict once.
                if page == 1:
                    yield page_items
                break

            if not page_items:
                break

            for item in page_items:
                yield item

            if len(page_items) < page_size:
                break

            page += 1
            if max_pages and page > max_pages:
                break

    @staticmethod
    def _retry_after_seconds(response: requests.Response, fallback: int) -> float:
        raw = response.headers.get("Retry-After")
        if raw is None:
            return min(5.0, 0.3 * (fallback + 1))
        try:
            return float(raw)
        except ValueError:
            return min(5.0, 0.3 * (fallback + 1))

    @staticmethod
    def _format_error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        return f"Blockfrost error {response.status_code}: {payload}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze voting participation for Cardano governance actions via Blockfrost.",
    )
    parser.add_argument(
        "--project-id",
        help="Blockfrost project id (falls back to BLOCKFROST_PROJECT_ID env variable).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Blockfrost base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--min-epoch",
        type=int,
        help="Only consider governance actions with a voting_start_epoch >= this epoch.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        help="Process at most this many actions (useful for exploratory runs).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Delay between requests to stay within rate limits. Default: %(default)s",
    )
    parser.add_argument(
        "--actions-endpoint",
        help=(
            "Override the governance actions endpoint path relative to the API base URL. "
            "Examples: /governance/actions, /governance/action_proposals."
        ),
    )
    parser.add_argument(
        "--drep-baseline-count",
        type=int,
        help="Override the number of eligible DReps. Otherwise the script attempts to query Blockfrost.",
    )
    parser.add_argument(
        "--cc-baseline-count",
        type=int,
        help="Override the number of active Constitutional Committee members.",
    )
    parser.add_argument(
        "--spo-baseline-count",
        type=int,
        help="Override the number of active stake pools. Auto-discovery may require many requests.",
    )
    parser.add_argument(
        "--disable-auto-baselines",
        action="store_true",
        help="Skip the auto-discovery of baseline counts entirely (requires manual overrides).",
    )
    parser.add_argument(
        "--output-json",
        help="Write the full per-action metrics as JSON to this path.",
    )
    parser.add_argument(
        "--output-csv",
        help="Write the per-action metrics to a CSV file at this path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def obtain_project_id(cli_override: Optional[str]) -> str:
    if cli_override:
        return cli_override
    for env_name in ENV_PROJECT_IDS:
        value = os.getenv(env_name)
        if value:
            return value
    raise ValueError(
        "Blockfrost project id not provided. Use --project-id or set BLOCKFROST_PROJECT_ID."
    )


def fetch_governance_actions(
    client: BlockfrostClient,
    *,
    min_epoch: Optional[int] = None,
    max_actions: Optional[int] = None,
    endpoint: str,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for item in client.paginate(endpoint, params={"order": "asc"}):
        if not isinstance(item, dict):
            logging.debug("Skipping unexpected governance action payload: %s", item)
            continue
        action_epoch = parse_int(item.get("voting_start_epoch")) or parse_int(item.get("created_epoch"))
        if min_epoch is not None and action_epoch is not None and action_epoch < min_epoch:
            continue
        actions.append(item)
        if max_actions and len(actions) >= max_actions:
            break
    logging.info("Fetched %d governance actions", len(actions))
    return actions


def fetch_votes_for_action(client: BlockfrostClient, actions_endpoint: str, action_identifier: str) -> List[Dict[str, Any]]:
    votes: List[Dict[str, Any]] = []
    path = f"{actions_endpoint.rstrip('/')}/{action_identifier}/votes"
    for vote in client.paginate(path, params={"order": "asc"}):
        if isinstance(vote, dict):
            votes.append(vote)
        else:
            logging.debug("Discarding unexpected vote payload for action %s: %s", action_identifier, vote)
    logging.debug("Fetched %d votes for action %s", len(votes), action_identifier)
    return votes


def fetch_active_dreps(client: BlockfrostClient) -> Set[str]:
    candidates: Set[str] = set()
    for params in ({"status": "active"}, {"status": "registered"}, {}):
        candidates.clear()
        try:
            for item in client.paginate("/governance/dreps", params=params):
                if not isinstance(item, dict):
                    continue
                status = (item.get("status") or "").lower()
                if status in {"deregistered", "retired"}:
                    continue
                drep_id = item.get("drep_id") or item.get("hash") or item.get("view")
                if drep_id:
                    candidates.add(drep_id)
        except BlockfrostAPIError as exc:
            logging.debug("Unable to fetch DReps with params %s: %s", params, exc)
            continue
        if candidates:
            break
    logging.info("Auto-discovered %d DReps", len(candidates))
    return set(candidates)


def fetch_committee_members(client: BlockfrostClient) -> Set[str]:
    """Attempt to extract the set of active constitutional committee members."""
    for path in ("/governance/committees/current", "/governance/committees"):
        try:
            payload = client.get(path)
        except BlockfrostAPIError as exc:
            logging.debug("Unable to query %s: %s", path, exc)
            continue

        members = extract_committee_members(payload)
        if members:
            logging.info("Auto-discovered %d CC members using %s", len(members), path)
            return members

    logging.warning("Could not automatically discover constitutional committee members.")
    return set()


def extract_committee_members(payload: Any) -> Set[str]:
    members: Set[str] = set()
    if isinstance(payload, dict):
        candidates = []
        for key in ("members", "committee_members", "active_members"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        if not candidates and "items" in payload and isinstance(payload["items"], list):
            candidates.extend(payload["items"])
        for member in candidates:
            member_id = normalise_committee_member(member)
            if member_id:
                members.add(member_id)
        return members

    if isinstance(payload, list):
        for entry in payload:
            members.update(extract_committee_members(entry))
    return members


def normalise_committee_member(member_payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(member_payload, dict):
        return None
    for key in (
        "credential",
        "hot_cred",
        "cold_cred",
        "key_hash",
        "cold_key_hash",
        "hot_key_hash",
        "bech32",
        "address",
    ):
        value = member_payload.get(key)
        if value:
            return str(value)
    return None


def fetch_total_pool_count(client: BlockfrostClient) -> int:
    count = 0
    try:
        for _pool in client.paginate("/pools", params={"order": "asc"}):
            count += 1
    except BlockfrostAPIError as exc:
        logging.warning("Unable to enumerate stake pools: %s", exc)
        return 0
    logging.info("Auto-discovered %d stake pools", count)
    return count


def summarise_votes(
    votes: Sequence[Dict[str, Any]],
    baseline_counts: Dict[RoleKey, Optional[int]],
) -> Dict[RoleKey, RoleParticipation]:
    aggregators: Dict[RoleKey, _RoleAccumulator] = {}
    for vote in votes:
        if not isinstance(vote, dict):
            continue
        role = normalise_role(vote.get("voter_type") or vote.get("voter_role"))
        actor = extract_actor_id(vote)
        choice = normalise_choice(vote.get("vote"))
        power = parse_int(vote.get("voting_power"))
        aggregator = aggregators.setdefault(role, _RoleAccumulator())
        aggregator.add_vote(actor, choice, power)

    tallies: Dict[RoleKey, RoleParticipation] = {}
    for role, aggregator in aggregators.items():
        baseline = baseline_counts.get(role)
        eligible = None
        participation_ratio = None
        if baseline is not None:
            eligible = max(baseline, 0)
            if eligible > 0:
                participation_ratio = aggregator.unique_voter_count / eligible
            else:
                participation_ratio = None
        if participation_ratio == 0:
            participation_ratio = 0.0
        tallies[role] = RoleParticipation(
            voters=aggregator.unique_voter_count,
            total_votes=aggregator.vote_count,
            votes_by_choice=dict(aggregator.votes_by_choice),
            voting_power=aggregator.total_voting_power,
            eligible_voters=eligible,
            participation_ratio=participation_ratio,
        )

    # Ensure all core roles are present even if no votes were cast.
    for role in ("drep", "cc", "spo"):
        if role not in tallies:
            baseline = baseline_counts.get(role)
            eligible = None
            participation_ratio = None
            if baseline is not None:
                eligible = max(baseline, 0)
                if eligible > 0:
                    participation_ratio = 0.0
            tallies[role] = RoleParticipation(
                voters=0,
                total_votes=0,
                votes_by_choice={},
                voting_power=0,
                eligible_voters=eligible,
                participation_ratio=participation_ratio,
            )

    return tallies


_TX_EPOCH_CACHE: Dict[str, Optional[int]] = {}
_BLOCK_EPOCH_CACHE: Dict[str, Optional[int]] = {}


def fetch_tx_epoch(client: BlockfrostClient, tx_hash: Optional[str]) -> Optional[int]:
    if not tx_hash:
        return None
    if tx_hash in _TX_EPOCH_CACHE:
        return _TX_EPOCH_CACHE[tx_hash]
    try:
        tx = client.get(f"/txs/{tx_hash}")
    except BlockfrostAPIError as exc:
        logging.debug("Unable to fetch tx %s: %s", tx_hash, exc)
        _TX_EPOCH_CACHE[tx_hash] = None
        return None

    block_hash = tx.get("block")
    if not block_hash:
        _TX_EPOCH_CACHE[tx_hash] = None
        return None

    epoch = fetch_block_epoch(client, block_hash)
    _TX_EPOCH_CACHE[tx_hash] = epoch
    return epoch


def fetch_block_epoch(client: BlockfrostClient, block_hash: Optional[str]) -> Optional[int]:
    if not block_hash:
        return None
    if block_hash in _BLOCK_EPOCH_CACHE:
        return _BLOCK_EPOCH_CACHE[block_hash]
    try:
        block = client.get(f"/blocks/{block_hash}")
    except BlockfrostAPIError as exc:
        logging.debug("Unable to fetch block %s: %s", block_hash, exc)
        _BLOCK_EPOCH_CACHE[block_hash] = None
        return None
    epoch = block.get("epoch")
    if epoch is None:
        _BLOCK_EPOCH_CACHE[block_hash] = None
        return None
    try:
        parsed = int(epoch)
    except (TypeError, ValueError):
        parsed = None
    _BLOCK_EPOCH_CACHE[block_hash] = parsed
    return parsed


def fetch_action_detail(client: BlockfrostClient, actions_endpoint: str, identifier: str) -> Dict[str, Any]:
    try:
        return client.get(f"{actions_endpoint.rstrip('/')}/{identifier}")
    except BlockfrostAPIError as exc:
        logging.debug("Unable to fetch action details for %s: %s", identifier, exc)
        return {}


def fetch_action_metadata(client: BlockfrostClient, actions_endpoint: str, identifier: str) -> Dict[str, Any]:
    try:
        return client.get(f"{actions_endpoint.rstrip('/')}/{identifier}/metadata")
    except BlockfrostAPIError as exc:
        logging.debug("Unable to fetch action metadata for %s: %s", identifier, exc)
        return {}


def fetch_drep_records(client: BlockfrostClient) -> List[DrepRecord]:
    cached = load_cached_drep_records()
    if cached is not None:
        logging.info("Loaded %d cached DRep records", len(cached))
        return cached

    records: List[DrepRecord] = []
    seen: Set[str] = set()
    for params in ({"status": "active"}, {"status": "registered"}, {}):
        for item in client.paginate("/governance/dreps", params=params):
            if not isinstance(item, dict):
                continue
            identifier = item.get("drep_id") or item.get("hash") or item.get("view")
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            detail = fetch_drep_detail(client, identifier)
            records.append(detail)
    logging.info("Collected detail for %d DReps", len(records))
    persist_drep_records(records)
    return records


def fetch_drep_detail(client: BlockfrostClient, identifier: str) -> DrepRecord:
    try:
        payload = client.get(f"/governance/dreps/{identifier}")
    except BlockfrostAPIError as exc:
        logging.debug("Unable to fetch detail for DRep %s: %s", identifier, exc)
        return DrepRecord(identifier=identifier, active_epoch=None, last_active_epoch=None, expired=False, retired=False)
    active_epoch = parse_optional_int(payload.get("active_epoch"))
    last_active_epoch = parse_optional_int(payload.get("last_active_epoch"))
    return DrepRecord(
        identifier=identifier,
        active_epoch=active_epoch,
        last_active_epoch=last_active_epoch,
        expired=bool(payload.get("expired")),
        retired=bool(payload.get("retired")),
    )


def count_active_dreps_at_epoch(records: Sequence[DrepRecord], epoch: int) -> int:
    count = 0
    for record in records:
        if record.active_epoch is None or record.active_epoch > epoch:
            continue
        if record.last_active_epoch is not None and record.last_active_epoch < epoch:
            continue
        if record.expired and (record.last_active_epoch is None or record.last_active_epoch < epoch):
            continue
        count += 1
    return count


def load_cached_drep_records() -> Optional[List[DrepRecord]]:
    if not DREP_CACHE_PATH.exists():
        return None
    try:
        with open(DREP_CACHE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        logging.debug("Failed to read cached DRep records: %s", exc)
        return None
    records: List[DrepRecord] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        records.append(
            DrepRecord(
                identifier=item.get("identifier", ""),
                active_epoch=parse_optional_int(item.get("active_epoch")),
                last_active_epoch=parse_optional_int(item.get("last_active_epoch")),
                expired=bool(item.get("expired")),
                retired=bool(item.get("retired")),
            )
        )
    return records


def persist_drep_records(records: Sequence[DrepRecord]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(DREP_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "identifier": record.identifier,
                        "active_epoch": record.active_epoch,
                        "last_active_epoch": record.last_active_epoch,
                        "expired": record.expired,
                        "retired": record.retired,
                    }
                    for record in records
                ],
                handle,
                indent=2,
                sort_keys=True,
            )
    except OSError as exc:
        logging.debug("Unable to persist DRep records cache: %s", exc)


class BaselineResolver:
    """Provides per-epoch electorate sizes for each constituency."""

    def __init__(
        self,
        overrides: Dict[RoleKey, Optional[int]],
        *,
        drep_records: Optional[Sequence[DrepRecord]] = None,
        default_counts: Optional[Dict[RoleKey, Optional[int]]] = None,
    ) -> None:
        self._overrides = overrides
        self._drep_records = list(drep_records) if drep_records else None
        self._defaults = default_counts.copy() if default_counts else {}

    def baseline_for(self, role: RoleKey, epoch: Optional[int]) -> Optional[int]:
        override = self._overrides.get(role)
        if override is not None:
            return override

        if role == "drep" and self._drep_records and epoch is not None:
            return count_active_dreps_at_epoch(self._drep_records, epoch)

        return self._defaults.get(role)

    def baseline_counts_for_epoch(self, epoch: Optional[int]) -> Dict[RoleKey, Optional[int]]:
        result: Dict[RoleKey, Optional[int]] = {}
        for role in ("drep", "cc", "spo"):
            result[role] = self.baseline_for(role, epoch)
        return result


class _RoleAccumulator:
    """Internal helper to collect per-role vote data."""

    def __init__(self) -> None:
        self._unique_voters: Set[str] = set()
        self.vote_count: int = 0
        self.votes_by_choice: Dict[str, int] = {}
        self.total_voting_power: int = 0

    def add_vote(self, actor: str, choice: Optional[str], power: int) -> None:
        if actor:
            self._unique_voters.add(actor)
        self.vote_count += 1
        if choice:
            self.votes_by_choice[choice] = self.votes_by_choice.get(choice, 0) + 1
        self.total_voting_power += power

    @property
    def unique_voter_count(self) -> int:
        return len(self._unique_voters)


def normalise_role(raw_role: Optional[str]) -> RoleKey:
    if not raw_role:
        return "unknown"
    normalised = raw_role.strip().lower()
    aliases = {
        "stake_pool": "spo",
        "stakepool": "spo",
        "stake_pool_operator": "spo",
        "stake_pool_voting": "spo",
        "spo": "spo",
        "pool": "spo",
        "constitutional_committee": "cc",
        "constitutional_committee_hot": "cc",
        "constitutional_committee_cold": "cc",
        "cc_hot": "cc",
        "cc_cold": "cc",
        "committee": "cc",
        "committee_hot": "cc",
        "committee_cold": "cc",
        "committee_member": "cc",
        "drep": "drep",
        "drep_voter": "drep",
    }
    return aliases.get(normalised, normalised)


def normalise_choice(choice: Optional[str]) -> Optional[str]:
    if not choice:
        return None
    choice = choice.strip().lower()
    aliases = {
        "abstain": "abstain",
        "no": "no",
        "nay": "no",
        "against": "no",
        "yes": "yes",
        "yea": "yes",
        "support": "yes",
        "no_confidence": "no_confidence",
        "noconfidence": "no_confidence",
    }
    return aliases.get(choice, choice)


def extract_actor_id(vote: Dict[str, Any]) -> str:
    for key in (
        "voter_hash",
        "voter",
        "delegator",
        "credential",
        "pool_id",
        "drep_id",
        "stake_credential",
    ):
        value = vote.get(key)
        if value:
            return str(value)
    # Fall back to the transaction hash (unique per vote) to keep counts consistent.
    tx_hash = vote.get("tx_hash")
    if tx_hash:
        return str(tx_hash)
    return ""


def parse_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def extract_action_identifier(action: Dict[str, Any]) -> Optional[str]:
    for key in (
        "id",
        "action_id",
        "hash",
        "proposal_id",
        "governance_action_id",
        "gov_action_id",
    ):
        value = action.get(key)
        if value:
            return str(value)
    # As a last resort fall back to the numeric index.
    index_value = action.get("action_index") or action.get("index") or action.get("cert_index")
    if index_value is not None:
        return str(index_value)
    return None


def build_action_participation(
    *,
    identifier: str,
    index: Optional[int],
    action_type: str,
    title: str,
    status: Optional[str],
    created_epoch: Optional[int],
    voting_start_epoch: Optional[int],
    voting_end_epoch: Optional[int],
    enacted_epoch: Optional[int],
    anchor_hash: Optional[str],
    tallies: Dict[RoleKey, RoleParticipation],
) -> ActionParticipation:
    return ActionParticipation(
        identifier=identifier,
        index=index,
        action_type=action_type,
        title=title,
        status=status,
        created_epoch=created_epoch,
        voting_start_epoch=voting_start_epoch,
        voting_end_epoch=voting_end_epoch,
        enacted_epoch=enacted_epoch,
        anchor_hash=anchor_hash,
        tallies=tallies,
    )


def derive_status(detail: Dict[str, Any]) -> Optional[str]:
    status = detail.get("status")
    if status:
        return str(status)

    enacted_epoch = parse_optional_int(detail.get("enacted_epoch"))
    dropped_epoch = parse_optional_int(detail.get("dropped_epoch"))
    expired_epoch = parse_optional_int(detail.get("expired_epoch"))
    ratified_epoch = parse_optional_int(detail.get("ratified_epoch"))

    if enacted_epoch is not None:
        return "enacted"
    if dropped_epoch is not None:
        return "dropped"
    if expired_epoch is not None:
        return "expired"
    if ratified_epoch is not None:
        return "ratified"
    return None


def extract_title_from_sources(
    raw_action: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    candidates: List[Any] = [
        raw_action.get("title"),
        raw_action.get("description"),
    ]

    json_metadata = metadata.get("json_metadata")
    if isinstance(json_metadata, dict):
        candidates.extend(
            [
                json_metadata.get("title"),
                json_metadata.get("name"),
            ]
        )
        body = json_metadata.get("body")
        if isinstance(body, dict):
            candidates.extend(
                [
                    body.get("title"),
                    body.get("name"),
                    body.get("headline"),
                ]
            )

    candidates.append(metadata.get("title"))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def determine_electorate_epoch(
    detail: Dict[str, Any],
    *,
    created_epoch: Optional[int],
    voting_start_epoch: Optional[int],
    voting_end_epoch: Optional[int],
    first_vote_epoch: Optional[int],
    last_vote_epoch: Optional[int],
) -> Optional[int]:
    for key in ("enacted_epoch", "dropped_epoch", "expired_epoch", "expiration"):
        value = parse_optional_int(detail.get(key))
        if value is not None:
            return value

    if voting_end_epoch is not None:
        return voting_end_epoch

    fallback_keys = ("voting_end_epoch", "voting_start_epoch", "created_epoch")
    for key in fallback_keys:
        value = parse_optional_int(detail.get(key))
        if value is not None:
            return value

    for epoch in (last_vote_epoch, first_vote_epoch, voting_start_epoch, created_epoch):
        if epoch is not None:
            return epoch
    return None


def summarise_trends(actions: Sequence[ActionParticipation]) -> Dict[str, Dict[str, float]]:
    trends: Dict[str, Dict[str, float]] = {}
    for role in ("drep", "cc", "spo"):
        ratios = [
            action.tallies[role].participation_ratio
            for action in actions
            if action.tallies[role].participation_ratio is not None
        ]
        if not ratios:
            continue

        first_window = ratios[: min(5, len(ratios))]
        last_window = ratios[-min(5, len(ratios)) :]
        trends[role] = {
            "avg_all": statistics.mean(ratios),
            "avg_first": statistics.mean(first_window),
            "avg_last": statistics.mean(last_window),
            "delta": statistics.mean(last_window) - statistics.mean(first_window),
        }
    return trends


def print_summary_table(
    actions: Sequence[ActionParticipation],
) -> None:
    headers = [
        "id",
        "type",
        "vote_epoch",
        "drep voters",
        "drep %",
        "cc voters",
        "cc %",
        "spo voters",
        "spo %",
    ]
    print("\t".join(headers))
    for action in actions:
        vote_epoch = action.voting_start_epoch or action.created_epoch or "-"
        drep = action.tallies["drep"]
        cc = action.tallies["cc"]
        spo = action.tallies["spo"]
        row = [
            format_action_label(action),
            action.action_type,
            str(vote_epoch),
            format_count_with_baseline(drep),
            format_ratio(drep.participation_ratio),
            format_count_with_baseline(cc),
            format_ratio(cc.participation_ratio),
            format_count_with_baseline(spo),
            format_ratio(spo.participation_ratio),
        ]
        print("\t".join(row))


def format_count_with_baseline(tally: RoleParticipation) -> str:
    baseline = tally.eligible_voters
    if baseline is None or baseline == 0:
        return str(tally.voters)
    return f"{tally.voters}/{baseline}"


def format_ratio(ratio: Optional[float]) -> str:
    if ratio is None:
        return "--"
    return f"{ratio * 100:.1f}%"


def format_action_label(action: ActionParticipation) -> str:
    identifier = action.identifier or "unknown"
    short_identifier = shorten_identifier(identifier)
    if action.index is None:
        return short_identifier
    if identifier.isdigit() and int(identifier) == action.index:
        return str(action.index)
    return f"{action.index}:{short_identifier}"


def shorten_identifier(identifier: str, keep_start: int = 6, keep_end: int = 4) -> str:
    identifier = identifier or ""
    if len(identifier) <= keep_start + keep_end + 1:
        return identifier
    return f"{identifier[:keep_start]}...{identifier[-keep_end:]}"


def export_json(path: str, actions: Sequence[ActionParticipation]) -> None:
    data = [action.to_dict() for action in actions]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    logging.info("Wrote JSON results to %s", path)


def export_csv(path: str, actions: Sequence[ActionParticipation]) -> None:
    fieldnames = [
        "action_index",
        "action_type",
        "title",
        "status",
        "created_epoch",
        "voting_start_epoch",
        "voting_end_epoch",
        "enacted_epoch",
        "anchor_hash",
        "action_identifier",
        "role",
        "unique_voters",
        "total_votes",
        "voting_power_lovelace",
        "participation_ratio",
        "eligible_voters",
        "non_voters",
        "yes_votes",
        "no_votes",
        "abstain_votes",
        "no_confidence_votes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for action in actions:
            for role, tally in action.tallies.items():
                row = {
                    "action_index": action.index,
                    "action_type": action.action_type,
                    "title": action.title,
                    "status": action.status,
                    "created_epoch": action.created_epoch,
                    "voting_start_epoch": action.voting_start_epoch,
                    "voting_end_epoch": action.voting_end_epoch,
                    "enacted_epoch": action.enacted_epoch,
                    "anchor_hash": action.anchor_hash,
                    "action_identifier": action.identifier,
                    "role": role,
                    "unique_voters": tally.voters,
                    "total_votes": tally.total_votes,
                    "voting_power_lovelace": tally.voting_power,
                    "participation_ratio": tally.participation_ratio,
                    "eligible_voters": tally.eligible_voters,
                    "non_voters": None,
                    "yes_votes": tally.votes_by_choice.get("yes", 0),
                    "no_votes": tally.votes_by_choice.get("no", 0),
                    "abstain_votes": tally.votes_by_choice.get("abstain", 0),
                    "no_confidence_votes": tally.votes_by_choice.get("no_confidence", 0),
                }
                if tally.eligible_voters is not None:
                    row["non_voters"] = max(tally.eligible_voters - tally.voters, 0)
                row = {key: ("" if value is None else value) for key, value in row.items()}
                writer.writerow(row)
    logging.info("Wrote CSV results to %s", path)


def print_trend_summary(trends: Dict[str, Dict[str, float]]) -> None:
    if not trends:
        print("\nNo trend statistics could be computed (missing baselines or votes).")
        return

    print("\nParticipation trend summary (averages, ratios as percentages):")
    for role, stats in trends.items():
        role_name = role.upper()
        avg_all = stats["avg_all"] * 100
        avg_first = stats["avg_first"] * 100
        avg_last = stats["avg_last"] * 100
        delta = stats["delta"] * 100
        print(
            f"  {role_name}: overall {avg_all:.1f}%, "
            f"early window {avg_first:.1f}%, "
            f"recent window {avg_last:.1f}%, "
            f"change {delta:+.1f} percentage points"
        )


def auto_discover_baselines(
    client: BlockfrostClient,
    *,
    skip: bool,
    overrides: Dict[RoleKey, Optional[int]],
) -> BaselineResolver:
    default_counts: Dict[RoleKey, Optional[int]] = {}
    drep_records: Optional[List[DrepRecord]] = None

    if skip:
        return BaselineResolver(overrides, drep_records=None, default_counts={})

    if overrides.get("drep") is None:
        logging.info("Fetching historical DRep registration data to build dynamic baselines...")
        drep_records = fetch_drep_records(client)
        default_counts["drep"] = len(fetch_active_dreps(client)) or None

    if overrides.get("cc") is None:
        default_counts["cc"] = len(fetch_committee_members(client)) or None
    if overrides.get("spo") is None:
        default_counts["spo"] = fetch_total_pool_count(client) or None

    return BaselineResolver(overrides, drep_records=drep_records, default_counts=default_counts)


def is_invalid_path_error(error: BlockfrostAPIError) -> bool:
    message = str(error).lower()
    return "invalid path" in message or "not found" in message


def discover_governance_actions_endpoint(
    client: BlockfrostClient,
    override: Optional[str],
) -> str:
    candidates = [override] if override else []
    candidates.extend(endpoint for endpoint in GOVERNANCE_ACTION_ENDPOINT_CANDIDATES if endpoint != override)

    for endpoint in candidates:
        if not endpoint:
            continue
        try:
            client.get(endpoint, params={"order": "asc", "count": 1, "page": 1})
        except BlockfrostAPIError as exc:
            if is_invalid_path_error(exc):
                logging.debug("Endpoint %s not supported by this Blockfrost deployment: %s", endpoint, exc)
                continue
            logging.debug("Endpoint %s returned unexpected error: %s", endpoint, exc)
            raise
        logging.info("Using governance actions endpoint: %s", endpoint)
        return endpoint

    message = (
        "None of the known governance action endpoints are available on this Blockfrost deployment. "
        "Use --actions-endpoint to provide the correct path or double-check the API base URL."
    )
    raise BlockfrostAPIError(message)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        project_id = obtain_project_id(args.project_id)
    except ValueError as exc:
        logging.error(str(exc))
        return 1

    overrides: Dict[str, Optional[int]] = {
        "drep": args.drep_baseline_count,
        "cc": args.cc_baseline_count,
        "spo": args.spo_baseline_count,
    }

    client = BlockfrostClient(
        project_id=project_id,
        base_url=args.base_url,
        sleep_seconds=max(0.0, args.sleep_seconds),
    )

    try:
        baseline_resolver = auto_discover_baselines(client, skip=args.disable_auto_baselines, overrides=overrides)

        actions_endpoint = discover_governance_actions_endpoint(client, args.actions_endpoint)

        actions_raw = fetch_governance_actions(
            client,
            min_epoch=args.min_epoch,
            max_actions=args.max_actions,
            endpoint=actions_endpoint,
        )

        action_results: List[ActionParticipation] = []
        for raw_action in actions_raw:
            identifier = extract_action_identifier(raw_action)
            if not identifier:
                logging.debug("Skipping action without a stable identifier: %s", raw_action)
                continue

            index_value = raw_action.get("action_index") or raw_action.get("index") or raw_action.get("cert_index")
            parsed_index: Optional[int] = None
            if isinstance(index_value, (int, float)):
                parsed_index = parse_int(index_value)
            elif isinstance(index_value, str):
                trimmed = index_value.strip()
                if trimmed.isdigit():
                    parsed_index = int(trimmed)

            detail = fetch_action_detail(client, actions_endpoint, identifier)
            metadata_info = fetch_action_metadata(client, actions_endpoint, identifier)

            try:
                votes = fetch_votes_for_action(client, actions_endpoint, identifier)
            except BlockfrostAPIError as exc:
                if is_invalid_path_error(exc):
                    logging.error(
                        "Votes endpoint derived from %s is not available. Override with --actions-endpoint.",
                        actions_endpoint,
                    )
                raise

            creation_tx_hash = raw_action.get("tx_hash") or detail.get("tx_hash")
            created_epoch = parse_optional_int(detail.get("created_epoch"))
            if created_epoch is None:
                created_epoch = fetch_tx_epoch(client, creation_tx_hash)

            voting_start_epoch = parse_optional_int(detail.get("voting_start_epoch"))
            voting_end_epoch = parse_optional_int(detail.get("voting_end_epoch"))

            if votes:
                first_vote_epoch = fetch_tx_epoch(client, votes[0].get("tx_hash"))
                last_vote_epoch = fetch_tx_epoch(client, votes[-1].get("tx_hash"))
            else:
                first_vote_epoch = None
                last_vote_epoch = None

            if voting_start_epoch is None:
                voting_start_epoch = first_vote_epoch
            if voting_end_epoch is None:
                expiry_epoch = parse_optional_int(detail.get("expiration"))
                voting_end_epoch = expiry_epoch if expiry_epoch is not None else last_vote_epoch

            enacted_epoch = parse_optional_int(detail.get("enacted_epoch"))
            anchor_hash = metadata_info.get("hash") or detail.get("anchor_hash")

            status = derive_status(detail) or raw_action.get("status")
            if isinstance(status, str):
                status = status.strip() or None

            action_type = str(
                raw_action.get("type")
                or raw_action.get("action_type")
                or raw_action.get("governance_type")
                or detail.get("type")
                or detail.get("governance_type")
                or "unknown"
            )

            title = extract_title_from_sources(raw_action, metadata_info)

            reference_epoch = determine_electorate_epoch(
                detail,
                created_epoch=created_epoch,
                voting_start_epoch=voting_start_epoch,
                voting_end_epoch=voting_end_epoch,
                first_vote_epoch=first_vote_epoch,
                last_vote_epoch=last_vote_epoch,
            )
            baseline_counts = baseline_resolver.baseline_counts_for_epoch(reference_epoch)
            tallies = summarise_votes(votes, baseline_counts)

            action_results.append(
                build_action_participation(
                    identifier=identifier,
                    index=parsed_index,
                    action_type=action_type,
                    title=title,
                    status=status,
                    created_epoch=created_epoch,
                    voting_start_epoch=voting_start_epoch,
                    voting_end_epoch=voting_end_epoch,
                    enacted_epoch=enacted_epoch,
                    anchor_hash=anchor_hash,
                    tallies=tallies,
                )
            )

        # Sort by the voting start epoch (or index as a fallback) to show chronological order.
        action_results.sort(
            key=lambda a: (
                a.voting_start_epoch or a.created_epoch or sys.maxsize,
                a.index if a.index is not None else sys.maxsize,
                a.identifier,
            )
        )

        print_summary_table(action_results)
        trends = summarise_trends(action_results)
        print_trend_summary(trends)

        if args.output_json:
            export_json(args.output_json, action_results)
        if args.output_csv:
            export_csv(args.output_csv, action_results)

        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
