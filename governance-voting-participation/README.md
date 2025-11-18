# Governance Voting Participation

Analyze Cardano CIP-1694 governance participation using the Blockfrost API.  
The script fetches all governance actions, tallies votes per role (DRep, Constitutional Committee, SPO), aggregates withdrawals, and prints trend summaries. Results can also be exported to JSON and CSV for downstream analysis.

## Prerequisites

- Python 3.10+ (3.11 recommended).
- `pip install requests`
- A Blockfrost project ID with access to the desired network (e.g., mainnet).

## Quick Start

```powershell
cd C:\Users\nicol\workspace\github.com\thenic95\cardano-governance-scripts\governance-voting-participation

# Optionally use a virtual environment (already present as .venv in this repo)
# .\.venv\Scripts\Activate.ps1

$env:BLOCKFROST_PROJECT_ID = "mainnetXXXXXXXXXXXX"
python .\analyze_voting_participation.py --max-actions 3 --verbose --output-csv results --output-json results
```

Notes:

- The script autodetects governance endpoints and electorate baselines unless disabled (`--disable-auto-baselines`).
- Providing a bare filename to `--output-csv` / `--output-json` writes to `results/YYYYMMDD_<name>.csv|json`. Use `--results-dir <path>` to override the directory or pass an absolute path to bypass timestamping.

## Common CLI Options

| Flag | Description |
| --- | --- |
| `--project-id <id>` | Blockfrost project ID (overrides env vars `BLOCKFROST_PROJECT_ID`, `BLOCKFROST_API_KEY`, or `BLOCKFROST_PROJECT_ID_MAINNET`). |
| `--base-url <url>` | Blockfrost base URL (default mainnet). |
| `--min-epoch`, `--max-actions` | Trim the action set for quicker exploratory runs. |
| `--sleep-seconds <float>` | Delay between API calls (default `0.2`). Increase if rate limited. |
| `--disable-auto-baselines` | Skips detecting electorate sizes; combine with `--drep-baseline-count`, `--cc-baseline-count`, `--spo-baseline-count`. |
| `--actions-endpoint <path>` | Force a specific governance endpoint if auto-detection fails. |
| `--output-json <name>` / `--output-csv <name>` | Export per-action data (auto timestamped unless you supply an absolute/relative path with directories). |
| `--results-dir <path>` | Directory used for timestamped exports (default: `results/`). |
| `--verbose` | Emits detailed logging (API calls, caching, etc.). |

## Output

1. **Console summary table** – per-action voters per constituency, participation %.
2. **Trend summary** – rolling averages of participation ratios (first vs. last window).
3. **Overall activity table** – unique voters, vote events, and action coverage per role.
4. **Exports (optional)** – JSON includes structured action/vote/withdrawal data; CSV expands each action/role row with tallies and withdrawal aggregates.

## Tips

- For “sanity check” runs, limit the scope with `--max-actions 5`. Only add `--disable-auto-baselines` if you also supply explicit `--drep/--cc/--spo` baseline counts; otherwise the `eligible_voters` columns will be blank by design.
- If Blockfrost changes endpoint names, use `--actions-endpoint /governance/actions`.
- Reuse cached DRep history in `.cache/drep_records.json` for faster reruns.
