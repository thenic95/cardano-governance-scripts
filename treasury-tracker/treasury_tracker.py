#!/usr/bin/env python3
"""
treasury_tracker.py

Tracks Cardano Treasury inflows and outflows from Koios.

Output columns (minimal):
- epoch
- treasury_balance_ada  (from /totals)
- inflow_ada            (only shown when balance rose and we can compute it)
- outflow_ada           (declared + inferred)

Notes:
- withdrawals are filtered by epoch to avoid 504s from Koios
- when the treasury balance drops in an epoch, we can't separate "real inflow" from
  "extra outflow" with Koios data alone, so we leave inflow blank for that epoch
"""

import requests
from collections import defaultdict
from decimal import Decimal
import argparse

BASE_URL = "https://api.koios.rest/api/v1"
HEADERS = {"accept": "application/json"}
PAGE_LIMIT = 1000

session = requests.Session()
session.headers.update(HEADERS)


def fetch_all_rows(endpoint: str, extra_params=None):
    """
    Fetch all rows from a Koios list endpoint, with paging.
    extra_params can be:
      - None
      - dict
      - list of (key, value) tuples (needed for epoch_no=gte.X & epoch_no=lte.Y)
    """
    all_rows = []
    offset = 0

    while True:
        # normalize params
        if extra_params is None:
            params = {}
        elif isinstance(extra_params, dict):
            params = extra_params.copy()
        else:
            # assume iterable of pairs
            params = list(extra_params)

        # add paging
        if isinstance(params, dict):
            params.update({"limit": PAGE_LIMIT, "offset": offset})
        else:
            params = list(params) + [("limit", PAGE_LIMIT), ("offset", offset)]

        resp = session.get(f"{BASE_URL}{endpoint}", params=params, timeout=60)
        resp.raise_for_status()
        chunk = resp.json()
        all_rows.extend(chunk)

        if len(chunk) < PAGE_LIMIT:
            break

        offset += PAGE_LIMIT

    return all_rows


def get_totals():
    """Get per-epoch totals, includes treasury"""
    resp = session.get(f"{BASE_URL}/totals", timeout=60)
    resp.raise_for_status()
    data = resp.json()
    data.sort(key=lambda x: int(x["epoch_no"]))
    return data


def get_treasury_withdrawals(epoch_min=None, epoch_max=None, epoch_whitelist=None):
    """
    Get treasury withdrawals, filtered by epoch so we don't pull thousands of rows.
    """
    # explicit list
    if epoch_whitelist is not None:
        epoch_list = ",".join(str(e) for e in sorted(epoch_whitelist))
        return fetch_all_rows(
            "/treasury_withdrawals",
            extra_params={"epoch_no": f"in.({epoch_list})"},
        )

    # range
    if epoch_min is not None and epoch_max is not None:
        # need two filters -> list of tuples
        return fetch_all_rows(
            "/treasury_withdrawals",
            extra_params=[("epoch_no", f"gte.{epoch_min}"), ("epoch_no", f"lte.{epoch_max}")],
        )
    elif epoch_min is not None:
        return fetch_all_rows(
            "/treasury_withdrawals",
            extra_params={"epoch_no": f"gte.{epoch_min}"},
        )
    elif epoch_max is not None:
        return fetch_all_rows(
            "/treasury_withdrawals",
            extra_params={"epoch_no": f"lte.{epoch_max}"},
        )

    # fallback (small datasets only)
    return fetch_all_rows("/treasury_withdrawals")


def lovelace_to_ada(lovelace: int) -> Decimal:
    return Decimal(lovelace) / Decimal(1_000_000)


def build_epoch_outflows(treasury_wds):
    """sum withdrawals per epoch"""
    outflows = defaultdict(int)
    for wd in treasury_wds:
        epoch = int(wd["epoch_no"])
        amount = int(wd["amount"])
        outflows[epoch] += amount
    return outflows


def compute_treasury_flows(epoch_min=None, epoch_max=None, epoch_whitelist=None):
    totals = get_totals()
    withdrawals = get_treasury_withdrawals(epoch_min, epoch_max, epoch_whitelist)
    wd_by_epoch = build_epoch_outflows(withdrawals)

    rows = []
    prev_treasury = None

    for row in totals:
        epoch = int(row["epoch_no"])

        # apply user filter
        if epoch_whitelist is not None:
            if epoch not in epoch_whitelist:
                continue
        else:
            if (epoch_min is not None and epoch < epoch_min) or (
                epoch_max is not None and epoch > epoch_max
            ):
                continue

        treasury_bal = int(row["treasury"])

        if prev_treasury is None:
            # first included epoch: we don't know inflow yet
            rows.append(
                {
                    "epoch": epoch,
                    "treasury_balance": treasury_bal,
                    "implied_inflow": None,
                    "total_outflow": wd_by_epoch.get(epoch, 0),
                }
            )
        else:
            net = treasury_bal - prev_treasury
            declared_outflow = wd_by_epoch.get(epoch, 0)

            if net < 0:
                # treasury shrank; we can say how much went out (declared + inferred),
                # but we can't credibly say how much came in → leave inflow blank later
                inferred_outflow = max(0, -net - declared_outflow)
                total_outflow = declared_outflow + inferred_outflow
                implied_inflow = 0  # placeholder; we'll turn this into "" in output
            else:
                # treasury grew
                inferred_outflow = 0
                total_outflow = declared_outflow
                # inflow is: growth + what we spent
                implied_inflow = net + declared_outflow

            rows.append(
                {
                    "epoch": epoch,
                    "treasury_balance": treasury_bal,
                    "implied_inflow": implied_inflow,
                    "total_outflow": total_outflow,
                }
            )

        prev_treasury = treasury_bal

    return rows


def main():
    parser = argparse.ArgumentParser(description="Minimal Cardano Treasury tracker (Koios)")
    parser.add_argument("--from-epoch", type=int, dest="from_epoch", help="First epoch to include")
    parser.add_argument("--to-epoch", type=int, dest="to_epoch", help="Last epoch to include")
    parser.add_argument(
        "--epochs",
        type=str,
        help="Comma-separated exact epochs to include (overrides --from-epoch/--to-epoch)",
    )
    args = parser.parse_args()

    epoch_whitelist = None
    if args.epochs:
        epoch_whitelist = {int(e.strip()) for e in args.epochs.split(",") if e.strip()}

    rows = compute_treasury_flows(
        epoch_min=args.from_epoch,
        epoch_max=args.to_epoch,
        epoch_whitelist=epoch_whitelist,
    )

    # minimal header
    print("epoch,treasury_balance_ada,inflow_ada,outflow_ada")

    for r in rows:
        bal_ada = lovelace_to_ada(r["treasury_balance"])

        # show inflow only if it's positive and known
        if r["implied_inflow"] is None:
            inflow = ""
        else:
            inflow = lovelace_to_ada(r["implied_inflow"]) if r["implied_inflow"] > 0 else ""

        outflow = lovelace_to_ada(r["total_outflow"])

        print(f"{r['epoch']},{bal_ada},{inflow},{outflow}")


if __name__ == "__main__":
    main()