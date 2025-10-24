#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, csv, argparse, collections, requests, json

KOIOS_BASE = os.getenv("KOIOS_BASE", "https://api.koios.rest/api/v1")
BF_BASE = os.getenv("BLOCKFROST_BASE", "https://cardano-mainnet.blockfrost.io/api/v0")

HEADERS_JSON = {"accept": "application/json", "content-type": "application/json"}
TIMEOUT_GET  = 60
TIMEOUT_POST = 120
SLEEP        = 0.15

POOL_INFO_BATCH = 80  # auto-reduces on 413

# ---------- helpers ----------
def ada(lovelace):
    try:
        return float(int(lovelace)) / 1_000_000.0
    except Exception:
        return 0.0

def normalize_literal(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    for ch in ["-", " ", ":", "."]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def extract_ticker(meta_json):
    """
    Try to pull the CIP-6 ticker from pool_info.meta_json.
    Handles dict, JSON string, or None. Returns '' if not found.
    """
    if not meta_json:
        return ""
    try:
        # If it's a string, parse to JSON
        if isinstance(meta_json, str):
            meta_json = json.loads(meta_json)
        if isinstance(meta_json, dict):
            t = meta_json.get("ticker") or meta_json.get("pool_ticker")
            return str(t) if t else ""
    except Exception:
        pass
    return ""

TARGETS = {
    "abstain",
    "always_abstain",
    "no_confidence",
    "always_no_confidence",
    "drep_always_abstain",
    "drep_always_no_confidence",
}

# ---------- pool id enumeration ----------
def koios_pool_list_try_post_pagination():
    sess = requests.Session()
    sess.headers.update(HEADERS_JSON)
    all_ids = []
    offset = 0
    page_size = 1000
    while True:
        r = sess.post(f"{KOIOS_BASE}/pool_list",
                      json={"_offset": offset, "_count": page_size},
                      timeout=TIMEOUT_POST)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        ids = [p["pool_id_bech32"] for p in page]
        all_ids.extend(ids)
        sys.stderr.write(f"Koios POST /pool_list: {len(ids)} (total {len(all_ids)}), offset {offset}\n")
        offset += len(ids)
        time.sleep(SLEEP)
        if len(ids) < page_size:
            break
        if offset >= 1000 and len(all_ids) == 1000:
            return None
    return all_ids

def koios_pool_list_try_get_no_params():
    r = requests.get(f"{KOIOS_BASE}/pool_list", headers=HEADERS_JSON, timeout=TIMEOUT_GET)
    r.raise_for_status()
    page = r.json()
    ids = [p["pool_id_bech32"] for p in page]
    sys.stderr.write(f"Koios GET(no params) /pool_list: {len(ids)} (likely capped)\n")
    return ids

def blockfrost_pool_ids(project_id):
    sess = requests.Session()
    sess.headers.update({"project_id": project_id})
    all_ids, page = [], 1
    while True:
        url = f"{BF_BASE}/pools?page={page}"
        r = sess.get(url, timeout=TIMEOUT_GET)
        if r.status_code == 429:
            time.sleep(2)
            continue
        r.raise_for_status()
        arr = r.json()
        if not arr:
            break
        ids = [p for p in arr]
        all_ids.extend(ids)
        sys.stderr.write(f"Blockfrost /pools page {page}: {len(ids)} (total {len(all_ids)})\n")
        page += 1
        time.sleep(SLEEP)
    return all_ids

def get_all_pool_ids(blockfrost_key=None):
    ids = koios_pool_list_try_post_pagination()
    if ids and len(ids) > 1000:
        return ids
    try:
        ids_get = koios_pool_list_try_get_no_params()
    except Exception:
        ids_get = []
    if (ids and len(ids) == 1000) or len(ids_get) == 1000 or not ids:
        if not blockfrost_key:
            sys.exit("Need all pool IDs. Provide --blockfrost-key or export BLOCKFROST_PROJECT_ID.")
        sys.stderr.write("Using Blockfrost to enumerate all pools…\n")
        return blockfrost_pool_ids(blockfrost_key)
    return ids_get or ids

# ---------- Koios pool_info ----------
def fetch_pool_info_rows(pool_ids):
    sess = requests.Session()
    sess.headers.update(HEADERS_JSON)
    rows, i, batch = [], 0, POOL_INFO_BATCH
    while i < len(pool_ids):
        chunk = pool_ids[i:i+batch]
        r = sess.post(f"{KOIOS_BASE}/pool_info",
                      json={"_pool_bech32_ids": chunk},
                      timeout=TIMEOUT_POST)
        if r.status_code == 413:
            if batch <= 10:
                r.raise_for_status()
            batch = max(10, batch // 2)
            sys.stderr.write(f"413: reducing /pool_info batch to {batch} and retrying\n")
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("X-RateLimit-Reset", "2"))
            sys.stderr.write(f"429: sleeping {wait}s then retrying\n")
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        for p in data:
            vp = p.get("voting_power")
            try:
                vp_int = int(vp) if vp is not None else 0
            except Exception:
                try:
                    vp_int = int(float(vp))
                except Exception:
                    vp_int = 0
            lit_raw = p.get("reward_addr_delegated_drep")
            lit_norm = normalize_literal(lit_raw)
            ticker = extract_ticker(p.get("meta_json"))
            rows.append({
                "pool_id_bech32": p.get("pool_id_bech32"),
                "ticker": ticker,
                "reward_addr": p.get("reward_addr"),
                "reward_addr_delegated_drep_raw": lit_raw,
                "reward_addr_delegated_drep_norm": lit_norm,
                "voting_power_lovelace": vp_int,
                "voting_power_ADA": f"{ada(vp_int):.6f}",
            })
        i += len(chunk)
        time.sleep(SLEEP)
    return rows

# ---------- write + summarize ----------
def write_csv_rows(rows, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def summarize(rows):
    groups = collections.defaultdict(lambda: {"count":0, "lovelace":0})
    total_lovelace = 0
    for r in rows:
        lit = r["reward_addr_delegated_drep_norm"]
        ll = int(r["voting_power_lovelace"])
        groups[lit]["count"] += 1
        groups[lit]["lovelace"] += ll
        total_lovelace += ll

    with open("voting_power_by_literal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["normalized_literal","pools","total_lovelace","total_ADA","share_%"])
        for lit, agg in sorted(groups.items(), key=lambda kv: (-kv[1]["lovelace"], str(kv[0]))):
            ada_sum = ada(agg["lovelace"])
            share = (agg["lovelace"] / total_lovelace * 100.0) if total_lovelace else 0.0
            w.writerow([lit if lit is not None else "", agg["count"], agg["lovelace"], f"{ada_sum:.6f}", f"{share:.4f}"])

    norm_targets = {normalize_literal(t) for t in TARGETS}
    abstain_total = sum(groups.get(k, {}).get("lovelace", 0) for k in norm_targets if "abstain" in (k or ""))
    noconf_total  = sum(groups.get(k, {}).get("lovelace", 0) for k in norm_targets if "no_confidence" in (k or ""))

    print("\n=== Voting power summary (targets) ===")
    print(f"Always Abstain total: {abstain_total} lovelace  ({ada(abstain_total):.6f} ADA)")
    print(f"Always No-Confidence total: {noconf_total} lovelace  ({ada(noconf_total):.6f} ADA)")
    print(f"Overall voting power seen: {total_lovelace} lovelace  ({ada(total_lovelace):.6f} ADA)")

def main():
    ap = argparse.ArgumentParser(description="Dump all pools with DRep delegation, voting power, and ticker; tally by literal.")
    ap.add_argument("--blockfrost-key", default=os.getenv("BLOCKFROST_PROJECT_ID"),
                    help="Blockfrost project ID (env BLOCKFROST_PROJECT_ID if not passed)")
    args = ap.parse_args()

    print("Enumerating pool IDs…", file=sys.stderr)
    pool_ids = get_all_pool_ids(args.blockfrost_key)
    print(f"Total pool IDs discovered: {len(pool_ids)}", file=sys.stderr)

    print("Fetching Koios /pool_info in batches…", file=sys.stderr)
    rows = fetch_pool_info_rows(pool_ids)

    out = "pools_with_drep_and_voting_power.csv"
    write_csv_rows(
        rows,
        out,
        [
            "pool_id_bech32",
            "ticker",
            "reward_addr",
            "reward_addr_delegated_drep_raw",
            "reward_addr_delegated_drep_norm",
            "voting_power_lovelace",
            "voting_power_ADA",
        ],
    )
    print(f"Wrote {len(rows)} rows to {out}", file=sys.stderr)

    summarize(rows)
    print("Wrote voting_power_by_literal.csv", file=sys.stderr)

if __name__ == "__main__":
    main()
