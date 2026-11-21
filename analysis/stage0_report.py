"""Stage 0 report: join model outputs with outcomes and write the numbers.

Reads ``results/stage0/*.json`` (written by serve/modal_app.py) and
``outcomes/*.jsonl``. Writes ``results/stage0/report.md``. It reports; it does
not decide. Gate thresholds are Alex's.

Run: ``python -m analysis.stage0_report``
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

from analysis.metrics import auroc, brier, cluster_bootstrap, log_softmax, softmax

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "stage0"
OUT = REPO / "outcomes"

# No fixed AUROC threshold: Alex decides from the reported value and interval (20 Nov 2026).


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def download_section(d: dict) -> list[str]:
    lines = ["### Weights and volume check", ""]
    for k, v in d.items():
        if k == "context_texts" and isinstance(v, list):
            ok = sum(1 for t in v if t.get("ok"))
            lines.append(f"- context_texts: {ok} of {len(v)} Gutenberg books admitted")
        elif isinstance(v, dict):
            checks = [f"{ck}={v[ck]}" for ck in ("sha256_ok", "vocab_sha1_ok") if ck in v]
            lines.append(f"- `{k}`: " + (", ".join(checks) if checks else "downloaded"))
    return lines + [""]


def health_section(d: dict) -> list[str]:
    return [
        "### Server health (base)", "",
        f"- model: `{d.get('model')}` (`{d.get('repo')}` @ `{d.get('revision', '')[:12]}`)",
        f"- dtype: `{d.get('dtype')}`, device: `{d.get('device')}`, "
        f"load_seconds: {d.get('load_seconds')}, max_seq_len: {d.get('max_seq_len')}", "",
    ]


def parity_section(d: dict) -> list[str]:
    summary = {k: v for k, v in d.items() if k != "per_prompt"}
    return ["### Parity (official vs vendored transformers port)", "",
            "```json", json.dumps(summary, indent=1), "```", ""]


def context_section(d: dict) -> list[str]:
    lines = ["### Context length", ""]
    lvc = d.get("loss_vs_context", {}).get("by_total", {})
    if lvc:
        lines += ["Loss on a fixed 256-token span against total sequence length:", "",
                  "| Total tokens | Mean loss (nats/tok) | Perplexity | n texts |",
                  "| --- | --- | --- | --- |"]
        for total, v in sorted(lvc.items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {total} | {v['mean_loss']:.4f} | {v['ppl']:.2f} | {v['n_texts']} |")
        lines.append("")
    lbp = d.get("loss_by_position", {})
    if lbp:
        lines += ["Mean loss by 512-token position bin (4,096-token window):", "",
                  ", ".join(f"{k}: {v:.4f}" if v is not None else f"{k}: n/a"
                            for k, v in lbp.items()), ""]
    ret = d.get("retrieval", {})
    by_t = ret.get("by_total", {})
    if by_t:
        lines += [f"Needle retrieval (chance {ret.get('chance', 0.25):.2f}, "
                  f"{ret.get('seconds', '?')}s):", "",
                  "| Total tokens | Accuracy | Chose distractor | n | By depth |",
                  "| --- | --- | --- | --- | --- |"]
        for total, v in sorted(by_t.items(), key=lambda kv: int(kv[0])):
            depths = ", ".join(f"{dk}: {dv:.2f}" for dk, dv in sorted(v.get("by_depth", {}).items()))
            lines.append(f"| {total} | {v['accuracy']:.3f} | {v['chose_distractor']:.3f} | "
                         f"{v['n']} | {depths} |")
        lines.append("")
    return lines


def throughput_section(d: dict) -> list[str]:
    lines = ["### Throughput", ""]
    if "tokens_per_1000_chars_gutenberg" in d:
        lines.append(f"Tokens per 1,000 characters (Gutenberg): "
                     f"**{d['tokens_per_1000_chars_gutenberg']:.1f}**")
        lines.append("")
    rows = d.get("rows", [])
    if rows:
        lines += ["| Seq len | Max batch tokens | Seqs | Tokens/s | Seconds | Peak VRAM (GiB) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            if r.get("oom"):
                lines.append(f"| {r['seq_len']} | {r['max_batch_tokens']} | — | OOM | — | — |")
            else:
                lines.append(f"| {r['seq_len']} | {r['max_batch_tokens']} | {r['seqs']} | "
                             f"{r['tokens_per_sec']} | {r['seconds']} | {r['peak_mem_gib']} |")
        lines.append("")
    return lines


def known_truth_section(kt: dict) -> list[str]:
    truth = {r["question_id"]: r["outcome"] for r in _jsonl(OUT / "known_truth.jsonl")}
    qmeta = {json.loads(line)["id"]: json.loads(line)
             for line in (REPO / "questions" / "known_truth.jsonl").read_text().splitlines()
             if line.strip()}
    items = [{"lo": f["logodds_raw"], "p": f["p_raw"], "y": truth[f["question_id"]],
              "family": f["family"], "cluster": qmeta[f["question_id"]]["cluster"]}
             for f in kt["forecasts"]]
    stat = lambda xs: auroc([x["lo"] for x in xs], [x["y"] for x in xs])  # noqa: E731
    # Families differ in a constant offset that has nothing to do with truth
    # (negated statements are longer, so every "negation" item leans towards A).
    # Pooling raw log-odds mixes those offsets into the ranking; the headline
    # therefore centres log-odds within each family first (no outcomes used).
    fam_mean: dict[str, float] = {}
    for fam in {x["family"] for x in items}:
        v = [x["lo"] for x in items if x["family"] == fam]
        fam_mean[fam] = sum(v) / len(v)
    for x in items:
        x["lo_c"] = x["lo"] - fam_mean[x["family"]]
    stat_c = lambda xs: auroc([x["lo_c"] for x in xs], [x["y"] for x in xs])  # noqa: E731
    est, lo, hi = cluster_bootstrap(stat_c, items, "cluster")
    lines = [f"### Known-truth check ({kt['model']}{kt.get('label', '')})", "",
             f"{len(items)} questions, closed book, outcomes 1900 to 1929. "
             f"Firewall rejections: {len(kt['rejected_by_firewall'])}.", "",
             "| Family | n | AUROC | Brier (raw p) | Mean log-odds when A true | when A false |",
             "| --- | --- | --- | --- | --- | --- |"]
    fams = sorted({x["family"] for x in items}) + ["all (uncentred)"]
    for fam in fams:
        sub = items if fam.startswith("all") else [x for x in items if x["family"] == fam]
        t = [x["lo"] for x in sub if x["y"]]
        f = [x["lo"] for x in sub if not x["y"]]
        lines.append(f"| {fam} | {len(sub)} | {stat(sub):.3f} | "
                     f"{brier([x['p'] for x in sub], [x['y'] for x in sub]):.3f} | "
                     f"{sum(t) / max(1, len(t)):+.2f} | {sum(f) / max(1, len(f)):+.2f} |")
    by_t: dict[str, list] = {}
    for r in kt["per_template"]:
        by_t.setdefault(r["template_id"], []).append(r)
    lines += ["", f"AUROC, all families, log-odds centred within family: **{est:.3f}** "
                  f"(95% interval {lo:.3f} to {hi:.3f}, "
                  f"bootstrap over {len({x['cluster'] for x in items})} clusters). "
                  "No fixed threshold; Alex decides.", "",
              "Per template (AUROC): " + ", ".join(
                  f"{tid} {auroc([r['logodds'] for r in rs], [truth[r['question_id']] for r in rs]):.3f}"
                  for tid, rs in sorted(by_t.items())), ""]
    bad = sum(not r["boundary_ok"] for r in kt["per_template"])
    if bad:
        lines.append(f"Warning: {bad} rows had a tokenisation boundary mismatch.")
    return lines


def leak_section(lk: dict) -> list[str]:
    truth = {r["probe_id"]: r for r in _jsonl(OUT / "leak_probes.jsonl")}
    lines = [f"### Leak probes ({lk['model']}{lk.get('label', '')})", "",
             "| Probe | Control | Rank (sum lp) | Rank (mean lp/tok) | Share | Uniform |",
             "| --- | --- | --- | --- | --- | --- |"]
    ctl_top, leak_top, leak_top_norm, n_ctl, n_leak, leak_ratio = 0, 0, 0, 0, 0, []
    for r in lk["rows"]:
        if r["kind"] != "choice":
            continue
        t = truth[r["probe_id"]]
        i = r["options"].index(t["true_option"])
        lps = r["logprobs"]
        ntoks = r.get("n_tokens", [1] * len(lps))
        mean_lps = [lp / max(1, nt) for lp, nt in zip(lps, ntoks)]
        log_sh = log_softmax(lps)
        sh = [math.exp(x) for x in log_sh]
        rank = 1 + sum(x > lps[i] for x in lps)
        rank_norm = 1 + sum(x > mean_lps[i] for x in mean_lps)
        uni = 1 / len(sh)
        lines.append(f"| {r['probe_id']} | {'yes' if t['control'] else ''} | {rank} of "
                     f"{len(sh)} | {rank_norm} of {len(sh)} | {sh[i]:.2f} | {uni:.2f} |")
        if t["control"]:
            n_ctl += 1
            ctl_top += rank == 1
        else:
            n_leak += 1
            leak_top += rank == 1
            leak_top_norm += rank_norm == 1
            leak_ratio.append(log_sh[i] - math.log(uni))
    exp_top = sum(1 / len(r["options"]) for r in lk["rows"]
                  if r["kind"] == "choice" and not truth[r["probe_id"]]["control"])
    lines += ["", f"Controls (before 1931) ranked first: {ctl_top} of {n_ctl}.",
              f"Post-1930 probes ranked first: {leak_top} of {n_leak} by sum logprob "
              f"({leak_top_norm} of {n_leak} by mean logprob/token; chance ~{exp_top:.1f}). "
              f"Mean log of (share / uniform share): "
              f"{sum(leak_ratio) / max(1, len(leak_ratio)):+.2f}.", "",
              "Year probes: probability mass on each year as 'the present year'", "",
              "| Header | Most likely year | Mass on 1931 or later | Expected year |",
              "| --- | --- | --- | --- |"]
    for r in lk["rows"]:
        if r["kind"] != "year":
            continue
        sh = softmax(r["logprobs"])
        ys = [int(y) for y in r["options"]]
        top = ys[max(range(len(sh)), key=lambda i: sh[i])]
        late = sum(p for p, y in zip(sh, ys) if y >= 1931)
        lines.append(f"| {r['probe_id']} | {top} | {late:.2f} | "
                     f"{sum(p * y for p, y in zip(sh, ys)):.1f} |")
    return lines + [""]


def _kb_answers(repo: str, revision: str, bucket: str) -> dict[tuple[str, str], int]:
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/by_bucket/{bucket}.jsonl"
    with urllib.request.urlopen(url, timeout=120) as r:
        rows = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
    return {
        (row["qid"], row["predicate"]): row["probe_mc"]["answer_index"]
        for row in rows if row.get("probe_mc")
    }


def knowledge_bench_section(d: dict) -> list[str]:
    lines = ["### Community probe set (closed book, multiple choice)", ""]
    for label, v in d.items():
        rows = v.get("rows", [])
        if "accuracy" in v:
            lines.append(f"- {label}: accuracy {v['accuracy']:.3f} against chance {v['chance']:.3f} "
                         f"(n = {len(rows)}, firewall rejected {v['rejected_by_firewall']})")
            continue
        ans = _kb_answers(v["kb_repo"], v["kb_revision"], v["bucket"])
        scored = [r for r in rows if (r["qid"], r["predicate"]) in ans]
        acc_sum = sum(
            max(range(len(r["logprobs"])), key=lambda i: r["logprobs"][i])
            == ans[(r["qid"], r["predicate"])]
            for r in scored
        ) / max(1, len(scored))
        acc_norm = sum(
            max(range(len(r["logprobs"])),
                key=lambda i: r["logprobs"][i] / max(1, r["n_tokens"][i]))
            == ans[(r["qid"], r["predicate"])]
            for r in scored
        ) / max(1, len(scored))
        chance = sum(1 / r["n_options"] for r in scored) / max(1, len(scored))
        lines.append(f"- {label}: accuracy {acc_sum:.3f} sum-lp / {acc_norm:.3f} mean-lp "
                     f"against chance {chance:.3f} (n = {len(scored)}, "
                     f"firewall rejected {v['rejected_by_firewall']})")
    return lines + [""]


def main() -> None:
    parts = ["# Stage 0 results", "",
             "Generated by analysis/stage0_report.py. Numbers only; gate decisions are Alex's.", ""]
    for name, fn in (("download.json", download_section),
                     ("health_base.json", health_section),
                     ("parity.json", parity_section),
                     ("context.json", context_section),
                     ("throughput.json", throughput_section)):
        p = RES / name
        if p.exists():
            parts += fn(json.loads(p.read_text()))
    for name, fn, label in (
        ("known_truth_base.json", known_truth_section, ""),
        ("known_truth_base_noeot.json", known_truth_section, ", no <|endoftext|> at document start"),
        ("known_truth_web.json", known_truth_section, ""),
        ("leak_base.json", leak_section, ""),
        ("leak_it.json", leak_section, ", raw completion"),
        ("leak_it_chat.json", leak_section, ", chat template"),
        ("leak_web.json", leak_section, ""),
    ):
        p = RES / name
        if p.exists():
            d = json.loads(p.read_text())
            if label:
                d["label"] = label
            parts += fn(d)
    kb = RES / "knowledge_bench_base.json"
    if kb.exists():
        parts += knowledge_bench_section(json.loads(kb.read_text()))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "report.md").write_text("\n".join(parts))
    print((RES / "report.md").read_text())


if __name__ == "__main__":
    main()
