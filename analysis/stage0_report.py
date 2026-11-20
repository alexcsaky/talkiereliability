"""Stage 0 report: join model outputs with outcomes and write the numbers.

Reads ``results/stage0/*.json`` (written by serve/modal_app.py) and
``outcomes/*.jsonl``. Writes ``results/stage0/report.md``. It reports; it does
not decide. Gate thresholds are Alex's.

Run: ``python -m analysis.stage0_report``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from analysis.metrics import auroc, brier, cluster_bootstrap, softmax

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "stage0"
OUT = REPO / "outcomes"

# No fixed AUROC threshold: Alex decides from the reported value and interval (20 Nov 2026).


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def known_truth_section(kt: dict) -> list[str]:
    truth = {r["question_id"]: r["outcome"] for r in _jsonl(OUT / "known_truth.jsonl")}
    qmeta = {json.loads(line)["id"]: json.loads(line)
             for line in (REPO / "questions" / "known_truth.jsonl").read_text().splitlines()}
    items = [{"lo": f["logodds_raw"], "p": f["p_raw"], "y": truth[f["question_id"]],
              "family": f["family"], "cluster": qmeta[f["question_id"]]["cluster"]}
             for f in kt["forecasts"]]
    stat = lambda xs: auroc([x["lo"] for x in xs], [x["y"] for x in xs])  # noqa: E731
    est, lo, hi = cluster_bootstrap(stat, items, "cluster")
    lines = [f"### Known-truth check ({kt['model']}{kt.get('label', '')})", "",
             f"{len(items)} questions, closed book, outcomes 1900 to 1929. "
             f"Firewall rejections: {len(kt['rejected_by_firewall'])}.", "",
             "| Family | n | AUROC | Brier (raw p) | Mean log-odds when A true | when A false |",
             "| --- | --- | --- | --- | --- | --- |"]
    fams = sorted({x["family"] for x in items}) + ["all"]
    for fam in fams:
        sub = items if fam == "all" else [x for x in items if x["family"] == fam]
        t = [x["lo"] for x in sub if x["y"]]
        f = [x["lo"] for x in sub if not x["y"]]
        lines.append(f"| {fam} | {len(sub)} | {stat(sub):.3f} | "
                     f"{brier([x['p'] for x in sub], [x['y'] for x in sub]):.3f} | "
                     f"{sum(t) / max(1, len(t)):+.2f} | {sum(f) / max(1, len(f)):+.2f} |")
    by_t: dict[str, list] = {}
    for r in kt["per_template"]:
        by_t.setdefault(r["template_id"], []).append(r)
    lines += ["", f"AUROC, all families: **{est:.3f}** (95% interval {lo:.3f} to {hi:.3f}, "
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
    lines = [f"### Leak probes ({lk['model']})", "",
             "| Probe | Control | True option rank | Share of true option | Uniform share |",
             "| --- | --- | --- | --- | --- |"]
    ctl_top, leak_top, n_ctl, n_leak, leak_ratio = 0, 0, 0, 0, []
    for r in lk["rows"]:
        if r["kind"] != "choice":
            continue
        t = truth[r["probe_id"]]
        i = r["options"].index(t["true_option"])
        sh = softmax(r["logprobs"])
        rank = 1 + sum(x > r["logprobs"][i] for x in r["logprobs"])
        uni = 1 / len(sh)
        lines.append(f"| {r['probe_id']} | {'yes' if t['control'] else ''} | {rank} of "
                     f"{len(sh)} | {sh[i]:.2f} | {uni:.2f} |")
        if t["control"]:
            n_ctl += 1
            ctl_top += rank == 1
        else:
            n_leak += 1
            leak_top += rank == 1
            leak_ratio.append(math.log(sh[i] / uni))
    exp_top = sum(1 / len(r["options"]) for r in lk["rows"]
                  if r["kind"] == "choice" and not truth[r["probe_id"]]["control"])
    lines += ["", f"Controls (before 1931) ranked first: {ctl_top} of {n_ctl}.",
              f"Post-1930 probes ranked first: {leak_top} of {n_leak} "
              f"(chance would give about {exp_top:.1f}). Mean log of (share / uniform share): "
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


def main() -> None:
    parts = ["# Stage 0 results", "",
             "Generated by analysis/stage0_report.py. Numbers only; gate decisions are Alex's.", ""]
    for name, fn in (("parity.json", None), ("context.json", None), ("throughput.json", None)):
        p = RES / name
        if p.exists():
            d = json.loads(p.read_text())
            d.pop("per_prompt", None)
            d.pop("rows", None)
            parts += [f"### {name[:-5]}", "", "```json", json.dumps(d, indent=1)[:4000], "```", ""]
    for name, fn in (("known_truth_base.json", known_truth_section),
                     ("known_truth_base_noeot.json", known_truth_section),
                     ("known_truth_web.json", known_truth_section),
                     ("leak_base.json", leak_section), ("leak_it.json", leak_section),
                     ("leak_web.json", leak_section)):
        p = RES / name
        if p.exists():
            d = json.loads(p.read_text())
            if "noeot" in name:
                d["label"] = ", no <|endoftext|> at document start"
            parts += fn(d)
    kb = RES / "knowledge_bench_base.json"
    if kb.exists():
        d = json.loads(kb.read_text())
        parts += ["### Community probe set (closed book, multiple choice)", ""]
        for k, v in d.items():
            parts.append(f"- {k}: accuracy {v['accuracy']:.3f} against chance {v['chance']:.3f} "
                         f"(n = {len(v['rows'])}, firewall rejected {v['rejected_by_firewall']})")
        parts.append("")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "report.md").write_text("\n".join(parts))
    print((RES / "report.md").read_text())


if __name__ == "__main__":
    main()
