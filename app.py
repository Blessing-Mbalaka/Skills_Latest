"""
Flask viewer for the TF-IDF results written by deduplicate.py.

    python app.py                      # looks for the result CSVs next to this file
    python app.py --results-dir C:\\path\\to\\folder --port 5000

Reads (created by deduplicate.py):
    tfidf_by_module_skill.csv, tfidf_overall_module.csv     (module level)
    tfidf_by_course_skill.csv, tfidf_overall_course.csv     (course level, optional)
    cleaned_skill_labels_only.csv                           (merged wording per skill, optional)
"""
import argparse
import io
import logging
import os

import numpy as np
import pandas as pd
from flask import Flask, Response, abort, render_template, request, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("RESULTS_DIR", BASE_DIR)

app = Flask(__name__)
# Persist server exceptions so failures in a separately launched terminal can be diagnosed.
error_log = logging.FileHandler(os.path.join(BASE_DIR, "app_errors.log"), encoding="utf-8")
error_log.setLevel(logging.ERROR)
error_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
app.logger.addHandler(error_log)
app.config["TEMPLATES_AUTO_RELOAD"] = True

LEVELS = {"module": ("Modules", "module", "modules")}
METRICS = {
    "presence": "Presence (mentions)",
    "tf": "TF: share of mentions",
    "lift": "Lift: vs the average module",
    "tf_idf": "TF-IDF",
}
SORTS = {
    "score": "Score",
    "share": "Average TF",
    "coverage": "Modules containing the skill",
    "jobs": "Job demand (job text)",
    "name": "Name (A to Z)",
}
LIGHT = (231, 245, 234)
DARK = (15, 55, 39)

_CACHE = {}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _align_paths(level):
    return (os.path.join(RESULTS_DIR, f"alignment_skills_{level}.csv"),
            os.path.join(RESULTS_DIR, f"alignment_groups_{level}.csv"))


def _paths(level):
    return (
        os.path.join(RESULTS_DIR, f"tfidf_by_{level}_skill.csv"),
        os.path.join(RESULTS_DIR, f"tfidf_overall_{level}.csv"),
        os.path.join(RESULTS_DIR, "cleaned_skill_labels_only.csv"),
    )


def available_levels():
    out = []
    for level in LEVELS:
        long_p, over_p, _ = _paths(level)
        if os.path.isfile(long_p) and os.path.isfile(over_p):
            out.append(level)
    return out


def _mtime(p):
    return os.path.getmtime(p) if os.path.isfile(p) else 0


def load_level(level):
    """Return a dict of prepared tables for one level, or {'error': ...}. Cached until a file changes."""
    long_p, over_p, syn_p = _paths(level)
    al_s_p, al_g_p = _align_paths(level)
    key = (_mtime(long_p), _mtime(over_p), _mtime(syn_p), _mtime(al_s_p), _mtime(al_g_p))
    hit = _CACHE.get(level)
    if hit and hit["key"] == key:
        return hit["data"]

    long = pd.read_csv(long_p)
    over = pd.read_csv(over_p)
    group_col = long.columns[0]
    count_col = f"skill_count_in_{level}"
    total_col = f"total_skills_in_{level}"
    contain_col = f"{level}s_containing_skill"

    need_long = [group_col, "canonical_skill_label", count_col, total_col, "tf", "share", "lift", "tf_idf"]
    need_over = ["canonical_skill_label", "total_occurrences", "share", contain_col,
                 "idf", "idf_smooth", "score", "mean_tf_idf", "max_tf_idf"]
    missing = [c for c in need_long if c not in long.columns] + [c for c in need_over if c not in over.columns]
    if missing:
        data = {"error": "These result files were written by an older version of deduplicate.py "
                         f"(missing columns: {', '.join(missing)}). Run deduplicate.py again."}
    else:
        groups = sorted(long[group_col].astype(str).unique())
        long[group_col] = long[group_col].astype(str)
        sizes = long.groupby(group_col)[total_col].first().reindex(groups)

        def pivot(col):
            return (long.pivot(index="canonical_skill_label", columns=group_col, values=col)
                    .reindex(columns=groups).fillna(0.0))

        synonyms = {}
        if os.path.isfile(syn_p):
            syn = pd.read_csv(syn_p)
            if {"canonical_skill_label", "synonym_variations"} <= set(syn.columns):
                for name, variants in zip(syn["canonical_skill_label"], syn["synonym_variations"]):
                    synonyms[name] = [v for v in str(variants).split(" | ") if v and v != name]

        align = align_groups = None
        job_total = job_chunk_total = None
        if os.path.isfile(al_s_p) and os.path.isfile(al_g_p):
            a = pd.read_csv(al_s_p)
            ag = pd.read_csv(al_g_p)
            need_a = ["canonical_skill_label", "job_mentions", "job_share", "course_coverage", "status", "gap_score"]
            if all(c in a.columns for c in need_a) and len(ag.columns) > 1:
                align = a.set_index("canonical_skill_label")
                ag[ag.columns[0]] = ag[ag.columns[0]].astype(str)
                align_groups = ag.set_index(ag.columns[0]).reindex(groups)
                job_total = int(align["job_mentions"].sum())
                if "job_prevalence" in align.columns and "job_chunks" in align.columns:
                    nz = align[align["job_prevalence"] > 0]
                    if len(nz):
                        job_chunk_total = int(round((nz["job_chunks"] / nz["job_prevalence"]).median()))

        data = {
            "align": align, "align_groups": align_groups,
            "job_total": job_total, "job_chunk_total": job_chunk_total,
            "level": level, "group_col": group_col, "count_col": count_col,
            "contain_col": contain_col, "groups": groups, "sizes": sizes,
            "n_groups": len(groups), "n_skills": len(over),
            "n_mentions": int(sizes.sum()),
            "long": long, "over": over.set_index("canonical_skill_label"),
            "pivots": {"tf": pivot("tf"), "lift": pivot("lift"),
                       "tf_idf": pivot("tf_idf"), "count": pivot(count_col)},
            "synonyms": synonyms,
        }
    _CACHE[level] = {"key": key, "data": data}
    return data


def context():
    """Pick the level for this request. Returns (ctx, ds) or (ctx, None) when nothing can be shown."""
    levels = available_levels()
    ctx = {"results_dir": RESULTS_DIR, "levels": [], "level": None, "words": LEVELS["module"], "has_align": False}
    if not levels:
        return ctx, None
    level = request.args.get("level")
    if level not in levels:
        level = "module" if "module" in levels else levels[0]
    ctx["level"] = level
    ctx["words"] = LEVELS[level]
    on_compare = request.endpoint in ("compare", "modules_index", None)
    ctx["levels"] = [
        {"key": k, "label": LEVELS[k][0], "active": k == level,
         "url": url_for(request.endpoint if on_compare and request.endpoint else "compare",
                        **({**request.args.to_dict(), "level": k} if on_compare else {"level": k}))}
        for k in levels
    ]
    ds = load_level(level)
    ctx["has_align"] = bool(ds and "error" not in ds and ds.get("align") is not None)
    return ctx, ds


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def heat(t):
    """Background colour and text colour for an intensity t in [0, 1]."""
    t = float(np.clip(t, 0, 1))
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(LIGHT, DARK))
    lum = 0.2126 * (rgb[0] / 255) ** 2.2 + 0.7152 * (rgb[1] / 255) ** 2.2 + 0.0722 * (rgb[2] / 255) ** 2.2
    text = "#ffffff" if lum < 0.2 else "#06101c"
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", text


def fmt(metric, v):
    if v <= 0:
        return ""
    if metric == "presence":
        return str(int(round(v)))
    if metric == "tf":
        p = v * 100
        return "<0.1" if p < 0.1 else f"{p:.1f}"
    if metric == "lift":
        return f"{v:.1f}x"
    return f"{v:.3f}"


def select_rows(ds, sort, n, min_cov, q):
    over = ds["over"].copy()
    contain = ds["contain_col"]
    if q:
        over = over[over.index.str.contains(q, case=False, regex=False)]
    over = over[over[contain] >= min_cov]
    if sort == "name":
        over = over.sort_index(key=lambda s: s.str.lower())
    elif sort == "share":
        over = over.sort_values(["share", "score"], ascending=False)
    elif sort == "coverage":
        over = over.sort_values([contain, "score"], ascending=False)
    elif sort == "jobs" and ds.get("align") is not None:
        over = over.assign(_job=ds["align"]["job_share"].reindex(over.index).fillna(0.0))
        over = over.sort_values(["_job", "score"], ascending=False).drop(columns="_job")
    else:
        over = over.sort_values("score", ascending=False)
    return len(over), over if n == "all" else over.head(n)


def build_matrix(ds, ctx, metric, sort, n, min_cov, q):
    total, over = select_rows(ds, sort, n, min_cov, q)
    mat = ds["pivots"]["count" if metric == "presence" else metric].loc[over.index]
    counts = ds["pivots"]["count"].loc[over.index]
    lifts = ds["pivots"]["lift"].loc[over.index]
    vals = mat.values[mat.values > 0]
    cap = float(np.percentile(vals, 95)) if len(vals) else 1.0
    cap = cap if cap > 0 else 1.0
    max_score = float(over["score"].max()) if len(over) else 1.0
    rows = []
    for skill, r in over.iterrows():
        cells = []
        for g in ds["groups"]:
            v = float(mat.at[skill, g])
            if v <= 0:
                cells.append({"absent": True, "text": "", "bg": "", "fg": "", "mark": False, "title": f"{g}: not present"})
                continue
            bg, fg = heat(0.62 if metric == "presence" else np.sqrt(min(v / cap, 1.0)))
            c = int(counts.at[skill, g])
            mark = bool(metric != "presence" and lifts.at[skill, g] >= 2 and c >= 2)
            cells.append({"absent": False, "text": fmt(metric, v), "bg": bg, "fg": fg, "mark": mark,
                          "title": f"{g}: present, {c} mention{'s' if c != 1 else ''}, TF {float(ds['pivots']['tf'].at[skill, g]) * 100:.2f}%"})
        rows.append({
            "name": skill, "cells": cells,
            "coverage": int(r[ds["contain_col"]]),
            "score": float(r["score"]), "score_w": 100 * float(r["score"]) / max_score if max_score else 0,
            "job_share": (float(ds["align"]["job_share"].get(skill, 0.0)) if ds.get("align") is not None else None),
            "url": url_for("skill", name=skill, level=ctx["level"]),
        })
    cols = [{"name": g, "size": int(ds["sizes"][g]),
             "url": url_for("module", name=g, level=ctx["level"])} for g in ds["groups"]]
    return rows, cols, total


# ----------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------
@app.route("/")
@app.route("/compare")
@app.route("/new-analysis")
def compare():
    ctx, ds = context()
    if ds is None:
        return render_template("empty.html", **ctx), 200
    if "error" in ds:
        return render_template("empty.html", error=ds["error"], **ctx), 200

    metric = request.args.get("metric", "presence")
    metric = metric if metric in METRICS else "presence"
    sort = request.args.get("sort", "score")
    sort = sort if sort in SORTS else "score"
    n = "all" if request.args.get("n") == "all" else _clamp(request.args.get("n"), 5, max(300, ds["n_skills"]), 20)
    min_cov = _clamp(request.args.get("min_cov"), 1, ds["n_groups"], 1)
    q = request.args.get("q", "").strip()

    rows, cols, total = build_matrix(ds, ctx, metric, sort, n, min_cov, q)
    form = {"metric": metric, "sort": sort, "n": n, "min_cov": min_cov, "q": q, "level": ctx["level"]}
    csv_url = url_for("compare_csv", **form)
    return render_template(
        "compare.html", ds=ds, rows=rows, cols=cols, total=total, form=form,
        metrics=METRICS, sorts={k: v for k, v in SORTS.items() if k != "jobs" or ctx["has_align"]},
        csv_url=csv_url, **ctx)


@app.route("/compare.csv")
def compare_csv():
    ctx, ds = context()
    if ds is None or "error" in ds:
        abort(404)
    metric = request.args.get("metric", "presence")
    metric = metric if metric in METRICS else "presence"
    sort = request.args.get("sort", "score")
    sort = sort if sort in SORTS else "score"
    n = "all" if request.args.get("n") == "all" else _clamp(request.args.get("n"), 5, max(300, ds["n_skills"]), 20)
    min_cov = _clamp(request.args.get("min_cov"), 1, ds["n_groups"], 1)
    q = request.args.get("q", "").strip()
    _, over = select_rows(ds, sort, n, min_cov, q)
    out = ds["pivots"]["count" if metric == "presence" else metric].loc[over.index].copy()
    out.insert(0, "score", over["score"])
    out.insert(1, "average_tf", over["share"])
    out.insert(2, ds["contain_col"], over[ds["contain_col"]])
    out.index.name = "skill"
    buf = io.StringIO()
    out.to_csv(buf)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=skill_comparison_{metric}.csv"})


def job_info(ds, name):
    """Job-market numbers for one skill, or None when no job data was loaded."""
    al = ds.get("align")
    if al is None or name not in al.index:
        return None if al is None else {"mentions": 0, "share": 0.0, "chunks": None, "prevalence": None,
                                         "gap": 0.0, "status": "taught_not_in_jobs", "chunk_total": ds["job_chunk_total"]}
    x = al.loc[name]
    return {"mentions": int(x["job_mentions"]), "share": float(x["job_share"]),
            "chunks": int(x["job_chunks"]) if "job_chunks" in al.columns else None,
            "prevalence": float(x["job_prevalence"]) if "job_prevalence" in al.columns else None,
            "gap": float(x["gap_score"]), "status": str(x["status"]), "chunk_total": ds["job_chunk_total"]}


@app.route("/skill")
def skill():
    ctx, ds = context()
    if ds is None or "error" in ds:
        return render_template("empty.html", error=(ds or {}).get("error"), **ctx), 200
    name = request.args.get("name", "")
    al = ds.get("align")
    in_course = name in ds["over"].index
    if not in_course and not (al is not None and name in al.index):
        abort(404)
    P = ds["pivots"]
    M = ds["n_groups"]

    def pv(kind, g):
        return float(P[kind].at[name, g]) if in_course else 0.0

    max_tf = max([pv("tf", g) for g in ds["groups"]] + [0.0]) or 1.0
    per = []
    for g in ds["groups"]:
        tf = pv("tf", g)
        per.append({
            "name": g, "tf": tf, "count": int(pv("count", g)), "size": int(ds["sizes"][g]),
            "lift": pv("lift", g), "tf_idf": pv("tf_idf", g),
            "width": 100 * tf / max_tf, "url": url_for("module", name=g, level=ctx["level"]),
            "emphasised": bool(pv("lift", g) >= 2 and pv("count", g) >= 2),
        })
    if in_course:
        r = ds["over"].loc[name]
        df_ = int(r[ds["contain_col"]])
        info = {"share": float(r["share"]), "idf_smooth": float(r["idf_smooth"]), "score": float(r["score"]),
                "df": df_, "M": M, "total": int(r["total_occurrences"]), "mean_tf_idf": float(r["mean_tf_idf"]),
                "rank": int((ds["over"]["score"] > r["score"]).sum()) + 1, "of": ds["n_skills"]}
    else:
        info = {"share": 0.0, "idf_smooth": float(np.log((1 + M) / 1) + 1), "score": 0.0, "df": 0, "M": M,
                "total": 0, "mean_tf_idf": 0.0, "rank": None, "of": ds["n_skills"]}
    info["present"] = [p for p in per if p["count"] > 0]
    info["absent"] = [p for p in per if p["count"] == 0]
    return render_template("skill.html", skill=name, per=per, info=info, job=job_info(ds, name),
                           synonyms=ds["synonyms"].get(name, [])[:40], back=url_for("compare", level=ctx["level"]),
                           **ctx)


MODULE_SORTS = {
    "tf": "Largest share of the module",
    "tf_idf": "Most distinctive (TF-IDF)",
    "lift": "Highest lift",
    "jobs": "Highest job demand",
    "rare": "Found in fewest modules",
    "common": "Found in most modules",
    "name": "Name (A to Z)",
}
MODULE_SCOPES = {
    "all": "All skills",
    "only": "Only in this module",
    "shared": "Shared with at least one other",
    "everywhere": "In every module",
    "missing": "Missing here",
    "demand_missing": "Asked for in jobs, missing here",
}
MISSING_SORTS = {
    "tf": "Highest score",
    "jobs": "Highest job demand",
    "rare": "Found in fewest modules",
    "common": "Found in most modules",
    "name": "Name (A to Z)",
}
PER_PAGE = 50


def job_share_of(ds, names):
    al = ds.get("align")
    if al is None:
        return pd.Series(0.0, index=names)
    return al["job_share"].reindex(names).fillna(0.0)


def module_table(ds, name, sort, scope, q):
    """All skills of one module with the numbers needed to analyse them."""
    long = ds["long"]
    contain = ds["contain_col"]
    ccol = ds["count_col"]
    sub = long[long[ds["group_col"]] == name].copy()
    M = ds["n_groups"]
    if q:
        sub = sub[sub["canonical_skill_label"].str.contains(q, case=False, regex=False)]
    if scope == "only":
        sub = sub[sub[contain] == 1]
    elif scope == "shared":
        sub = sub[sub[contain] >= 2]
    elif scope == "everywhere":
        sub = sub[sub[contain] == M]
    sub["lift_sort"] = sub["lift"].where(sub[contain] >= 2, -1.0)
    sub["job_share"] = job_share_of(ds, sub["canonical_skill_label"]).values
    if sort == "tf_idf":
        sub = sub.sort_values(["tf_idf", "tf"], ascending=False)
    elif sort == "lift":
        sub = sub.sort_values(["lift_sort", ccol], ascending=False)
    elif sort == "jobs" and ds.get("align") is not None:
        sub = sub.sort_values(["job_share", "tf"], ascending=False)
    elif sort == "rare":
        sub = sub.sort_values([contain, "tf"], ascending=[True, False])
    elif sort == "common":
        sub = sub.sort_values([contain, "tf"], ascending=[False, False])
    elif sort == "name":
        sub = sub.sort_values("canonical_skill_label", key=lambda c: c.str.lower())
    else:
        sub = sub.sort_values(["tf", "canonical_skill_label"], ascending=[False, True])
    return sub


def module_missing(ds, name, sort, q, demand_only=False):
    """Skills absent from this module. 'missing' = found in other modules; 'demand_missing' = asked for in the job text."""
    present = set(ds["long"].loc[ds["long"][ds["group_col"]] == name, "canonical_skill_label"])
    over = ds["over"]
    al = ds.get("align")
    names = set(over.index)
    if demand_only and al is not None:
        names |= set(al.index[al["job_mentions"] > 0])
    names = sorted(n for n in names if n not in present)
    miss = pd.DataFrame(index=pd.Index(names, name="canonical_skill_label"))
    miss["contain"] = over[ds["contain_col"]].reindex(miss.index).fillna(0).astype(int)
    miss["share"] = over["share"].reindex(miss.index).fillna(0.0)
    miss["score"] = over["score"].reindex(miss.index).fillna(0.0)
    miss["job_share"] = job_share_of(ds, miss.index).values
    if demand_only:
        miss = miss[miss["job_share"] > 0]
    else:
        miss = miss[miss["contain"] > 0]
    if q:
        miss = miss[miss.index.str.contains(q, case=False, regex=False)]
    if sort == "name":
        miss = miss.sort_index(key=lambda c: c.str.lower())
    elif sort == "jobs" and al is not None:
        miss = miss.sort_values(["job_share", "score"], ascending=False)
    elif sort == "rare":
        miss = miss.sort_values(["contain", "score"], ascending=[True, False])
    elif sort == "common":
        miss = miss.sort_values(["contain", "score"], ascending=[False, False])
    elif demand_only and al is not None:
        miss = miss.sort_values(["job_share", "score"], ascending=False)
    else:
        miss = miss.sort_values("score", ascending=False)
    return miss


def holders_for(ds, skill):
    if skill not in ds["pivots"]["count"].index:
        return []
    cnt = ds["pivots"]["count"].loc[skill]
    return [g for g in ds["groups"] if cnt[g] > 0]


def others_for(ds, skill, me):
    return [g for g in holders_for(ds, skill) if g != me]


@app.route("/module")
def module():
    ctx, ds = context()
    if ds is None or "error" in ds:
        return render_template("empty.html", error=(ds or {}).get("error"), **ctx), 200
    name = request.args.get("name", "")
    if name not in ds["groups"]:
        abort(404)
    has_align = ctx["has_align"]
    scopes = {k: v for k, v in MODULE_SCOPES.items() if k != "demand_missing" or has_align}
    sort = request.args.get("sort", "tf")
    scope = request.args.get("scope", "all")
    scope = scope if scope in scopes else "all"
    missing_mode = scope in ("missing", "demand_missing")
    sorts = {k: v for k, v in (MISSING_SORTS if missing_mode else MODULE_SORTS).items() if k != "jobs" or has_align}
    sort = sort if sort in sorts else "tf"
    q = request.args.get("q", "").strip()
    page = _clamp(request.args.get("page"), 1, 100000, 1)

    long = ds["long"]
    contain = ds["contain_col"]
    ccol = ds["count_col"]
    M = ds["n_groups"]
    whole = long[long[ds["group_col"]] == name]
    stats = {"mentions": int(ds["sizes"][name]), "skills": len(whole),
             "only_here": int((whole[contain] == 1).sum()),
             "everywhere": int((whole[contain] == M).sum()),
             "absent": ds["n_skills"] - len(whole),
             "coverage": 100 * len(whole) / ds["n_skills"] if ds["n_skills"] else 0,
             "demand_missing": len(module_missing(ds, name, "tf", "", True)) if has_align else 0}
    if has_align:
        ag = ds["align_groups"].loc[name]
        stats["demand_covered"] = 100 * float(ag["share_of_job_demand_covered"])
        stats["in_jobs"] = 100 * float(ag["share_of_mentions_in_jobs"])

    if missing_mode:
        sub = module_missing(ds, name, sort, q, demand_only=(scope == "demand_missing"))
    else:
        sub = module_table(ds, name, sort, scope, q)
    total = len(sub)
    pages = max(1, -(-total // PER_PAGE))
    page = min(page, pages)
    max_tf = float(whole["tf"].max()) if len(whole) else 1.0
    chunk = sub.iloc[(page - 1) * PER_PAGE: page * PER_PAGE]
    rows = []
    if missing_mode:
        for skill_name, x in chunk.iterrows():
            holders = holders_for(ds, skill_name)
            rows.append({
                "name": skill_name, "url": url_for("skill", name=skill_name, level=ctx["level"]),
                "contain": int(x["contain"]), "share": float(x["share"]), "score": float(x["score"]),
                "job_share": float(x["job_share"]),
                "others": holders[:6], "more": max(0, len(holders) - 6), "all_others": ", ".join(holders),
                "other_urls": [url_for("module", name=g, level=ctx["level"]) for g in holders[:6]],
            })
    else:
        for _, x in chunk.iterrows():
            skill_name = x["canonical_skill_label"]
            others = others_for(ds, skill_name, name)
            rows.append({
                "name": skill_name, "url": url_for("skill", name=skill_name, level=ctx["level"]),
                "count": int(x[ccol]), "tf": float(x["tf"]), "width": 100 * float(x["tf"]) / max_tf,
                "tf_idf": float(x["tf_idf"]), "contain": int(x[contain]), "job_share": float(x["job_share"]),
                "lift": float(x["lift"]) if x[contain] >= 2 else None,
                "emphasised": bool(x[contain] >= 2 and x["lift"] >= 2 and x[ccol] >= 2),
                "others": others[:5], "more": max(0, len(others) - 5), "all_others": ", ".join(others),
                "other_urls": [url_for("module", name=g, level=ctx["level"]) for g in others[:5]],
            })

    form = {"name": name, "level": ctx["level"], "sort": sort, "scope": scope, "q": q}

    def page_url(p):
        return url_for("module", **{**form, "page": p})

    idx = ds["groups"].index(name)
    prev_url = url_for("module", name=ds["groups"][idx - 1], level=ctx["level"]) if idx > 0 else None
    next_url = url_for("module", name=ds["groups"][idx + 1], level=ctx["level"]) if idx < M - 1 else None
    return render_template(
        "module.html", ds=ds, mod=name, rows=rows, stats=stats, form=form, total=total, missing_mode=missing_mode,
        demand_mode=(scope == "demand_missing"),
        missing_url=url_for("module", name=name, level=ctx["level"], scope="missing"),
        demand_url=url_for("module", name=name, level=ctx["level"], scope="demand_missing", sort="jobs"),
        present_url=url_for("module", name=name, level=ctx["level"]),
        page=page, pages=pages, per_page=PER_PAGE, page_url=page_url,
        prev_url=prev_url, next_url=next_url, sorts=sorts, scopes=scopes,
        csv_url=url_for("module_csv", **form), back=url_for("modules_index", level=ctx["level"]), **ctx)


@app.route("/module.csv")
def module_csv():
    ctx, ds = context()
    if ds is None or "error" in ds:
        abort(404)
    name = request.args.get("name", "")
    if name not in ds["groups"]:
        abort(404)
    sort = request.args.get("sort", "tf")
    scope = request.args.get("scope", "all")
    scope = scope if scope in MODULE_SCOPES else "all"
    q = request.args.get("q", "").strip()
    contain = ds["contain_col"]
    ccol = ds["count_col"]
    plural = ctx["words"][2]
    if scope in ("missing", "demand_missing"):
        miss = module_missing(ds, name, sort, q, demand_only=(scope == "demand_missing"))
        out = pd.DataFrame({
            "skill": miss.index, "average_tf": miss["share"].values, "job_share": miss["job_share"].values,
            f"{plural}_containing_skill": miss["contain"].values,
            f"{plural}_with_skill": [" | ".join(holders_for(ds, k)) for k in miss.index],
        })
        fname = f"skills_missing_from_{name}.csv"
    else:
        sub = module_table(ds, name, sort, scope, q)
        out = pd.DataFrame({
            "skill": sub["canonical_skill_label"],
            "mentions": sub[ccol], "tf": sub["tf"], "lift": sub["lift"].where(sub[contain] >= 2),
            "tf_idf": sub["tf_idf"], "job_share": sub["job_share"],
            f"{plural}_containing_skill": sub[contain],
            f"other_{plural}": [" | ".join(others_for(ds, k, name)) for k in sub["canonical_skill_label"]],
        })
        fname = f"skills_in_{name}.csv"
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.route("/modules")
def modules_index():
    ctx, ds = context()
    if ds is None or "error" in ds:
        return render_template("empty.html", error=(ds or {}).get("error"), **ctx), 200
    long = ds["long"]
    contain = ds["contain_col"]
    rows = []
    for g in ds["groups"]:
        sub = long[long[ds["group_col"]] == g]
        top = sub.nlargest(3, "tf")["canonical_skill_label"].tolist()
        fit = ds["align_groups"].loc[g] if ds.get("align_groups") is not None else None
        rows.append({"name": g, "url": url_for("module", name=g, level=ctx["level"]),
                     "demand": (100 * float(fit["share_of_job_demand_covered"]) if fit is not None else None),
                     "mentions": int(ds["sizes"][g]), "skills": len(sub),
                     "coverage": 100 * len(sub) / ds["n_skills"] if ds["n_skills"] else 0,
                     "only": int((sub[contain] == 1).sum()), "top": top})
    return render_template("modules.html", ds=ds, rows=rows, **ctx)


ALIGN_STATUS = {
    "all": "All skills",
    "aligned": "In jobs and taught",
    "in_demand_not_taught": "In jobs, not taught",
    "taught_not_in_jobs": "Taught, not in the job text",
}
ALIGN_SORTS = {
    "gap": "Largest gap",
    "demand": "Highest job demand",
    "coverage": "Taught in most modules",
    "name": "Name (A to Z)",
}


def align_select(ds, status, sort, q):
    al = ds["align"].copy()
    al = al[al["status"] != "none"]
    if status in ("aligned", "in_demand_not_taught", "taught_not_in_jobs"):
        al = al[al["status"] == status]
    if q:
        al = al[al.index.str.contains(q, case=False, regex=False)]
    if sort == "demand":
        al = al.sort_values(["job_share", "course_coverage"], ascending=[False, True])
    elif sort == "coverage":
        al = al.sort_values(["course_coverage", "job_share"], ascending=False)
    elif sort == "name":
        al = al.sort_index(key=lambda c: c.str.lower())
    else:
        al = al.sort_values(["gap_score", "job_mentions"], ascending=False)
    return al


@app.route("/alignment")
def alignment():
    ctx, ds = context()
    if ds is None or "error" in ds:
        return render_template("empty.html", error=(ds or {}).get("error"), **ctx), 200
    if not ctx["has_align"]:
        return render_template("alignment.html", ds=ds, ready=False, **ctx)
    al_all = ds["align"]
    status = request.args.get("status", "in_demand_not_taught")
    status = status if status in ALIGN_STATUS else "in_demand_not_taught"
    sort = request.args.get("sort", "gap")
    sort = sort if sort in ALIGN_SORTS else "gap"
    q = request.args.get("q", "").strip()
    page = _clamp(request.args.get("page"), 1, 100000, 1)

    in_jobs = al_all["job_mentions"] > 0
    taught = al_all[ds["contain_col"]] > 0
    covered = float(al_all.loc[in_jobs & taught, "job_share"].sum())
    course_total = int(al_all["course_mentions"].sum())
    course_in_jobs = int(al_all.loc[in_jobs, "course_mentions"].sum())
    summary = {
        "job_total": ds["job_total"], "job_chunks": ds["job_chunk_total"],
        "job_skills": int(in_jobs.sum()),
        "covered": 100 * covered, "uncovered": 100 * (1 - covered),
        "course_total": course_total,
        "course_in_jobs": 100 * course_in_jobs / course_total if course_total else 0.0,
        "n": {k: int((al_all["status"] == k).sum()) for k in ("aligned", "in_demand_not_taught", "taught_not_in_jobs")},
    }

    sub = align_select(ds, status, sort, q)
    total = len(sub)
    pages = max(1, -(-total // PER_PAGE))
    page = min(page, pages)
    chunk = sub.iloc[(page - 1) * PER_PAGE: page * PER_PAGE]
    counts = ds["pivots"]["count"].reindex(chunk.index).fillna(0)
    max_demand = float(al_all["job_share"].max()) or 1.0
    rows = []
    for skill_name, x in chunk.iterrows():
        strip = [{"g": g, "n": int(counts.at[skill_name, g]),
                  "url": url_for("module", name=g, level=ctx["level"])} for g in ds["groups"]]
        rows.append({
            "name": skill_name, "url": url_for("skill", name=skill_name, level=ctx["level"]),
            "job_mentions": int(x["job_mentions"]), "job_share": float(x["job_share"]),
            "job_chunks": int(x["job_chunks"]) if "job_chunks" in x.index else None,
            "demand_w": 100 * float(x["job_share"]) / max_demand,
            "taught_in": int(x[ds["contain_col"]]), "strip": strip,
            "gap": float(x["gap_score"]), "status": str(x["status"]),
        })

    fits = []
    for g, x in ds["align_groups"].iterrows():
        fits.append({"name": g, "url": url_for("module", name=g, level=ctx["level"]),
                     "mentions": int(x["total_mentions"]), "present": int(x["skills_present"]),
                     "in_jobs": int(x["skills_in_jobs"]),
                     "covered": 100 * float(x["share_of_job_demand_covered"]),
                     "relevant": 100 * float(x["share_of_mentions_in_jobs"])})
    form = {"level": ctx["level"], "status": status, "sort": sort, "q": q}

    def page_url(p):
        return url_for("alignment", **{**form, "page": p})

    def status_url(st):
        return url_for("alignment", level=ctx["level"], status=st, sort=sort, q=q)

    return render_template(
        "alignment.html", ready=True, ds=ds, summary=summary, rows=rows, fits=fits, form=form,
        total=total, page=page, pages=pages, per_page=PER_PAGE, page_url=page_url, status_url=status_url,
        statuses=ALIGN_STATUS, sorts=ALIGN_SORTS, csv_url=url_for("alignment_csv", **form), **ctx)


@app.route("/alignment.csv")
def alignment_csv():
    ctx, ds = context()
    if ds is None or "error" in ds or not ctx["has_align"]:
        abort(404)
    status = request.args.get("status", "all")
    status = status if status in ALIGN_STATUS else "all"
    sort = request.args.get("sort", "gap")
    sort = sort if sort in ALIGN_SORTS else "gap"
    sub = align_select(ds, status, sort, request.args.get("q", "").strip())
    buf = io.StringIO()
    sub.to_csv(buf)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=job_alignment.csv"})



@app.route("/clusters")
def clusters():
    ctx, ds = context()
    path = os.path.join(RESULTS_DIR, "cleaned_skill_labels_only.csv")
    module_path = _paths("module")[0]
    if not os.path.isfile(path):
        return render_template("empty.html", error="Cluster results are unavailable.", **ctx)
    frame = pd.read_csv(path).fillna("")
    modules = pd.read_csv(module_path) if os.path.isfile(module_path) else pd.DataFrame()
    options = sorted(modules.iloc[:, 0].astype(str).unique()) if not modules.empty else []
    module = request.args.get("module", "")
    selected_modules = list(dict.fromkeys(request.args.getlist("module")))
    selected_modules = [m for m in selected_modules if m]
    if any(m not in options for m in selected_modules):
        abort(404)
    if selected_modules:
        names = modules.loc[modules.iloc[:, 0].astype(str).isin(selected_modules), "canonical_skill_label"]
        frame = frame[frame["canonical_skill_label"].isin(names)]
    q = request.args.get("q", "").strip()
    if q:
        frame = frame[frame["canonical_skill_label"].str.contains(q, case=False, regex=False) |
                      frame["synonym_variations"].str.contains(q, case=False, regex=False)]
    frame = frame.sort_values("canonical_skill_label", key=lambda col: col.str.lower())
    rows = [{"id": row.cluster_id, "name": row.canonical_skill_label,
             "mentions": row.total_occurrences,
             "synonyms": [v.strip() for v in str(row.synonym_variations).split(" | ") if v.strip()]}
            for row in frame.itertuples()]
    # Scope every plotted count to module material and the active filters.
    visible = {row["name"] for row in rows}
    scoped = modules[modules["canonical_skill_label"].isin(visible)].copy()
    if selected_modules:
        scoped = scoped[scoped.iloc[:, 0].astype(str).isin(selected_modules)]
    totals = scoped.groupby("canonical_skill_label")["skill_count_in_module"].sum().sort_values(ascending=False)
    chart_order = request.args.get("order", "high")
    chart_order = chart_order if chart_order in ("high", "low") else "high"
    totals = totals.sort_values(ascending=chart_order == "low", kind="stable")
    top = request.args.get("top", "20")
    top = top if top in ("20", "25", "50", "100", "all") else "20"
    ranked = totals if top == "all" else totals.head(int(top))
    if request.args.get("download") == "csv":
        export = ranked.rename("module_mentions").to_frame()
        denominator = float(scoped.groupby(scoped.columns[0])["total_skills_in_module"].first().sum())
        export["combined_tf"] = export["module_mentions"] / denominator if denominator else 0
        export["modules_containing_skill"] = scoped.groupby("canonical_skill_label")[scoped.columns[0]].nunique().reindex(export.index)
        return Response(export.to_csv(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=selected_module_clusters.csv"})
    max_count = float(ranked.max()) if len(ranked) else 1.0
    bars = [{"name": name, "count": int(count), "width": 100 * float(count) / max_count}
            for name, count in ranked.items()]
    columns = selected_modules if selected_modules else options
    matrix = scoped.pivot_table(index="canonical_skill_label", columns=scoped.columns[0],
                                values="tf", aggfunc="sum", fill_value=0)
    peak = float(matrix.reindex(index=ranked.index, columns=columns, fill_value=0).max().max()) if len(ranked) else 0
    heat = [{"name": name, "cells": [{"module": col,
             "tf": float(matrix.at[name, col]) if col in matrix.columns else 0,
             "alpha": (float(matrix.at[name, col]) / peak if col in matrix.columns and peak else 0)}
             for col in columns]} for name in ranked.index]
    return render_template("clusters.html", rows=rows, modules=options, module=module, q=q,
                           bars=bars, heat=heat, heat_columns=columns, selected_modules=selected_modules, top=top, chart_order=chart_order,
                           export_url=url_for("clusters", module=selected_modules, q=q, top=top, order=chart_order, download="csv"),
                           plotted_clusters=len(totals), scoped_mentions=int(totals.sum()), **ctx)

@app.route("/method")
def method():
    ctx, ds = context()
    return render_template("method.html", ds=ds if ds and "error" not in ds else None, **ctx)


@app.errorhandler(404)
def not_found(_):
    ctx, _ds = context()
    return render_template("empty.html", error="That page or skill was not found.", **ctx), 404


@app.template_filter("pct")
def pct(v):
    p = v * 100
    return "<0.1%" if 0 < p < 0.1 else f"{p:.1f}%"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="View TF-IDF results in the browser.")
    ap.add_argument("--results-dir", default=RESULTS_DIR, help="Folder with the tfidf_*.csv files")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    RESULTS_DIR = os.path.abspath(args.results_dir)
    found = available_levels()
    print(f"Results folder: {RESULTS_DIR}")
    print(f"Levels found  : {', '.join(found) if found else 'none yet - run deduplicate.py first'}")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
