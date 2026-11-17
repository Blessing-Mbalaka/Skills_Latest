# Ollama Skill Label Deduplicator (Skill Label Data Only)

This tool extracts **only the `skill_label` data** from your dataset, generates semantic vector embeddings using **Sentence Transformers** (`all-MiniLM-L6-v2`), and groups synonymous skill variations using **Agglomerative Cosine Clustering with a Distance Threshold**.

It converts noisy, redundant Ollama-scraped skill labels into a clean, standalone CSV containing unique canonical skill labels, occurrence counts, and consolidated synonym lists.

---

## Included Files

1. `deduplicate.py`: Pipeline: clusters skill labels, then computes TF-IDF and the cross-module score.
2. `app.py`, `templates/`, `static/`: Flask app for viewing the results in the browser.
3. `requirements.txt`: Dependencies (`pandas`, `sentence-transformers`, `scikit-learn`, `numpy`, `openpyxl`, `flask`).
4. `README.md`: Instructions and documentation.

---

## Usage Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
Copy your input file (`.csv` or `.xlsx`) into this folder. It only needs two columns: `course` and `skill_label` (all other columns are ignored). If `skills_long.csv` is not present and there is exactly one data file in the folder, it is used automatically. Otherwise pass the file name:
```bash
python deduplicate.py skills_long.csv
# other options: --course-column course --skill-column skill_label --threshold 0.5 --sheet Sheet1
```

### 3. Output Data
The script creates `cleaned_skill_labels_only.csv` with the following schema:
* `canonical_skill_label`: The primary consolidated skill name.
* `total_occurrences`: Total count of occurrences across all synonym variations.
* `cluster_id`: Unique numerical cluster index.
* `synonym_variations`: Pipe-separated list (`|`) of all raw variations merged into this canonical label.

---

## TF-IDF (skill presence per module)

After clustering, every **canonical skill** is scored per module, so synonyms count as one skill.

```
TF     = (times the skill appears in the module) / (total skill mentions in that module)
IDF    = ln(total number of modules / number of modules containing the skill)
TF-IDF = TF * IDF
```

### What is a "module"?
The `course` column is not granular enough: the value `sample` holds 9 different modules. The module code (e.g. `252BCI9X01`) is therefore parsed from the `relative_path` column (the `Course_<code>_...` folder). If your file already has a column named `module`, that is used instead. Rows with no module (job postings) are excluded from TF-IDF but still take part in clustering.

Columns used: `skill_label`, `course`, and `relative_path` (or `module`). All other columns are ignored.

### Output files
| File | Contents |
|---|---|
| `tfidf_by_module_skill.csv` | one row per (module, skill): count, TF, share, lift, modules containing the skill, IDF, TF-IDF |
| `tfidf_overall_module.csv` | one row per skill across all modules: share, coverage, IDF, smoothed IDF, **score**, mean/max TF-IDF, pooled TF |
| `tfidf_matrix_module.csv` | module x skill grid of TF-IDF (0 = skill absent from that module) |
| `tfidf_by_course_skill.csv`, `tfidf_overall_course.csv`, `tfidf_matrix_course.csv` | the same three files using the `course` column |

### Reading the scores
* High TF-IDF: prominent in that module and rare across other modules.
* A skill present in **every** module has IDF = 0, so its TF-IDF is 0 even though it is present. Use `skill_count_in_module`, `tf` or the coverage columns to see presence regardless of rarity.

Options: `--module-column`, `--module-source-column`, `--course-column`, `--skill-column`, `--threshold`, `--sheet`.

## Comparing skills across all modules

TF-IDF alone is not the right tool for ranking skills against each other, so the "overall" file now carries a score built for that. With M modules:

```
TF(s,m)     = count(s,m) / N(m)                 N(m) = skill mentions in module m
Share(s)    = (1/M) * sum over m of TF(s,m)     average TF, 0 where the skill is absent
Coverage(s) = df(s) / M                          df(s) = modules containing the skill
IDF(s)      = ln((1 + M) / (1 + df(s))) + 1     smoothed, never 0
Score(s)    = Share(s) * IDF(s)
Lift(s,m)   = TF(s,m) / Share(s)                 above 1: module m emphasises the skill more than average
```

* **Share** gives every module equal weight, so a large module cannot dominate (modules here range from 53 to 834 mentions).
* **Smoothed IDF** keeps a skill that appears in every module at weight 1, so widely taught skills still rank.
* `mean_tf_idf` in the overall file now divides by **all** M modules (absent modules count as 0). Earlier versions averaged only over modules where the skill appears, which overstated niche skills. `overall_tf` is the pooled version and is kept for reference only.

Per-module files gained two columns, `share` and `lift`. Overall files gained `share`, `idf_smooth` and `score`, and are sorted by `score`.

## Courses against the job market

Rows marked as job rows (default: `source_type = job`) are used to measure what the job text asks for, using the **same canonical skills** as the courses, so synonyms line up across both sides. For each skill:

```
job_share       = job mentions of the skill / all job mentions
job_prevalence  = job text chunks containing the skill / all job chunks      (needs a chunk_id column)
course_coverage = modules containing the skill / M                          (breadth of presence, not depth)
gap_score       = job_share * (1 - course_coverage)                         (0 for skills not in the job text)
status          = aligned | in_demand_not_taught | taught_not_in_jobs
```

Per module:
* `share_of_job_demand_covered` = sum of `job_share` over the skills the module has.
* `share_of_mentions_in_jobs` = the module's mentions of skills that also appear in the job text / the module's mentions.

Overall: the share of job mentions that are on skills at least one module has.

New files per level (`module`, `course`): `alignment_skills_<level>.csv` and `alignment_groups_<level>.csv`.

Options: `--source-column source_type`, `--job-value job`, `--chunk-column chunk_id`.

**Read it as a first comparison.** The result is only as good as the job text and the extraction behind it: check that the job text is relevant to the programme, that it holds enough separate postings (postings in one text file are not separated), and that clusters merge true synonyms only, because job and course rows are clustered together and a wrong merge creates a false match. "Taught" means present in at least one module, not covered in depth.

## Viewing the results (Flask app)

```bash
pip install -r requirements.txt
python deduplicate.py skills_long.csv      # writes the CSV results
python app.py                              # then open http://127.0.0.1:5000
```

Options: `--results-dir <folder>` if the CSVs are elsewhere, `--port`, `--host`. Re-run `deduplicate.py` and reload the page to see new results.

Pages:
* **Grid** (home; with job data it also has a Jobs column and a "Job demand" sort): skills as rows, modules as columns. The default view is **presence**: a filled cell means the skill is present in that module (the number is how many times it was mentioned), a hatched cell means it was not found. You can also switch the cells to TF, lift or TF-IDF, order rows by score, average TF, coverage or name, search skills, and set a minimum number of modules. "Download these rows as CSV" exports the current view.
* **Skill page**: the modules where the skill is present (with mention counts) and the modules where it is not, the score worked out step by step, every module compared (TF, mentions, lift, TF-IDF), and the wording that was merged into the skill.
* **Job market fit** (`/alignment`): how much of the job demand the modules cover and how much of the course content appears in the job text, a module-by-module table of both, and a skill-by-skill list split into in jobs and not taught, in jobs and taught, and taught but not in the job text. Each skill shows its job demand, how many job chunks mention it, a strip of which modules have it, and its gap score. Search, sort and download as CSV.
* **Each module** (`/modules`): one row per module with its mentions, how many of all skills it has (skills present), how many are unique to it, and its three largest-share skills. Start here to analyse one module.
* **Module page**: every skill in the module in one table, with its share of the module (TF), mentions, lift, TF-IDF and which other modules also have it (each a link). Search skills, filter (all / only in this module / shared / in every module / **missing here**: skills found in other modules but absent from this one / **asked for in jobs, missing here**), order by largest share, most distinctive, highest lift, found in fewest or most modules, or name. It pages through long lists, has previous, next and jump-to controls, and "Download this list as CSV".
* **How the scores work**: the formulas and what to read with care.

If the course-level files exist, a Courses switch appears. Results written by an older `deduplicate.py` are detected and the app asks you to re-run the pipeline.


## Branded standalone Flask viewer

No Django installation or server is required. CurriculumMatch branding and the expandable sidebar are copied from Jobs Word 2 Vec; its older navigation is preserved in a template comment.

Run from this directory:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000/new-analysis or http://127.0.0.1:5000/clusters.
Clusters uses the existing cluster IDs and synonym variations in cleaned_skill_labels_only.csv. Course membership comes from tfidf_by_course_skill.csv. Filtering never recomputes clusters or changes the CSVs. Synonyms and mention counts remain global cluster values.

The analysis views, detail links and CSV downloads remain available. Job market status buttons now retain the search filter.
