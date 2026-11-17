import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

def normalize_text(text):
    """
    Standardizes phrase casing and delimiters for clean vector comparison.
    Converts snake_case, camelCase, hyphens, and multi-spaces into unified lowercase text.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[_\-]', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_EXTS = ('.csv', '.xlsx', '.xlsm', '.xls')
OUTPUT_PREFIXES = ('cleaned_', 'tfidf_', 'skill_idf_')

def resolve_input_path(path):
    """
    Find the input file. Tries the path as given, then next to this script.
    If still not found, falls back to the only .csv/.xlsx file in the script's
    folder (ignoring this pipeline's own output files).
    """
    for candidate in (path, os.path.join(SCRIPT_DIR, path)):
        if os.path.isfile(candidate):
            return candidate

    found = sorted(
        f for f in os.listdir(SCRIPT_DIR)
        if f.lower().endswith(INPUT_EXTS) and not f.lower().startswith(OUTPUT_PREFIXES)
    )
    if len(found) == 1:
        print(f"'{path}' not found - using the only data file in the folder: {found[0]}")
        return os.path.join(SCRIPT_DIR, found[0])

    hint = f"Data files found next to the script: {found}" if found else "No .csv/.xlsx file found next to the script."
    raise FileNotFoundError(
        f"Input file '{path}' not found.\n{hint}\n"
        f"Copy your data file into: {SCRIPT_DIR}\n"
        f"or run:  python deduplicate.py \"full\\path\\to\\your_file.csv\""
    )

def load_table(path, sheet_name=0, usecols=None):
    """Read a .csv or Excel (.xlsx/.xlsm/.xls) file into a DataFrame (optionally only some columns)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xlsm', '.xls'):
        return pd.read_excel(path, sheet_name=sheet_name, usecols=usecols)
    return pd.read_csv(path, usecols=usecols)

COURSE_COLUMN_CANDIDATES = [
    'course_id', 'course', 'course_name', 'course_title', 'course_code',
    'courseid', 'coursename', 'title', 'document_id', 'doc_id', 'id',
]

def detect_course_column(df, label_column):
    """Guess which column identifies the course (case-insensitive)."""
    lowered = {c.lower(): c for c in df.columns if c != label_column}
    for cand in COURSE_COLUMN_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    return None

MODULE_CODE_PATTERN = r'Course_(\d{3}[A-Z]{3}\d[A-Z]\d{2})'

def read_columns(path, sheet_name=0):
    """Return just the column names of a .csv / Excel file (cheap header read)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xlsm', '.xls'):
        return list(pd.read_excel(path, sheet_name=sheet_name, nrows=0).columns)
    return list(pd.read_csv(path, nrows=0).columns)

def extract_module(series):
    """
    Pull the module code (e.g. 252BCI9X01) out of a path such as
    'course_sample\\Course_252BCI9X01_BLOCK_CHAIN..._.1542779\\...'.
    Rows with no match (e.g. job postings) become NaN.
    """
    return series.astype(str).str.extract(MODULE_CODE_PATTERN, expand=False)

def compute_tfidf(
    df, raw_to_canonical, label_column, group_column, level,
    tfidf_output_path, overall_output_path, matrix_output_path=None,
):
    """
    Computes TF-IDF where each `group_column` value (a course or a module) is one document.

    PER GROUP (file 1):
      TF      = (times the skill appears in the group) / (total skill mentions in that group)
      IDF     = ln(total number of groups / number of groups containing the skill)
      TF-IDF  = TF * IDF
      share   = average TF of the skill over ALL groups (0 where the skill is absent)
      lift    = TF / share   (above 1: this group gives the skill more than the average group)

    OVERALL, one row per skill, for comparing skills across all groups (file 2):
      share      = (1/M) * sum over groups of TF           (every group counts equally)
      coverage   = groups containing the skill / M
      idf_smooth = ln((1 + M) / (1 + df)) + 1              (never 0, so skills present everywhere still count)
      score      = share * idf_smooth
      mean_tf_idf = sum of TF-IDF over groups / M          (absent groups count as 0)
      overall_tf  = pooled count / pooled mentions         (large groups dominate; kept for reference)

    Skills are counted at the canonical level, so synonyms merged by the
    clustering step count as the same skill.
    """
    data = df[[group_column, label_column]].copy()
    data[label_column] = data[label_column].fillna("").astype(str)
    data = data[data[label_column] != ""]
    n_before = len(data)
    data = data.dropna(subset=[group_column])
    if len(data) < n_before:
        print(f"Note: {n_before - len(data)} rows have no '{group_column}' value and are excluded from the {level}-level TF-IDF.")
    data['canonical_skill_label'] = data[label_column].map(raw_to_canonical)

    total_groups = data[group_column].nunique()
    total_mentions = len(data)

    count_col = f'skill_count_in_{level}'
    total_col = f'total_skills_in_{level}'
    containing_col = f'{level}s_containing_skill'

    # ---- per group ----
    counts = (
        data.groupby([group_column, 'canonical_skill_label'])
        .size()
        .reset_index(name=count_col)
    )
    counts[total_col] = counts.groupby(group_column)[count_col].transform('sum')
    counts['tf'] = counts[count_col] / counts[total_col]

    groups_with_skill = (
        counts.groupby('canonical_skill_label')[group_column]
        .nunique()
        .rename(containing_col)
    )
    counts = counts.merge(groups_with_skill, on='canonical_skill_label')
    counts['idf'] = np.log(total_groups / counts[containing_col])
    counts['tf_idf'] = counts['tf'] * counts['idf']

    # share = mean TF over all groups (absent groups contribute 0); lift = TF / share
    share_by_skill = counts.groupby('canonical_skill_label')['tf'].sum() / total_groups
    counts['share'] = counts['canonical_skill_label'].map(share_by_skill)
    counts['lift'] = counts['tf'] / counts['share']

    counts = counts.sort_values([group_column, 'tf_idf'], ascending=[True, False])
    out_cols = [group_column, 'canonical_skill_label', count_col, total_col,
                'tf', 'share', 'lift', containing_col, 'idf', 'tf_idf']
    counts[out_cols].to_csv(tfidf_output_path, index=False)

    # ---- overall: one comparable score per skill across all groups ----
    overall = (
        counts.groupby('canonical_skill_label')
        .agg(
            total_occurrences=(count_col, 'sum'),
            **{containing_col: (containing_col, 'first')},
            idf=('idf', 'first'),
            tf_idf_sum=('tf_idf', 'sum'),
            max_tf_idf=('tf_idf', 'max'),
        )
        .reset_index()
    )
    overall['share'] = overall['canonical_skill_label'].map(share_by_skill)
    overall['overall_tf'] = overall['total_occurrences'] / total_mentions
    overall['idf_smooth'] = np.log((1 + total_groups) / (1 + overall[containing_col])) + 1
    overall['score'] = overall['share'] * overall['idf_smooth']
    overall['overall_tf_idf'] = overall['overall_tf'] * overall['idf']
    overall['mean_tf_idf'] = overall['tf_idf_sum'] / total_groups   # divides by ALL groups
    overall[f'{level}_coverage_pct'] = 100 * overall[containing_col] / total_groups
    overall = overall[['canonical_skill_label', 'total_occurrences', 'share', containing_col,
                       f'{level}_coverage_pct', 'idf', 'idf_smooth', 'score',
                       'overall_tf', 'overall_tf_idf', 'mean_tf_idf', 'max_tf_idf']]
    overall = overall.sort_values('score', ascending=False)
    overall.to_csv(overall_output_path, index=False)

    if matrix_output_path:
        matrix = counts.pivot(index=group_column, columns='canonical_skill_label', values='tf_idf').fillna(0)
        matrix.to_csv(matrix_output_path)

    print(f"[{level}] {total_groups} {level}s | {len(overall)} skills | {total_mentions} skill mentions")
    print(f"   per-{level} TF-IDF -> {tfidf_output_path}")
    print(f"   overall TF-IDF     -> {overall_output_path}")
    if matrix_output_path:
        print(f"   {level} x skill matrix -> {matrix_output_path}")
    return counts

def compute_alignment(
    job_df, raw_to_canonical, label_column, chunk_column, counts, group_column, level,
    skills_output_path, groups_output_path,
):
    """
    Compares what the courses carry with what the job text asks for, per canonical skill.

    JOB SIDE (demand):
      job_share      = job mentions of the skill / all job mentions
      job_prevalence = job text chunks containing the skill / all job chunks (if a chunk column exists)
    COURSE SIDE (supply), over the M groups (modules or courses):
      course_coverage = groups containing the skill / M      (breadth of presence, not depth)
      course_share    = average TF over all M groups
    ALIGNMENT:
      status    = aligned | in_demand_not_taught | taught_not_in_jobs
      gap_score = job_share * (1 - course_coverage), for skills that appear in the job text
      share of job demand covered = sum of job_share over skills present in at least one group

    PER GROUP:
      share_of_job_demand_covered = sum of job_share over the skills the group contains
      share_of_mentions_in_jobs   = group mentions of skills that also appear in jobs / group mentions
    """
    count_col = f'skill_count_in_{level}'
    total_col = f'total_skills_in_{level}'
    containing_col = f'{level}s_containing_skill'

    cols = [label_column] + ([chunk_column] if chunk_column and chunk_column in job_df.columns else [])
    jobs = job_df[cols].copy()
    jobs[label_column] = jobs[label_column].fillna("").astype(str)
    jobs = jobs[jobs[label_column] != ""]
    jobs['canonical_skill_label'] = jobs[label_column].map(raw_to_canonical)
    job_total = len(jobs)
    if job_total == 0:
        print("No job skill rows found - skipping alignment.")
        return None

    job_mentions = jobs.groupby('canonical_skill_label').size().rename('job_mentions')
    parts = [job_mentions]
    has_chunks = chunk_column in jobs.columns
    if has_chunks:
        job_chunk_total = jobs[chunk_column].nunique()
        parts.append(jobs.groupby('canonical_skill_label')[chunk_column].nunique().rename('job_chunks'))

    n_groups = counts[group_column].nunique()
    course = counts.groupby('canonical_skill_label').agg(
        course_mentions=(count_col, 'sum'),
        **{containing_col: (count_col, 'size')},
        tf_sum=('tf', 'sum'),
    )
    skills = pd.concat(parts + [course], axis=1).fillna(0)
    skills['job_mentions'] = skills['job_mentions'].astype(int)
    skills['course_mentions'] = skills['course_mentions'].astype(int)
    skills[containing_col] = skills[containing_col].astype(int)
    skills['job_share'] = skills['job_mentions'] / job_total
    if has_chunks:
        skills['job_chunks'] = skills['job_chunks'].astype(int)
        skills['job_prevalence'] = skills['job_chunks'] / job_chunk_total
    skills['course_coverage'] = skills[containing_col] / n_groups
    skills['course_share'] = skills['tf_sum'] / n_groups
    in_jobs = skills['job_mentions'] > 0
    in_course = skills[containing_col] > 0
    skills['status'] = np.select(
        [in_jobs & in_course, in_jobs & ~in_course, ~in_jobs & in_course],
        ['aligned', 'in_demand_not_taught', 'taught_not_in_jobs'], default='none')
    skills['gap_score'] = np.where(in_jobs, skills['job_share'] * (1 - skills['course_coverage']), 0.0)

    order = ['job_mentions'] + (['job_chunks'] if has_chunks else []) + ['job_share'] + \
            (['job_prevalence'] if has_chunks else []) + \
            ['course_mentions', containing_col, 'course_coverage', 'course_share', 'status', 'gap_score']
    skills = skills[order].sort_values(['gap_score', 'job_mentions'], ascending=False)
    skills.index.name = 'canonical_skill_label'
    skills.to_csv(skills_output_path)

    # per group
    c = counts[[group_column, 'canonical_skill_label', count_col, total_col]].copy()
    c['job_share'] = c['canonical_skill_label'].map(skills['job_share']).fillna(0.0)
    c['in_jobs'] = c['job_share'] > 0
    c['mentions_in_jobs'] = c[count_col].where(c['in_jobs'], 0)
    groups = c.groupby(group_column).agg(
        total_mentions=(total_col, 'first'),
        skills_present=('canonical_skill_label', 'size'),
        skills_in_jobs=('in_jobs', 'sum'),
        share_of_job_demand_covered=('job_share', 'sum'),
        mentions_in_jobs=('mentions_in_jobs', 'sum'),
    ).reset_index()
    groups['share_of_mentions_in_jobs'] = groups['mentions_in_jobs'] / groups['total_mentions']
    groups = groups.drop(columns='mentions_in_jobs').sort_values('share_of_job_demand_covered', ascending=False)
    groups.to_csv(groups_output_path, index=False)

    covered = skills.loc[in_jobs & in_course, 'job_share'].sum()
    print(f"[{level}] alignment: {job_total} job mentions over {int(in_jobs.sum())} skills; "
          f"{100 * covered:.1f}% of job mentions are on skills present in at least one {level}.")
    print(f"   skills       -> {skills_output_path}")
    print(f"   per {level}    -> {groups_output_path}")
    return skills

def deduplicate_skill_labels_only(
    input_path='skills_long.csv', 
    output_path='cleaned_skill_labels_only.csv', 
    target_column='skill_label', 
    course_column='course',
    module_column='module',
    module_source_column='relative_path',
    source_column='source_type',
    job_value='job',
    chunk_column='chunk_id',
    sheet_name=0,
    distance_threshold=0.1,
    model_name='all-MiniLM-L6-v2'
):
    """
    Reads dataset, extracts ONLY the skill_label column, computes semantic embeddings,
    clusters synonymous variations using distance thresholding, and exports a standalone
    clean list of unique canonical skill labels with variations and frequency counts.
    """
    input_path = resolve_input_path(input_path)
    available = read_columns(input_path, sheet_name=sheet_name)
    required = [target_column] + ([course_column] if course_column else [])
    missing = [c for c in required if c not in available]
    if missing:
        raise ValueError(f"Column(s) {missing} not found in file. Available columns: {available}")

    # Optional columns used to work out the module (only loaded if present)
    optional = [c for c in (module_column, module_source_column, source_column, chunk_column) if c and c in available]
    wanted = list(dict.fromkeys(required + optional))
    print(f"Loading columns {wanted} from {input_path}...")
    df = load_table(input_path, sheet_name=sheet_name, usecols=wanted)

    # Module = a column named `module_column` if the file has one, otherwise parsed from the path column
    if module_column and module_column in df.columns:
        module_column_name = module_column
    elif module_source_column and module_source_column in df.columns:
        df['module'] = extract_module(df[module_source_column])
        module_column_name = 'module'
    else:
        module_column_name = None

    # Extract strictly the target skill_label column data
    raw_labels = df[target_column].fillna("").astype(str)
    unique_raw_labels = raw_labels.unique()
    
    print(f"Extracted {len(raw_labels)} total skill label entries with {len(unique_raw_labels)} unique variations.")

    # Normalize text strings before embedding calculation
    normalized_map = {raw: normalize_text(raw) for raw in unique_raw_labels}
    normalized_unique_labels = [normalized_map[raw] for raw in unique_raw_labels]

    print(f"Generating vector embeddings using '{model_name}'...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(normalized_unique_labels, show_progress_bar=True, batch_size=256)

    print(f"Clustering skill_label synonyms (Distance Threshold = {distance_threshold}, Cosine Metric)...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(embeddings)

    # Calculate frequencies across the dataset
    frequency_map = raw_labels.value_counts().to_dict()
    
    mapping_df = pd.DataFrame({
        'raw_skill_label': unique_raw_labels,
        'normalized_label': normalized_unique_labels,
        'cluster_id': cluster_labels
    })
    mapping_df['frequency'] = mapping_df['raw_skill_label'].map(frequency_map)

    # Determine canonical phrase per cluster (highest frequency first)
    cluster_to_canonical = {}
    for cluster_id, group in mapping_df.groupby('cluster_id'):
        sorted_group = group.sort_values(
            by=['frequency', 'normalized_label'],
            ascending=[False, True]
        )
        canonical_label = sorted_group.iloc[0]['raw_skill_label']
        cluster_to_canonical[cluster_id] = canonical_label

    mapping_df['canonical_skill_label'] = mapping_df['cluster_id'].map(cluster_to_canonical)

    # Group strictly by canonical skill label to output clean deduplicated skill_label dataset
    clean_skills_df = mapping_df.groupby(['cluster_id', 'canonical_skill_label']).agg(
        total_occurrences=('frequency', 'sum'),
        synonym_variations=('raw_skill_label', lambda x: " | ".join(sorted(list(set(x)))))
    ).reset_index().sort_values(by='total_occurrences', ascending=False)

    # ------------------------------------------------------------------
    # TF-IDF on canonical skills (synonyms already merged by clustering)
    # Run once per course and once per module; each also gets an "overall" file.
    # ------------------------------------------------------------------
    raw_to_canonical = dict(zip(mapping_df['raw_skill_label'], mapping_df['canonical_skill_label']))
    levels = []
    if course_column:
        levels.append((course_column, 'course'))
    if module_column_name:
        levels.append((module_column_name, 'module'))
    else:
        print("WARNING: no module column and no path column to derive it from - skipping module-level TF-IDF.")

    # Job rows (for the course vs job-market comparison)
    job_df = None
    if source_column and source_column in df.columns:
        job_df = df[df[source_column].astype(str).str.lower() == str(job_value).lower()]
        if job_df.empty:
            job_df = None
    if job_df is None:
        print(f"Note: no rows with {source_column} = '{job_value}' found - skipping the course vs job comparison.")

    for group_col, level in levels:
        print(f"Computing TF-IDF per {level} using column '{group_col}'...")
        counts = compute_tfidf(
            df=df,
            raw_to_canonical=raw_to_canonical,
            label_column=target_column,
            group_column=group_col,
            level=level,
            tfidf_output_path=f'tfidf_by_{level}_skill.csv',
            overall_output_path=f'tfidf_overall_{level}.csv',
            matrix_output_path=f'tfidf_matrix_{level}.csv',
        )
        if job_df is not None:
            compute_alignment(
                job_df=job_df,
                raw_to_canonical=raw_to_canonical,
                label_column=target_column,
                chunk_column=chunk_column,
                counts=counts,
                group_column=group_col,
                level=level,
                skills_output_path=f'alignment_skills_{level}.csv',
                groups_output_path=f'alignment_groups_{level}.csv',
            )

    num_canonical_unique = len(clean_skills_df)
    reduction = ((len(unique_raw_labels) - num_canonical_unique) / len(unique_raw_labels)) * 100

    print("\n==========================================")
    print(" SKILL_LABEL DEDUPLICATION SUMMARY ")
    print("==========================================")
    print(f"Original Unique skill_labels : {len(unique_raw_labels)}")
    print(f"Cleaned Canonical Skill Labels: {num_canonical_unique}")
    print(f"Total Synonym Reduction       : {reduction:.2f}%")
    print("==========================================\n")

    # Export clean standalone skill_label dataset
    output_cols = ['canonical_skill_label', 'total_occurrences', 'cluster_id', 'synonym_variations']
    clean_skills_df[output_cols].to_csv(output_path, index=False)
    print(f"Saved cleaned skill_label dataset to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate skill labels and compute TF-IDF per course.")
    parser.add_argument("input", nargs="?", default="skills_long.csv",
                        help="Input .csv or .xlsx file (default: skills_long.csv)")
    parser.add_argument("--course-column", default="course", help="Course column name (default: course)")
    parser.add_argument("--module-column", default="module",
                        help="Module column name if the file has one (default: module). "
                             "If absent, the module code is parsed from --module-source-column")
    parser.add_argument("--module-source-column", default="relative_path",
                        help="Column holding the file path that contains 'Course_<code>' (default: relative_path)")
    parser.add_argument("--source-column", default="source_type",
                        help="Column telling job rows from course rows (default: source_type)")
    parser.add_argument("--job-value", default="job", help="Value of the source column that marks job rows (default: job)")
    parser.add_argument("--chunk-column", default="chunk_id",
                        help="Column identifying the text chunk of a row, used for job prevalence (default: chunk_id)")
    parser.add_argument("--skill-column", default="skill_label", help="Skill column name (default: skill_label)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Cosine distance threshold (default: 0.50)")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index (default: first sheet)")
    args = parser.parse_args()

    deduplicate_skill_labels_only(
        input_path=args.input,
        output_path='cleaned_skill_labels_only.csv',
        target_column=args.skill_column,
        course_column=args.course_column,
        module_column=args.module_column,
        module_source_column=args.module_source_column,
        source_column=args.source_column,
        job_value=args.job_value,
        chunk_column=args.chunk_column,
        sheet_name=args.sheet,
        distance_threshold=args.threshold,
    )
