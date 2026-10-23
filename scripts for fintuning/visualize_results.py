"""Build a standalone HTML report from translation experiment metrics CSVs.

QUICK START (from the project root):
    source .venv/bin/activate
    python "scripts for fintuning/visualize_results.py" \
        --input comparison_outputs_clean/metrics.csv \
        --output comparison_outputs_clean/results_visualization.html \
        --language Rajasthani

For another language, change --input, --output, and --language, for example:
    python "scripts for fintuning/visualize_results.py" \
        --input results/odia_metrics.csv --output results/odia_report.html \
        --language Odia

CSV format: one row per model/run, with a model column, recognized metric
columns (for example BLEU and chrF++), and optional language/training settings.
If the CSV has a language column, it labels each run automatically. All columns
are included in the report table, including settings such as epochs, learning
rate, batch size, LoRA rank/alpha/dropout, precision, seed, and dataset sizes.
Unknown metric names can be selected explicitly with --metrics, e.g. --metrics
BLEU chrF++ exact_match. The output is a standalone HTML file that can be shared.
"""

import argparse
import csv
import html
import math
from pathlib import Path


MODEL_COLUMNS = {"model", "modelname", "system", "run", "experiment", "checkpoint"}
LANGUAGE_COLUMNS = {"language", "lang", "sourcelanguage", "targetlanguage"}
METRIC_LABELS = {
    "bleu": "BLEU", "sacrebleu": "SacreBLEU", "chrf": "chrF", "chrfpp": "chrF++",
    "comet": "COMET", "meteor": "METEOR", "ter": "TER", "loss": "Loss",
    "evalloss": "Validation loss", "valloss": "Validation loss",
    "validationloss": "Validation loss", "testloss": "Test loss", "trainloss": "Training loss",
    "runtime": "Runtime", "runtimesec": "Runtime (s)",
    "trainingruntime": "Training runtime", "trainingruntimesec": "Training runtime (s)",
    "trainingtimemin": "Training time (min)", "peakvramgb": "Peak GPU memory (GB)",
    "gpupeakgb": "Peak GPU memory (GB)",
}
LOWER_IS_BETTER = {
    "ter", "loss", "evalloss", "valloss", "validationloss", "testloss", "trainloss",
    "runtime", "runtimesec", "trainingruntime", "trainingruntimesec", "trainingtimemin",
    "peakvramgb", "gpupeakgb",
}
PALETTE = ["#2673a6", "#23936a", "#8054a1", "#d17a22", "#c64b55", "#268c91"]


def compact(text):
    return "".join(ch.lower() for ch in text if ch.isalnum())


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has a header but no data rows")
    return reader.fieldnames, rows


def find_column(headers, choices):
    return next((column for column in headers if compact(column) in choices), None)


def metric_columns(headers, requested):
    if requested:
        lookup = {compact(header): header for header in headers}
        missing = [name for name in requested if compact(name) not in lookup]
        if missing:
            raise ValueError(f"Metric column(s) not found: {', '.join(missing)}")
        return [lookup[compact(name)] for name in requested]

    found = []
    prefixes = ("validation", "eval", "test", "train", "val")
    for header in headers:
        name = compact(header)
        for prefix in prefixes:
            if name.startswith(prefix) and name[len(prefix):] in METRIC_LABELS:
                name = name[len(prefix):]
                break
        if name in METRIC_LABELS:
            found.append(header)
    return found


def metric_name(column):
    name = compact(column)
    if name == "chrf" and "+" in column:
        return "chrF++"
    for prefix, label in (("validation", "Validation "), ("eval", "Validation "),
                          ("test", "Test "), ("train", "Training "), ("val", "Validation ")):
        if name.startswith(prefix) and name[len(prefix):] in METRIC_LABELS:
            return label + METRIC_LABELS[name[len(prefix):]]
    return METRIC_LABELS.get(name, column)


def is_lower_better(column):
    name = compact(column)
    for prefix in ("validation", "eval", "test", "train", "val"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name in LOWER_IS_BETTER


def parse_number(value):
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def run_color(model):
    name = model.lower()
    if any(word in name for word in ("lora", "fine-tun", "finetun", "adapter")):
        return "#23936a"
    if "base" in name or "pretrained" in name:
        return "#2673a6"
    return PALETTE[sum(map(ord, name)) % len(PALETTE)]


def make_chart(column, rows, model_column, language_column, default_language):
    values = []
    for row in rows:
        score = parse_number(row.get(column, ""))
        if score is None:
            continue
        model = (row.get(model_column) or "Unnamed run").strip()
        language = ((row.get(language_column) or default_language) if language_column else default_language).strip()
        label = f"{language} / {model}" if language else model
        values.append((label, model, score))
    if not values:
        return ""

    width, left, right = 1120, 350, 90
    row_height, top, bottom = 30, 56, 42
    height = top + bottom + row_height * len(values)
    plot_width = width - left - right
    low = min(0.0, min(value for _, _, value in values))
    high = max(0.0, max(value for _, _, value in values))
    if math.isclose(low, high):
        high = low + 1
    padding = (high - low) * 0.06
    low -= padding
    high += padding

    def x_coord(value):
        return left + (value - low) / (high - low) * plot_width

    zero = x_coord(0)
    direction = "lower is better" if is_lower_better(column) else "higher is better"
    svg = [f'<section class="chart"><svg viewBox="0 0 {width} {height}" role="img">']
    svg.append(f'<text x="12" y="28" class="chart-title">{html.escape(metric_name(column))} <tspan class="hint">({direction})</tspan></text>')
    for index in range(5):
        tick = low + (high - low) * index / 4
        tx = x_coord(tick)
        svg.append(f'<line x1="{tx:.1f}" y1="{top-8}" x2="{tx:.1f}" y2="{height-bottom+3}" class="grid"/>')
        svg.append(f'<text x="{tx:.1f}" y="{height-12}" text-anchor="middle" class="tick">{tick:.2f}</text>')
    for index, (label, model, value) in enumerate(values):
        y = top + index * row_height
        end = x_coord(value)
        bar_x = min(zero, end)
        bar_width = max(1.0, abs(end-zero))
        svg.append(f'<text x="{left-12}" y="{y+14}" text-anchor="end" class="row">{html.escape(label)}</text>')
        svg.append(f'<rect x="{bar_x:.1f}" y="{y}" width="{bar_width:.1f}" height="18" rx="3" fill="{run_color(model)}"/>')
        value_x = min(width-right+5, end+7) if value >= 0 else max(left-6, end-7)
        anchor = "start" if value >= 0 else "end"
        svg.append(f'<text x="{value_x:.1f}" y="{y+14}" text-anchor="{anchor}" class="value">{value:.3f}</text>')
    svg.append("</svg></section>")
    return "\n".join(svg)


def render(headers, rows, language, title, selected_metrics=None):
    model_column = find_column(headers, MODEL_COLUMNS)
    if not model_column:
        raise ValueError("CSV needs a run/model column named model, model_name, system, run, or experiment")
    language_column = find_column(headers, LANGUAGE_COLUMNS)
    metrics = metric_columns(headers, selected_metrics)
    if not metrics:
        raise ValueError("No recognized metric columns found; specify them with --metrics")

    charts = "\n".join(make_chart(m, rows, model_column, language_column, language) for m in metrics)
    ths = "".join(f"<th>{html.escape(column)}</th>" for column in headers)
    table_rows = []
    for row in rows:
        tds = "".join(f"<td>{html.escape(str(row.get(column) or '—'))}</td>" for column in headers)
        table_rows.append(f"<tr>{tds}</tr>")
    extra_columns = [h for h in headers if compact(h) not in MODEL_COLUMNS | LANGUAGE_COLUMNS and h not in metrics]
    note = ""
    if not extra_columns:
        note = ('<p class="note">This metrics CSV has no run-configuration columns. Add fields such as '
                'language, train/validation/test pair counts, epochs, learning rate, batch size, LoRA '
                'rank/alpha/dropout, precision, seed, runtime, and peak VRAM to compare settings.</p>')
    lang_label = language or "Language(s) are shown per run when provided"
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
body{{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:#19212b;background:#f4f6f8;margin:0;padding:28px}}
main{{max-width:1220px;margin:auto;background:white;padding:30px;border-radius:12px;box-shadow:0 2px 14px #17212b12}}
h1{{font-size:28px;margin:0 0 4px}}h2{{font-size:20px;margin:32px 0 8px}}.sub,.legend{{color:#596574}}
.chart{{overflow-x:auto;margin:16px 0 26px;border:1px solid #e4e8ed;border-radius:8px;padding:8px}}
svg{{width:100%;min-width:780px;height:auto}}.chart-title{{font-size:17px;font-weight:700;fill:#19212b}}
.hint,.tick{{font-size:12px;fill:#647080;font-weight:400}}.grid{{stroke:#e6eaf0;stroke-width:1}}
.row{{font-size:13px;fill:#334155}}.value{{font-size:12px;fill:#263444}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid #e4e8ed;text-align:left;white-space:nowrap}}
th{{background:#f1f4f7}}.note{{background:#fff8df;padding:12px 14px;border-radius:6px;color:#604c14}}
footer{{margin-top:28px;color:#647080;font-size:12px}}@media(max-width:700px){{body{{padding:10px}}main{{padding:16px}}h1{{font-size:23px}}}}
</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="sub">{html.escape(lang_label)} · {len(rows)} runs · {len(metrics)} plotted metrics</p>
<p class="legend">Green bars: LoRA/fine-tuned/adapter. Blue bars: base/pretrained. Other runs get consistent colors.</p>
<h2>Metric comparison</h2>{charts}<h2>Scores and run parameters</h2>{note}
<div class="table-wrap"><table><thead><tr>{ths}</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<footer>When sharing or publishing, include the test split, preprocessing, decoding settings, metric implementation/version, and run configuration.</footer>
</main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="comparison_outputs_clean/metrics.csv", help="CSV with one row per model/run")
    parser.add_argument("--output", default="comparison_outputs_clean/results_visualization.html", help="Standalone HTML report path")
    parser.add_argument("--language", default="", help="Language label if CSV has no language column, e.g. Rajasthani or Odia")
    parser.add_argument("--title", default="Translation Model Comparison")
    parser.add_argument("--metrics", nargs="*", help="Metric column names; known metrics are detected automatically")
    args = parser.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    headers, rows = read_csv(input_path)
    report = render(headers, rows, args.language, args.title, args.metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Created {output_path} from {len(rows)} runs in {input_path}")


if __name__ == "__main__":
    main()
