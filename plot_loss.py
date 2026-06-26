import argparse
import json
from pathlib import Path
from collections import defaultdict


def _load_scalar_json(path: Path):
    """Load TensorBoard-downloaded scalar JSON.

    Supports two common formats:
      1) list of [wall_time, step, value]
      2) list of {"wall_time":..., "step":..., "value":...}
    Returns: steps(list[int]), values(list[float]) sorted by step.
    """
    data = json.loads(path.read_text(encoding='utf-8'))

    steps = []
    values = []

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, list) and len(first) >= 3:
            for row in data:
                if not isinstance(row, list) or len(row) < 3:
                    continue
                steps.append(int(row[1]))
                values.append(float(row[2]))
        elif isinstance(first, dict):
            for row in data:
                if not isinstance(row, dict):
                    continue
                if 'step' not in row or 'value' not in row:
                    continue
                steps.append(int(row['step']))
                values.append(float(row['value']))
        else:
            raise ValueError(f"Unrecognized JSON row format in {path}")
    else:
        raise ValueError(f"Empty or invalid JSON: {path}")

    merged = {}
    for s, v in zip(steps, values):
        merged[int(s)] = float(v)

    steps_sorted = sorted(merged.keys())
    values_sorted = [merged[s] for s in steps_sorted]
    return steps_sorted, values_sorted


def _infer_curve_kind(filename: str):
    low = filename.lower()
    if 'train' in low:
        return 'train'
    if 'val' in low or 'valid' in low:
        return 'val'
    return None


def _infer_model_key(filename: str):
    # Strip known suffixes
    for suf in [
        '_epoch_loss_train.json',
        '_epoch_loss_val.json',
        '_train-loss.json',
        '_val-loss.json',
        '_train_loss.json',
        '_val_loss.json',
        '.json',
    ]:
        if filename.endswith(suf):
            return filename[: -len(suf)]
    return filename.rsplit('.', 1)[0]


def main():
    parser = argparse.ArgumentParser(description='Plot loss curves from TensorBoard-downloaded JSON files')
    parser.add_argument('--loss-dir', type=str, default='./loss', help='Directory containing JSON files')
    parser.add_argument('--output-dir', type=str, default=None, help='Where to save plots (default: loss-dir)')
    parser.add_argument('--format', type=str, default='pdf', choices=['png', 'svg', 'pdf'], help='Output format')
    parser.add_argument('--title-prefix', type=str, default='', help='Title prefix')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model keys to plot (default: auto-detect all model groups)')

    # Plot styling (paper-friendly defaults)
    parser.add_argument('--size', type=float, default=None,
                        help='(Deprecated) Square figure size in inches (width=height). Use --width/--height instead.')
    parser.add_argument('--width', type=float, default=7.0,
                        help='Figure width in inches.')
    parser.add_argument('--height', type=float, default=5.0,
                        help='Figure height in inches (lower = shorter plot).')
    parser.add_argument('--label-size', type=float, default=22.0,
                        help='Font size for axis labels.')
    parser.add_argument('--tick-size', type=float, default=18.0,
                        help='Font size for tick labels.')
    parser.add_argument('--legend-size', type=float, default=20.0,
                        help='Font size for legend text.')
    parser.add_argument('--title-size', type=float, default=22.0,
                        help='Font size for title.')
    parser.add_argument('--show-title', action='store_true',
                        help='Show figure title (default: off).')
    parser.add_argument('--line-width', type=float, default=2.5,
                        help='Line width for curves.')
    parser.add_argument('--y-scale', type=str, default='log', choices=['linear', 'log', 'symlog'],
                        help='Y-axis scale. "log" corresponds to semilogy.')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for raster outputs (png). Ignored by svg/pdf backends.')

    args = parser.parse_args()

    fig_w = float(args.width)
    fig_h = float(args.height)
    if args.size is not None:
        fig_w = fig_h = float(args.size)

    loss_dir = Path(args.loss_dir)
    if not loss_dir.exists():
        raise FileNotFoundError(f"loss-dir not found: {loss_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else loss_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(loss_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {loss_dir}")

    # model_key -> {'train': (steps, values), 'val': (steps, values)}
    curves = defaultdict(dict)

    for p in json_files:
        kind = _infer_curve_kind(p.name)
        if kind is None:
            continue
        model_key = _infer_model_key(p.name)
        steps, values = _load_scalar_json(p)
        curves[model_key][kind] = (steps, values)

    if args.models:
        wanted = [m.strip() for m in args.models.split(',') if m.strip()]
        curves = {k: v for k, v in curves.items() if k in set(wanted)}

    if not curves:
        raise RuntimeError(
            f"No train/val loss curves detected in {loss_dir}. "
            "Expected filenames containing 'train'/'val'."
        )

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    saved = []
    for model_key, kv in sorted(curves.items()):
        if 'train' not in kv or 'val' not in kv:
            # skip incomplete pairs
            continue

        x_tr, y_tr = kv['train']
        x_va, y_va = kv['val']

        # Make x axis 1-based epoch when steps look like epochs
        x_tr_plot = [int(x) + 1 for x in x_tr]
        x_va_plot = [int(x) + 1 for x in x_va]

        title = f"{args.title_prefix}{model_key}" if args.title_prefix else model_key
        out_path = out_dir / f"{model_key}_loss.{args.format}"

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if args.y_scale == 'log':
            min_y = min(min(y_tr), min(y_va))
            if min_y <= 0:
                raise ValueError(
                    f"y-scale=log requires all loss values > 0, but got min={min_y}. "
                    "Try --y-scale symlog or linear."
                )

        ax.plot(x_tr_plot, y_tr, label='train-loss', linewidth=args.line_width)
        ax.plot(x_va_plot, y_va, label='val-loss', linewidth=args.line_width)
        ax.set_yscale(args.y_scale)

        ax.set_xlabel('Epoch', fontsize=args.label_size)
        ax.set_ylabel('Loss', fontsize=args.label_size)
        if args.show_title:
            ax.set_title(title, fontsize=args.title_size)

        ax.tick_params(axis='both', which='both', labelsize=args.tick_size)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend(fontsize=args.legend_size, loc='upper right')

        fig.tight_layout()
        fig.savefig(str(out_path), bbox_inches='tight', dpi=args.dpi)
        plt.close(fig)

        saved.append(out_path)

    if not saved:
        missing = {k: sorted(v.keys()) for k, v in curves.items()}
        raise RuntimeError(f"No complete (train+val) model pairs found. Found: {missing}")

    for p in saved:
        print(f"✓ Saved: {p}")


if __name__ == '__main__':
    main()
