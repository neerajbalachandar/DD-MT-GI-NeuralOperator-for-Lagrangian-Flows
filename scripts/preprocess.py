import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def main():
    parser = argparse.ArgumentParser(description="Preprocess generated FLOWVPM particle and field data.")
    parser.add_argument("--config", type=Path, default=HERE / "configs/default.yaml",
                        help="YAML configuration supplying query budget, domain, and sampling strategy.")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Generated root containing train/, validate/, and test/ (default: configured FLOWVPM data root).")
    parser.add_argument("--merged-frames", type=Path, default=None,
                        help="Existing merged_frames/<case>/<case>__frame_<step>.npz tree; rebuild the final archive from it.")
    parser.add_argument("--raw-root", type=Path, default=None,
                        help="Raw train/validate/test case tree used by merged-frame source paths.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-pairs-per-case", type=int, default=0,
                        help="Optional cap on consecutive transitions per case; default 0 processes every available pair.")
    parser.add_argument("--field-grid-points", type=int, default=None,
                        help="Native cubic field grid side length; applied before importing preprocessing code.")
    parser.add_argument("--max-field-query-points", type=int, default=None,
                        help="Maximum retained Eulerian queries per pair (use 32768 for the full 32^3 grid).")
    parser.add_argument("--case-prefix", type=str, default=None,
                        help="Include only cases whose case.toml case_id starts with this stable prefix.")
    parser.add_argument("--preproc-max-rss-gb", type=float, default=None,
                        help="Optional RSS guard for preprocessing in GB. Lower this for large VPM cases.")
    parser.add_argument("--rollout-memory-cap-gb", type=float, default=None,
                        help="Optional cap on temporary rollout arrays in GB. Lower this to avoid OOM during large-case preprocessing.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Use CUDA for field-gradient preprocessing when available.")
    args = parser.parse_args()
    from gino.utils import load_config
    cfg = load_config(str(args.config))
    import os
    if args.field_grid_points is not None:
        if args.field_grid_points < 2:
            parser.error("--field-grid-points must be at least 2")
        os.environ["FIELD_GRID_POINTS"] = str(args.field_grid_points)
    query_budget = args.max_field_query_points
    if query_budget is None:
        query_budget = int(cfg["data"].get("max_queries", 4096))
    if query_budget is not None:
        if query_budget < 1:
            parser.error("--max-field-query-points must be positive")
        os.environ["MAX_FIELD_QUERY_POINTS"] = str(query_budget)
    if args.case_prefix:
        os.environ["CASE_PREFIX"] = args.case_prefix
    if args.preproc_max_rss_gb is not None:
        os.environ["PREPROC_MAX_RSS_GB"] = str(args.preproc_max_rss_gb)
    if args.rollout_memory_cap_gb is not None:
        os.environ["ROLLOUT_MEMORY_CAP_GB"] = str(args.rollout_memory_cap_gb)
    from gino.data.preprocessing import run_preprocessing, run_preprocessing_from_merged_frames

    if args.merged_frames:
        if args.source_root is not None:
            parser.error("--source-root and --merged-frames are mutually exclusive")
        print(run_preprocessing_from_merged_frames(args.merged_frames, args.output_dir,
                                                   raw_root=args.raw_root, device=args.device,
                                                   field_sampling=cfg["data"].get("field_sampling", {"mode": "regular_grid"}),
                                                   field_domain_mode=cfg["data"].get("field_domain_mode", "full_native_grid"),
                                                   max_queries=query_budget))
    else:
        max_pairs = args.max_pairs_per_case or None
        print(run_preprocessing(args.source_root, args.output_dir,
                                max_pairs_per_case=max_pairs,
                                device=args.device))


if __name__ == "__main__":
    main()
