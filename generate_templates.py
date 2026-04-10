#!/usr/bin/env python3
"""
generate_templates.py
=====================
Generate deterministic arrhythmic templates and save to templates.csv.

Output schema:
  template_index,pair_index,onset_s,intra_gap_s
"""

import argparse
import csv
import os
import random

# Keep these in sync with cue_experiment.py defaults.
TONE_DURATION = 0.15
ARHYTHM_INTER_MIN = 0.1
ARHYTHM_INTER_MAX = 0.9
ARHYTHM_INTRA_MIN = 0.1
ARHYTHM_INTRA_MAX = 0.9
NUM_TEMPLATES = 20
DEFAULT_BLOCKTIME = 15.0
DEFAULT_SEED = 1

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'templates.csv'
)


def generate_arrhythmic_templates(n: int, blocktime_s: float) -> list[list[tuple[float, float]]]:
    """Create n templates with non-overlapping low/high pairs."""
    templates = []
    tone_dur = TONE_DURATION

    for _ in range(n):
        events = []
        t = random.uniform(0.1, 0.6)
        while True:
            intra_gap = random.uniform(ARHYTHM_INTRA_MIN, ARHYTHM_INTRA_MAX)
            pair_dur = (2 * tone_dur) + intra_gap
            if t + pair_dur > blocktime_s:
                break
            events.append((round(t, 4), round(intra_gap, 4)))
            inter_gap = random.uniform(ARHYTHM_INTER_MIN, ARHYTHM_INTER_MAX)
            t += pair_dur + inter_gap
        templates.append(events)

    return templates


def write_templates_csv(path: str, templates: list[list[tuple[float, float]]]):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['template_index', 'pair_index', 'onset_s', 'intra_gap_s'])
        for template_index, template in enumerate(templates):
            for pair_index, (onset_s, intra_gap_s) in enumerate(template):
                writer.writerow([
                    template_index,
                    pair_index,
                    f"{onset_s:.4f}",
                    f"{intra_gap_s:.4f}",
                ])


def main():
    parser = argparse.ArgumentParser(
        description='Generate arrhythmic templates CSV for cue_experiment.py'
    )
    parser.add_argument('--num-templates', type=int, default=NUM_TEMPLATES)
    parser.add_argument('--blocktime', type=float, default=DEFAULT_BLOCKTIME)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output', type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    templates = generate_arrhythmic_templates(args.num_templates, args.blocktime)
    write_templates_csv(args.output, templates)

    print(f"Saved {args.num_templates} templates to {args.output}")
    for i, tpl in enumerate(templates):
        if tpl:
            print(f"  Template {i:2d}: {len(tpl)} pairs, span {tpl[-1][0]:.2f}s")
        else:
            print(f"  Template {i:2d}: empty")


if __name__ == '__main__':
    main()
