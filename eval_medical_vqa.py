import argparse
import json
import os
import re
import sys
import tqdm
from pathlib import Path

DATASETS_BASE = "/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/datasets"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_deepseek_client = None

def judge_by_llm(raw_output, item, gt_label, api_key):
    """Ask DeepSeek whether the model's raw output is correct."""
    global _deepseek_client
    if _deepseek_client is None:
        from openai import OpenAI
        _deepseek_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    options = {}
    for letter, key in [('A', 'option_A'), ('B', 'option_B'),
                         ('C', 'option_C'), ('D', 'option_D')]:
        if key in item:
            options[letter] = item[key]
    if not options:
        for letter, key in [('A', 'Choice A'), ('B', 'Choice B'),
                             ('C', 'Choice C'), ('D', 'Choice D')]:
            if key in item:
                val = re.sub(r'^\s*[ABCD]:\s*', '', item[key]).strip()
                options[letter] = val

    opts_str = '\n'.join('(%s) %s' % (l, t) for l, t in options.items())
    correct_text = options.get(gt_label, '')
    prompt = (
        "A model was shown a medical image and asked a multiple-choice question.\n\n"
        "Options:\n%s\n\n"
        "Correct answer: (%s) %s\n\n"
        "Model's response: \"%s\"\n\n"
        "Does the model's response indicate the correct answer? "
        "Reply with only \"yes\" or \"no\"."
    ) % (opts_str, gt_label, correct_text, raw_output)

    try:
        resp = _deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0,
        )
        ans = resp.choices[0].message.content.strip().lower()
        return ans.startswith('y')
    except Exception as e:
        print("\n[LLM judge error] %s" % e)
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['omnimed', 'pmcvqa'])
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--split', default='test')
    parser.add_argument('--vlm_backend', default='llava_med',
                        choices=['llava', 'llava_med', 'med_flamingo', 'chexagent'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='./results/eval')
    parser.add_argument('--max_new_tokens', default=1024, type=int)
    parser.add_argument('--modality_filter', nargs='+', default=None)
    parser.add_argument('--max_samples', default=None, type=int)
    parser.add_argument('--deepseek_api_key', default=None)
    return parser.parse_args()


def load_dataset(args):
    from datasets.medical_vqa import OmniMedVQADataset, PMCVQADataset
    if args.dataset == 'omnimed':
        root = args.data_path or os.path.join(DATASETS_BASE, 'OmniMedVQA/OmniMedVQA')
        return OmniMedVQADataset(root, modality_filter=args.modality_filter)
    else:
        root = args.data_path or os.path.join(DATASETS_BASE, 'PMC-VQA')
        csv_name = 'test_clean.csv' if args.split == 'test' else '%s.csv' % args.split
        return PMCVQADataset(
            csv_file=os.path.join(root, csv_name),
            img_dir=os.path.join(root, 'images'),
        )


def load_model(args):
    if args.vlm_backend == 'llava_med':
        from models.llava_med import LlavaMed
        return LlavaMed(args.device)
    elif args.vlm_backend == 'med_flamingo':
        from models.med_flamingo import MedFlamingo
        return MedFlamingo(args.device)
    elif args.vlm_backend == 'chexagent':
        from models.chexagent import CheXAgent
        return CheXAgent(args.device)
    else:
        from models.llava import Llava
        return Llava(args.device)


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading dataset: %s" % args.dataset)
    ds = load_dataset(args)
    if args.max_samples:
        ds.items = ds.items[:args.max_samples]
    print("Total samples: %d" % len(ds))

    print("Loading VLM: %s" % args.vlm_backend)
    model = load_model(args)

    use_llm_judge = bool(args.deepseek_api_key)
    if use_llm_judge:
        print("DeepSeek judge enabled")

    results = []
    correct_llm = 0
    judge_errors = 0

    for idx in tqdm.tqdm(range(len(ds))):
        image, item = ds[idx]
        prompt = ds.get_prompt(item)
        gt = item.get('_answer_label')

        try:
            raw_output = model.prompt(prompt, image, max_tokens=args.max_new_tokens)[0]
        except Exception as e:
            raw_output = '[ERROR] %s' % e

        llm_correct = None
        if use_llm_judge and gt:
            llm_correct = judge_by_llm(raw_output, item, gt, args.deepseek_api_key)
            if llm_correct is None:
                judge_errors += 1
            elif llm_correct:
                correct_llm += 1

        results.append({
            'idx': idx,
            'question_id': item.get('question_id', str(idx)),
            'gt': gt,
            'raw': raw_output,
            'llm_correct': llm_correct,
        })

    total = len(results)
    judged = sum(1 for r in results if r['llm_correct'] is not None)

    summary = {
        'dataset': args.dataset,
        'vlm_backend': args.vlm_backend,
        'total': total,
        'judged': judged,
        'correct': correct_llm,
        'judge_errors': judge_errors,
        'accuracy': round(correct_llm / total, 4) if total else 0,
        'accuracy_on_judged': round(correct_llm / judged, 4) if judged else 0,
    }
    print("\n=== Results ===")
    for k, v in summary.items():
        print("  %s: %s" % (k, v))

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print("\nResults saved to %s" % args.output_dir)


if __name__ == '__main__':
    main()
