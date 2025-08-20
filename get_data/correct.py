#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple

from tqdm import tqdm

# ---------------------------
# Small, self-contained utils
# ---------------------------

def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def read_jsonl(path: str) -> List[Dict]:
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            res.append(json.loads(line))
    return res

def append_jsonl(path: str, rows: List[Dict]) -> None:
    ensure_dir(path)
    with open(path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def write_jsonl(path: str, rows: List[Dict]) -> None:
    ensure_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ---------------------------
# Prompt template handling
# ---------------------------

DEFAULT_TEMPLATE = (
    "You are a careful corrector. The user's previous response is below.\n\n"
    "Response:\n{response}\n\n"
    "Constraint to enforce:\n{constraint}\n\n"
    "Rewrite the response to satisfy the constraint. "
    "Return only the revised content, and (optionally) a header like 'Revised Response:' followed by the text."
)

def load_template(args) -> str:
    # 1) external python module (backwards compatible with your original import)
    if args.use_prompt_module:
        try:
            from prompts import corrector_template  # type: ignore
            if isinstance(corrector_template, str) and "{response}" in corrector_template and "{constraint}" in corrector_template:
                return corrector_template
        except Exception:
            pass
    # 2) file path to a template with {response} and {constraint}
    if args.template and os.path.exists(args.template):
        with open(args.template, 'r', encoding='utf-8') as f:
            t = f.read()
        if "{response}" in t and "{constraint}" in t:
            return t
    # 3) built-in fallback
    return DEFAULT_TEMPLATE

def render_prompt(template: str, response: str, constraint: str) -> str:
    return template.format(response=response, constraint=constraint)

# ---------------------------
# Output extraction
# ---------------------------

RE_REVISED = re.compile(r'-?-?-?r?R?evised r?R?esponse-?-?-?:?\n*([\s\S]*)', re.IGNORECASE)

def extract_revised_text(model_out: str) -> str:
    m = RE_REVISED.findall(model_out)
    return (m[0] if m else model_out).strip()

# ---------------------------
# Backends (vLLM / OpenAI)
# ---------------------------

class Generator:
    def generate(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError

class VLLMGenerator(Generator):
    def __init__(self, args):
        from vllm import LLM, SamplingParams  # type: ignore

        # Normalize sampling params to ints (no None for top_k).
        if args.temperature == 0.0:
            # Greedy path: disable sampling knobs explicitly
            effective_top_k = -1
            effective_top_p = 1.0
        else:
            effective_top_k = args.top_k if args.top_k is not None else -1
            # Clamp to valid domain: vLLM expects int >= -1
            if effective_top_k < -1:
                effective_top_k = -1
            effective_top_p = args.top_p

        self.sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=int(args.max_tokens),
            top_p=float(effective_top_p),
            top_k=int(effective_top_k),  # must be an int
        )

        # Some vLLM versions don't support enable_prefix_caching; keep it but you can
        # drop it if you hit a TypeError here.
        self.llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=bool(args.enforce_eager),
            enable_prefix_caching=True,
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.use_chat_template = args.use_chat_template
        self.system_prompt = args.system_prompt

    def _apply_chat_template(self, text: str) -> str:
        if not self.use_chat_template:
            return text
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompts: List[str]) -> List[str]:
        if self.use_chat_template:
            prompts = [self._apply_chat_template(p) for p in prompts]
        outs = self.llm.generate(prompts, self.sampling)
        return [o.outputs[0].text for o in outs]

class OpenAIGenerator(Generator):
    """
    Minimal OpenAI wrapper. Requires `OPENAI_API_KEY` env var or --api_key.
    Uses `gpt-4o-mini` by default (change with --openai_model).
    """
    def __init__(self, args):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("Please `pip install openai` to use --teacher_backend openai.") from e

        api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key. Pass --api_key or set OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)
        self.model = args.openai_model

    def generate(self, prompts: List[str]) -> List[str]:
        # One API call per prompt to keep it simple & predictable
        outs = []
        for p in prompts:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": p}],
                temperature=0.0,
                max_tokens=512,
            )
            outs.append(resp.choices[0].message.content or "")
        return outs

# ---------------------------
# Correction pipeline
# ---------------------------

@dataclass
class SampleState:
    sample_idx: int
    prompt: str
    constraints: List[str]
    failed_indices: List[int]            # indices into constraints that were not followed
    next_pos: int = 0                    # pointer in failed_indices
    chosen: str = ""                     # evolving corrected response
    dpo_rejects: List[Tuple[int, str]] = field(default_factory=list)  # (constraint_index, reject_text)

def build_initial_states(datas: List[Dict]) -> List[SampleState]:
    states: List[SampleState] = []
    for i, data in enumerate(datas):
        prompt = data['prompt']
        constraints = data['constraints']
        followed = data['follow_instruction_list']
        if len(followed) != len(constraints):
            # skip malformed entries
            continue
        failed = [idx for idx, ok in enumerate(followed) if not ok]
        if not failed:
            continue
        st = SampleState(
            sample_idx=i,
            prompt=prompt,
            constraints=constraints,
            failed_indices=failed,
            next_pos=0,
            chosen=data['response'],
            dpo_rejects=[],
        )
        states.append(st)
    return states

def make_prompts_for_batch(states: List[SampleState], template: str, batch_size: int) -> Tuple[List[str], List[int]]:
    """
    Returns:
        prompts: rendered prompts for current step of each state
        owners: indices into `states` for mapping outputs back
    """
    prompts, owners = [], []
    for si, st in enumerate(states):
        if st.next_pos >= len(st.failed_indices):
            continue
        cidx = st.failed_indices[st.next_pos]
        prompts.append(render_prompt(template, st.chosen, st.constraints[cidx]))
        owners.append(si)
        if len(prompts) >= batch_size:
            break
    return prompts, owners

def apply_corrections(states: List[SampleState], owners: List[int], outputs: List[str]) -> None:
    """
    For each owner state, apply the correction, advance pointer,
    and record reject text for DPO.
    """
    for si, out in zip(owners, outputs):
        st = states[si]
        cidx = st.failed_indices[st.next_pos]
        reject_text = st.chosen
        corrected = extract_revised_text(out)
        st.chosen = corrected
        st.dpo_rejects.append((cidx, reject_text))
        st.next_pos += 1

def all_done(states: List[SampleState]) -> bool:
    return all(st.next_pos >= len(st.failed_indices) for st in states)

def flush_finished_states(states: List[SampleState]) -> Tuple[List[SampleState], List[SampleState]]:
    """
    Split states into (finished, unfinished).
    """
    finished, unfinished = [], []
    for st in states:
        if st.next_pos >= len(st.failed_indices):
            finished.append(st)
        else:
            unfinished.append(st)
    return finished, unfinished

# ---------------------------
# End-to-end
# ---------------------------

def process(
    args,
    generate_fn: Generator,
    template: str,
) -> None:
    datas = read_jsonl(args.res_path)

    # Build per-sample correction state machines
    states = build_initial_states(datas)

    # Early exit if nothing to correct
    if not states:
        # Still create empty outputs for consistency
        write_jsonl(args.dpo_data_path, [])
        write_jsonl(args.ift_data_path, [])
        return

    # Streams to collect final lines to write in one go
    dpo_lines: List[Dict] = []
    ift_lines: List[Dict] = []

    # Batch across samples while walking each sample's failed constraint list in order
    pbar = tqdm(total=sum(len(st.failed_indices) for st in states), desc="Correcting", unit="step")

    while states:
        prompts, owners = make_prompts_for_batch(states, template, args.batch_size)
        if not prompts:
            # No ready prompts => some states may be finished
            finished, states = flush_finished_states(states)
            for st in finished:
                # finalize DPO rows: each reject uses the *final* chosen
                final = st.chosen
                for cidx, reject_text in st.dpo_rejects:
                    dpo_lines.append({
                        "prompt": st.prompt,
                        "constraint": st.constraints[cidx],
                        "reject": reject_text,
                        "chosen": final,
                    })
                # IFT row: copy original sample with response replaced
                original = datas[st.sample_idx].copy()
                original["response"] = final
                ift_lines.append(original)
            continue

        # Generate in one go
        outs = generate_fn.generate(prompts)
        apply_corrections(states, owners, outs)
        pbar.update(len(owners))

        # Opportunistic flush of any newly-finished states to keep memory small
        finished, states = flush_finished_states(states)
        for st in finished:
            final = st.chosen
            for cidx, reject_text in st.dpo_rejects:
                dpo_lines.append({
                    "prompt": st.prompt,
                    "constraint": st.constraints[cidx],
                    "reject": reject_text,
                    "chosen": final,
                })
            original = datas[st.sample_idx].copy()
            original["response"] = final
            ift_lines.append(original)

    pbar.close()

    # Optional post-filter: drop rows where reject==chosen (same behavior as before)
    if args.filter_equal_pairs:
        dpo_lines = [r for r in dpo_lines if r["reject"] != r["chosen"]]

    # Write once (fast)
    write_jsonl(args.dpo_data_path, dpo_lines)
    write_jsonl(args.ift_data_path, ift_lines)

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="../data/data_response.jsonl",
                        help="Input jsonl with fields: prompt, constraints, follow_instruction_list, response")
    parser.add_argument("--ift_data_path", type=str, default="../data/train/ift_data.jsonl",
                        help="Output IFT jsonl")
    parser.add_argument("--dpo_data_path", type=str, default="../data/train/dpo_data.jsonl",
                        help="Output DPO jsonl")

    # Backend selection
    parser.add_argument("--teacher_backend", type=str, choices=["vllm", "openai"], default="vllm")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")

    # vLLM configs
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1, help="Use -1 to disable (recommended when temperature=0)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Eager is usually slower; leave off unless debugging")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", type=str, default="")

    # batching / misc
    parser.add_argument("--batch_size", type=int, default=64, help="Number of prompts per vLLM batch")
    parser.add_argument("--filter_equal_pairs", action="store_true",
                        help="Drop DPO rows where reject == chosen")
    parser.add_argument("--use_prompt_module", action="store_true",
                        help="Import prompts.corrector_template if available")
    parser.add_argument("--template", type=str, default="",
                        help="Path to a template file containing {response} and {constraint}")

    args = parser.parse_args()

    template = load_template(args)

    # Build generator
    if args.teacher_backend == "vllm":
        generator = VLLMGenerator(args)
    else:
        generator = OpenAIGenerator(args)

    # Run pipeline
    process(args, generator, template)

if __name__ == "__main__":
    main()
