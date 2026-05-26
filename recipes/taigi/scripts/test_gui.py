"""Gradio test GUI for trained YourTTS-Taigi checkpoints.

Spin up a local web UI to quickly audition your model:
    - Pick a checkpoint (auto-finds the latest best_model under runs/)
    - Pick an example sentence (loaded from dev CSVs) OR type your own
    - Pick a training speaker  OR  upload a reference wav for cloning
    - Click 生成 → listen in browser

Usage:
    python recipes/taigi/scripts/test_gui.py
    python recipes/taigi/scripts/test_gui.py --share          # tunnel via gradio.live
    python recipes/taigi/scripts/test_gui.py --port 8080
"""

import argparse
import functools
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = PROJECT_ROOT / "recipes" / "taigi" / "runs"
DATA_DIR = PROJECT_ROOT / "data"

# Dev CSVs we'll mine for example sentences (label, path).
DEV_CSVS = [
    ("CV nan-tw",   DATA_DIR / "common_voice_nan_tw" / "prepared" / "metadata_dev_wav.csv"),
    ("媠聲",         DATA_DIR / "suisiann" / "metadata_suisiann_dev.csv"),
    ("TAT-TTS",     DATA_DIR / "tat_tts" / "metadata_tat_dev.csv"),
    ("MOE leku",    DATA_DIR / "moe_sutian" / "metadata_moe_dev.csv"),
]

# Fallback if no dev CSVs exist yet.
FALLBACK_SENTENCES = [
    "Lí hó, sè-kài.",
    "Tâi-uân ê thinn-khì tsiok hó.",
    "A-bú beh tsia̍h pn̄g.",
    "Guá ài lim ka-pi.",
    "Kin-á-jit lí beh khì tó-uī?",
    "Suann-lōo tsiah kiânn bô guā kú niā-niā, tiunn tuā-tsí tō tshuán phe̍nnh-phe̍nnh.",
]


def _find_checkpoints() -> list[Path]:
    """All best_model*.pth and checkpoint_*.pth under runs/, newest first."""
    if not RUNS_DIR.exists():
        return []
    ckpts: list[Path] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        ckpts.extend(run_dir.glob("best_model*.pth"))
        ckpts.extend(run_dir.glob("checkpoint_*.pth"))
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts


def _ckpt_label(ckpt: Path) -> str:
    """Short display label for a ckpt path."""
    return f"{ckpt.parent.name}/{ckpt.name}"


def _load_example_sentences() -> list[tuple[str, str]]:
    """Return [(label, sentence), ...] sampled from dev CSVs.

    Pulls up to 8 sentences per corpus, tagged with the corpus label.
    """
    samples: list[tuple[str, str]] = []
    for corpus, csv_path in DEV_CSVS:
        if not csv_path.exists():
            continue
        lines = [l for l in csv_path.read_text(encoding="utf-8").splitlines() if l]
        for line in lines[:8]:
            parts = line.split("|")
            if len(parts) >= 2 and parts[1].strip():
                # Truncate display label so dropdown stays compact.
                preview = parts[1][:60] + ("…" if len(parts[1]) > 60 else "")
                samples.append((f"[{corpus}] {preview}", parts[1]))
    if not samples:
        samples = [(s, s) for s in FALLBACK_SENTENCES]
    return samples


@functools.lru_cache(maxsize=2)
def _load_synth(ckpt_path: str, use_cuda: bool):
    """Lazy-load + cache by ckpt path so swapping ckpts re-loads cleanly."""
    from TTS.utils.synthesizer import Synthesizer
    ckpt = Path(ckpt_path)
    run_dir = ckpt.parent
    config = run_dir / "config.json"
    speakers_file = run_dir / "speakers.pth"
    if not config.exists():
        raise FileNotFoundError(f"config.json missing next to {ckpt}")
    return Synthesizer(
        tts_checkpoint=str(ckpt),
        tts_config_path=str(config),
        tts_speakers_file=str(speakers_file) if speakers_file.exists() else "",
        use_cuda=use_cuda,
    )


def _list_speakers(ckpt_path: str) -> list[str]:
    """Read speaker names from the speakers.pth next to ckpt."""
    if not ckpt_path:
        return []
    run_dir = Path(ckpt_path).parent
    speakers_file = run_dir / "speakers.pth"
    if not speakers_file.exists():
        return []
    try:
        import torch
        speakers = torch.load(speakers_file, map_location="cpu", weights_only=False)
        # speakers.pth is usually a dict {speaker_name: d_vector_tensor}
        if isinstance(speakers, dict):
            return sorted(speakers.keys())
    except Exception as exc:
        print(f"warn: could not read speakers from {speakers_file}: {exc}")
    return []


def build_app(use_cuda: bool):
    import gradio as gr

    ckpts = _find_checkpoints()
    ckpt_choices = [_ckpt_label(c) for c in ckpts]
    ckpt_map = {_ckpt_label(c): str(c) for c in ckpts}
    default_ckpt_label = ckpt_choices[0] if ckpt_choices else None

    examples = _load_example_sentences()
    example_choices = [label for label, _ in examples]
    example_map = dict(examples)

    default_speakers = _list_speakers(ckpt_map[default_ckpt_label]) if default_ckpt_label else []
    default_speaker = default_speakers[0] if default_speakers else "taigi_0"

    def synthesize(ckpt_label, sentence, ref_wav_path, mode, speaker_name):
        if not ckpt_label or ckpt_label not in ckpt_map:
            return None, "❌ 沒有可用的 checkpoint。先訓練到出 best_model.pth"
        if not sentence or not sentence.strip():
            return None, "❌ 請輸入或選一句 Tâi-lô"
        if mode == "Voice Clone" and not ref_wav_path:
            return None, "❌ Voice Clone 模式需要上傳參考音檔（3-10 秒）"

        ckpt = ckpt_map[ckpt_label]
        try:
            syn = _load_synth(ckpt, use_cuda=use_cuda)
        except Exception as exc:
            return None, f"❌ 模型載入失敗：{exc}"

        kwargs = {"text": sentence.strip(), "language_name": "nan-tw", "split_sentences": False}
        if mode == "Voice Clone":
            kwargs["speaker_wav"] = ref_wav_path
            tag = "clone"
        else:
            kwargs["speaker_name"] = speaker_name or default_speaker
            tag = speaker_name or default_speaker

        try:
            wav = syn.tts(**kwargs)
        except Exception as exc:
            return None, f"❌ 合成失敗：{exc}"

        sr = syn.output_sample_rate
        # Gradio audio expects (sr, np.ndarray)
        import numpy as np
        wav_np = np.asarray(wav, dtype=np.float32)
        msg = f"✅ {tag} | {len(wav_np)/sr:.2f}s | ckpt={Path(ckpt).name}"
        return (sr, wav_np), msg

    def on_ckpt_change(ckpt_label):
        speakers = _list_speakers(ckpt_map.get(ckpt_label, ""))
        if not speakers:
            return gr.update(choices=[], value=None, info="此 ckpt 沒有 speakers.pth → 只能用 Voice Clone 模式")
        return gr.update(choices=speakers, value=speakers[0], info=f"{len(speakers)} 位訓練語者")

    def on_example_pick(label):
        return example_map.get(label, "")

    with gr.Blocks(title="Taigi TTS 測試", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🗣️ Taigi YourTTS 測試介面")
        gr.Markdown(
            "選 checkpoint → 選/打 Tâi-lô 例句 → 選訓練語者**或**上傳參考音檔 → 生成。"
        )

        with gr.Row():
            ckpt_dd = gr.Dropdown(
                choices=ckpt_choices,
                value=default_ckpt_label,
                label="Checkpoint",
                info=f"自動掃 {RUNS_DIR.relative_to(PROJECT_ROOT)}/* — 最新在最上",
                interactive=True,
            )

        with gr.Row():
            with gr.Column(scale=2):
                example_dd = gr.Dropdown(
                    choices=example_choices,
                    label="選例句（從 dev set / fallback 抓）",
                    value=None,
                    interactive=True,
                )
                text_box = gr.Textbox(
                    label="Tâi-lô 文本（可手打、可從上方下拉選後修改）",
                    placeholder="Lí hó, sè-kài.",
                    lines=3,
                    value=FALLBACK_SENTENCES[0],
                )

            with gr.Column(scale=1):
                mode_radio = gr.Radio(
                    choices=["Speaker ID", "Voice Clone"],
                    value="Speaker ID",
                    label="語者模式",
                )
                speaker_dd = gr.Dropdown(
                    choices=default_speakers,
                    value=default_speaker if default_speakers else None,
                    label="訓練語者",
                    info=f"{len(default_speakers)} 位訓練語者" if default_speakers else "未載入 speakers",
                    interactive=True,
                    visible=True,
                )
                ref_wav = gr.Audio(
                    label="參考音檔（3-10 秒，clone 你想要的聲音）",
                    type="filepath",
                    sources=["upload", "microphone"],
                    visible=False,
                )

        with gr.Row():
            go_btn = gr.Button("🎙️ 生成", variant="primary", size="lg")

        with gr.Row():
            audio_out = gr.Audio(label="輸出", type="numpy", autoplay=True)
        status = gr.Markdown("")

        # Wiring
        example_dd.change(on_example_pick, inputs=example_dd, outputs=text_box)
        ckpt_dd.change(on_ckpt_change, inputs=ckpt_dd, outputs=speaker_dd)

        def on_mode_change(mode):
            return (
                gr.update(visible=(mode == "Speaker ID")),
                gr.update(visible=(mode == "Voice Clone")),
            )

        mode_radio.change(on_mode_change, inputs=mode_radio, outputs=[speaker_dd, ref_wav])

        go_btn.click(
            synthesize,
            inputs=[ckpt_dd, text_box, ref_wav, mode_radio, speaker_dd],
            outputs=[audio_out, status],
        )

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true", help="Expose via gradio.live tunnel")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = p.parse_args()

    try:
        import gradio  # noqa: F401
    except ImportError:
        sys.exit("missing dependency: gradio. Run `pip install gradio` first.")

    try:
        import torch
        use_cuda = torch.cuda.is_available() and not args.cpu
    except ImportError:
        use_cuda = False

    print(f"device: {'cuda' if use_cuda else 'cpu'}")
    print(f"runs:   {RUNS_DIR}")
    print(f"data:   {DATA_DIR}")
    app = build_app(use_cuda=use_cuda)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
