"""YourTTS recipe for Taiwanese Hokkien (Taigi) — single language, multi-speaker.

Data: Common Voice nan-tw 25.0, re-split speaker-stratified via
``recipes/taigi/data/prepare_common_voice.py``.

Text: Tâi-lô (台羅) romanization, NFD-normalized, lowercased.
Tokenizer: TTS.tts.utils.text.tailo.TailoCharacters + tailo_basic_cleaner.

Architecture: VITS + external H/ASP speaker encoder (YourTTS-style).
Language embedding is OFF for now (single language); enable later if we
add 媠聲 or any other locale alongside.

To launch:
    python recipes/taigi/train_yourtts_taigi.py
"""

import os
from pathlib import Path


def _load_dotenv(env_path: Path) -> None:
    """Tiny .env loader (no python-dotenv dependency).

    Shell environment wins over .env so users can still override per-run with
    `$env:TAIGI_BATCH_SIZE=8; python ...`. Quietly no-ops if .env missing.
    """
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _envbool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _envfloat(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _envint(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


# Load .env at the very top so all the constants below can read it.
_load_dotenv(Path(__file__).resolve().parents[2] / ".env")


import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tailo import (
    TAILO_BLANK,
    TAILO_BOS,
    TAILO_CHARACTERS,
    TAILO_EOS,
    TAILO_PAD,
    TAILO_PUNCTUATIONS,
)

torch.set_num_threads(min(8, os.cpu_count() or 1))

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_PATH, "..", ".."))

RUN_NAME = "YourTTS-Taigi-CV25"
OUT_PATH = os.path.join(CURRENT_PATH, "runs")

# All tuneable settings below can be overridden via `.env` at the project
# root or any TAIGI_* shell env var. Defaults match this branch's GPU profile.

# Set True to warm-start from Coqui's official YourTTS multilingual checkpoint
# (pretrained on VCTK / LibriTTS / CSS10 — includes zero-shot cloning).
# For our 11.5 hr Taigi corpus, fine-tuning typically converges 5-10x faster
# than from-scratch and gives better cloning quality.
#
# The text-embedding table is re-initialized (different 46-token Tâi-lô vocab
# vs the original English+IPA set) but everything else warm-starts.
#
# Set False for from-scratch training (slower, but cleaner — useful for
# benchmarking or avoiding any English/Portuguese accent bleed-through).
FINETUNE_FROM_COQUI_YOURTTS = _envbool("TAIGI_FINETUNE", True)

# LR multiplier applied to lr_gen / lr_disc when fine-tuning. Coqui's Trainer
# resets LR to config defaults on restore (i.e. 2e-4 for VITS) — that's the
# "from scratch" LR. Hammering pretrained weights at full LR will catastrophic-
# forget the upstream knowledge in the first few hundred steps. 0.1 (→ 2e-5)
# is the standard fine-tune setting for VITS / YourTTS class models.
# Ignored when FINETUNE_FROM_COQUI_YOURTTS=False.
FINETUNE_LR_SCALE = _envfloat("TAIGI_LR_SCALE", 0.1)

# Minimum expected size (bytes) of the pretrained ckpt — sanity check after
# download. The Coqui YourTTS multilingual model is ~350-700 MB depending on
# version. Anything below ~50 MB is almost certainly a partial download.
FINETUNE_CKPT_MIN_BYTES = 50 * 1024 * 1024

# Explicit checkpoint path; non-None overrides FINETUNE_FROM_COQUI_YOURTTS.
# Use TAIGI_RESTORE_PATH=<path> to set without editing the recipe.
RESTORE_PATH = os.environ.get("TAIGI_RESTORE_PATH") or None

SKIP_TRAIN_EPOCH = False
BATCH_SIZE = _envint("TAIGI_BATCH_SIZE", 16)  # 4060 default: batch=8 → 2.9GB; 16 → ~5.8GB
SAMPLE_RATE = 16000
MAX_AUDIO_LEN_IN_SECONDS = _envint("TAIGI_MAX_AUDIO_LEN", 10)

# Corpus + prepared-metadata paths.
CORPUS_ROOT = os.path.join(
    PROJECT_ROOT, "data", "common_voice_nan_tw",
    "cv-corpus-25.0-2026-03-09", "nan-tw",
)
PREPARED_DIR = os.path.join(
    PROJECT_ROOT, "data", "common_voice_nan_tw", "prepared",
)

cv_taigi_config = BaseDatasetConfig(
    formatter="common_voice_taigi",
    dataset_name="cv_nan_tw_25",
    # Use the WAV-converted metadata (produced by resample_to_wav.py).
    # Loading WAV via soundfile sidesteps torchaudio's torchcodec dep and
    # skips MP3 decode every step. Fall back to metadata_*.csv (MP3) only
    # if you haven't run resample_to_wav.py.
    meta_file_train=os.path.join(PREPARED_DIR, "metadata_train_wav.csv"),
    meta_file_val=os.path.join(PREPARED_DIR, "metadata_dev_wav.csv"),
    path=CORPUS_ROOT,
    language="nan-tw",
)

DATASETS_CONFIG_LIST = [cv_taigi_config]

# Pre-trained Coqui H/ASP speaker encoder (YourTTS-style external embedding).
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"
)

# Project-local cache for pretrained upstream checkpoints.
# Keeps the multi-GB download out of %USERPROFILE%\.local\share\tts and
# on the same drive as the project. Gitignored via the top-level `models/`
# entry in .gitignore.
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "models", "pretrained")


def _verify_checkpoint(path: str) -> None:
    """Sanity-check a downloaded pretrained ckpt before training tries to load it.

    Catches truncated downloads / corrupted files at startup instead of mid-
    epoch when Trainer.restore_model() tries to torch.load() them.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ckpt not found after download: {path}")
    size = os.path.getsize(path)
    if size < FINETUNE_CKPT_MIN_BYTES:
        raise RuntimeError(
            f"ckpt at {path} is only {size / 1024**2:.1f} MB — looks truncated "
            f"(expected >= {FINETUNE_CKPT_MIN_BYTES / 1024**2:.0f} MB). "
            f"Delete the file and re-run to retry."
        )
    print(f"  ckpt sanity ok: {size / 1024**2:.1f} MB")


def _resolve_restore_path() -> str | None:
    """Return the .pth checkpoint path for Trainer's restore_path.

    Precedence: explicit RESTORE_PATH > FINETUNE_FROM_COQUI_YOURTTS auto-fetch
    > None (from scratch).
    """
    if RESTORE_PATH is not None:
        print(f">>> Warm-start from explicit RESTORE_PATH: {RESTORE_PATH}")
        _verify_checkpoint(RESTORE_PATH)
        return RESTORE_PATH
    if FINETUNE_FROM_COQUI_YOURTTS:
        print(">>> Fine-tune mode: fetching Coqui YourTTS multilingual ckpt...")
        os.makedirs(PRETRAINED_DIR, exist_ok=True)
        from TTS.utils.manage import ModelManager
        # ModelManager appends "tts/" to output_prefix, so the file ends up at
        # <PRETRAINED_DIR>/tts/<model_name>/model_file.pth — within the project.
        mm = ModelManager(output_prefix=PRETRAINED_DIR)
        path, _, _ = mm.download_model(
            "tts_models/multilingual/multi-dataset/your_tts"
        )
        print(f">>> Will warm-start from: {path}")
        _verify_checkpoint(path)
        return path
    print(">>> Training from scratch (FINETUNE_FROM_COQUI_YOURTTS=False).")
    return None


def _verify_wav_metadata() -> None:
    """Fail fast if metadata_*_wav.csv mirrors haven't been generated yet.

    The recipe points at WAV-converted metadata. Without running resample_to_wav.py
    first, the dataset formatter would raise an opaque "0 items found" later.
    """
    for split in ("train", "dev"):
        p = os.path.join(PREPARED_DIR, f"metadata_{split}_wav.csv")
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"missing {p}\n"
                f"  Run this first to produce WAV mirrors:\n"
                f"  python recipes/taigi/data/resample_to_wav.py --workers 8"
            )


def _print_mode_banner(mode: str, restore_path: str | None,
                       base_lr: float, eff_lr: float) -> None:
    """Print a hard-to-miss banner stating training mode + key params.

    Modes: 'FINE-TUNE' (auto Coqui pretrained), 'WARM-START' (explicit ckpt),
    'FROM-SCRATCH'.
    """
    bar = "=" * 72
    print()
    print(bar)
    print(f"  TAIGI TTS — {mode} MODE")
    print(bar)
    if restore_path:
        print(f"  warm-start from : {restore_path}")
    if mode == "FINE-TUNE":
        print(f"  lr (gen / disc) : {eff_lr:.2e}  "
              f"(scaled {FINETUNE_LR_SCALE}x from default {base_lr:.2e})")
        print(f"  note            : LR scaled down to protect pretrained weights "
              f"from catastrophic forgetting")
        print(f"  note            : TensorBoard global_step starts from upstream "
              f"ckpt's step (cosmetic only)")
    elif mode == "WARM-START":
        print(f"  lr (gen / disc) : {base_lr:.2e}  (config default, NOT scaled)")
        print(f"  note            : LR not scaled — assumes you're resuming or "
              f"have set FINETUNE_LR_SCALE intentionally")
    else:
        print(f"  lr (gen / disc) : {base_lr:.2e}  (config default)")
    print(f"  RUN_NAME        : {RUN_NAME}")
    print(f"  BATCH_SIZE      : {BATCH_SIZE}")
    print(f"  max audio       : {MAX_AUDIO_LEN_IN_SECONDS} sec @ {SAMPLE_RATE} Hz")
    print(bar)
    print()


def main() -> None:
    # Step 0 — fail fast on missing prerequisites, decide mode, show banner.
    _verify_wav_metadata()
    restore_path = _resolve_restore_path()
    # Auto-fetched Coqui pretrained ⇒ true fine-tune (LR must be scaled down).
    # Explicit RESTORE_PATH ⇒ assume user knows what they're doing (resume or
    # custom warm-start), don't touch LR.
    is_finetune_mode = (restore_path is not None) and (RESTORE_PATH is None)
    if restore_path is None:
        mode_label = "FROM-SCRATCH"
    elif is_finetune_mode:
        mode_label = "FINE-TUNE"
    else:
        mode_label = "WARM-START"
    # VitsConfig.lr_gen default is 2e-4 (verified in TTS/tts/configs/vits_config.py).
    base_lr = 2e-4
    eff_lr = base_lr * FINETUNE_LR_SCALE if is_finetune_mode else base_lr
    _print_mode_banner(mode_label, restore_path, base_lr, eff_lr)

    # Pre-compute speaker embeddings (d-vectors) once per dataset.
    d_vector_files = []
    for dataset_conf in DATASETS_CONFIG_LIST:
        embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
        if not os.path.isfile(embeddings_file):
            print(f">>> Computing speaker embeddings for {dataset_conf.dataset_name}")
            compute_embeddings(
                SPEAKER_ENCODER_CHECKPOINT_PATH,
                SPEAKER_ENCODER_CONFIG_PATH,
                embeddings_file,
                old_speakers_file=None,
                config_dataset_path=None,
                formatter_name=dataset_conf.formatter,
                dataset_name=dataset_conf.dataset_name,
                dataset_path=dataset_conf.path,
                meta_file_train=dataset_conf.meta_file_train,
                meta_file_val=dataset_conf.meta_file_val,
                disable_cuda=not torch.cuda.is_available(),
                no_eval=False,
            )
        d_vector_files.append(embeddings_file)

    audio_config = VitsAudioConfig(
        sample_rate=SAMPLE_RATE,
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0.0,
        mel_fmax=None,
        num_mels=80,
    )

    model_args = VitsArgs(
        d_vector_file=d_vector_files,
        use_d_vector_file=True,
        d_vector_dim=512,
        num_layers_text_encoder=10,
        speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
        speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
        resblock_type_decoder="2",
        # Single language for now; enable when we add 媠聲 (different ortho variant).
        # use_language_embedding=True,
        # embedded_language_dim=4,
        # Speaker-consistency loss tightens cloning fidelity; turn on after warmup.
        # use_speaker_encoder_as_loss=True,
    )

    config = VitsConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name="YourTTS-Taigi",
        run_description="YourTTS on Common Voice nan-tw 25.0 (Tâi-lô, 79 speakers)",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=5000,
        save_n_checkpoints=2,
        save_checkpoints=True,
        target_loss="loss_1",
        print_eval=False,
        use_phonemes=False,
        compute_input_seq_cache=True,
        add_blank=True,
        text_cleaner="tailo_basic_cleaner",
        characters=CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            pad=TAILO_PAD,
            eos=TAILO_EOS,
            bos=TAILO_BOS,
            blank=TAILO_BLANK,
            characters=TAILO_CHARACTERS,
            punctuations=TAILO_PUNCTUATIONS,
            phonemes="",
            is_unique=True,
            is_sorted=False,
        ),
        phoneme_cache_path=None,
        precompute_num_workers=4,
        start_by_longest=True,
        datasets=DATASETS_CONFIG_LIST,
        cudnn_benchmark=False,
        max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
        mixed_precision=True,  # consumer GPU: enable AMP to fit batch
        # Even-out speaker exposure so the heavy-tail spk_0 doesn't dominate.
        use_weighted_sampler=True,
        weighted_sampler_attrs={"speaker_name": 1.0},
        weighted_sampler_multipliers={},
        # Speaker-consistency loss (SCL) coefficient — currently DORMANT because
        # use_speaker_encoder_as_loss=False in model_args (Coqui default). To
        # enable SCL for cloning fidelity, uncomment the flag in VitsArgs above
        # AFTER initial convergence (e.g. epoch 30+); this alpha then takes effect.
        speaker_encoder_loss_alpha=9.0,
        # A handful of held-out cloning_eval speaker IDs are used as reference
        # speakers for periodic test-sentence generation during training.
        test_sentences=[
            ["Lí hó, sè-kài.", "taigi_0", None, "nan-tw"],
            ["Tâi-uân ê thinn-khì tsiok hó.", "taigi_1", None, "nan-tw"],
            ["A-bú beh tsia̍h pn̄g.", "taigi_2", None, "nan-tw"],
            ["Guá ài lim ka-pi.", "taigi_3", None, "nan-tw"],
            ["Kin-á-jit lí beh khì tó-uī?", "taigi_4", None, "nan-tw"],
        ],
    )

    # Load samples and let Coqui split off an eval slice.
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Bake the fine-tune LR override into the config now (must happen BEFORE
    # Trainer construction so Trainer.restore_lr() picks up the scaled value).
    # See _resolve_restore_path() / banner at top of main() for mode logic.
    if is_finetune_mode:
        config.lr_gen *= FINETUNE_LR_SCALE
        config.lr_disc *= FINETUNE_LR_SCALE

    model = Vits.init_from_config(config)

    trainer = Trainer(
        TrainerArgs(restore_path=restore_path, skip_train_epoch=SKIP_TRAIN_EPOCH),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    # Windows uses spawn for DataLoader workers; without this guard each
    # worker re-imports the module and re-enters trainer.fit() recursively.
    import multiprocessing

    multiprocessing.freeze_support()
    main()
