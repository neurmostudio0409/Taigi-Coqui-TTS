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

RUN_NAME = "YourTTS-Taigi-CV25-RTX5090"
OUT_PATH = os.path.join(CURRENT_PATH, "runs")

# Set to a YourTTS multilingual ckpt path to warm-start; None = from scratch.
RESTORE_PATH = None

SKIP_TRAIN_EPOCH = False
BATCH_SIZE = 64  # RTX 5090 32GB VRAM (Blackwell); fp16. batch=32 only used 11.5/31.5GB; 64 fits ~22GB
SAMPLE_RATE = 16000
MAX_AUDIO_LEN_IN_SECONDS = 10

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

def main() -> None:
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
        num_loader_workers=8,
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
        precompute_num_workers=8,
        start_by_longest=True,
        datasets=DATASETS_CONFIG_LIST,
        cudnn_benchmark=False,
        max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
        mixed_precision=True,  # consumer GPU: enable AMP to fit batch
        # Even-out speaker exposure so the heavy-tail spk_0 doesn't dominate.
        use_weighted_sampler=True,
        weighted_sampler_attrs={"speaker_name": 1.0},
        weighted_sampler_multipliers={},
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

    model = Vits.init_from_config(config)

    trainer = Trainer(
        TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
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
