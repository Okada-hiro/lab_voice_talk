import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import sys

import numpy as np
import torch
from torch.serialization import add_safe_globals

# 【修正パッチ】PyTorch 2.6対応の決定版
# エラーログで指摘されたクラス(TorchVersion)を「安全リスト」に登録します
try:
    # torch.torch_version.TorchVersion をインポートして許可リストに追加
    from torch.torch_version import TorchVersion
    add_safe_globals([TorchVersion])
except ImportError:
    # 万が一インポートできない場合の予備策として、weights_only=Falseも残しておく
    pass

# 念のための強制上書きパッチも維持（二重の対策）
_original_load = torch.load
def _hack_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)
torch.load = _hack_torch_load

from numpy.typing import NDArray
import torchaudio

# AudioMetaDataがない環境向けのダミークラス定義
if not hasattr(torchaudio, "AudioMetaData"):
    class AudioMetaData:
        def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding
    torchaudio.AudioMetaData = AudioMetaData

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from pyannote.audio import Inference, Model
from tqdm import tqdm

from config import get_config
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT

config = get_config()

# モデルロード
model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")
device = torch.device(config.style_gen_config.device)
inference.to(device)


class NaNValueError(ValueError):
    """カスタム例外クラス。NaN値が見つかった場合に使用されます。"""


def get_style_vector(wav_path: str) -> NDArray[Any]:
    return inference(wav_path)  # type: ignore


def save_style_vector(wav_path: str):
    try:
        style_vec = get_style_vector(wav_path)
    except Exception as e:
        print("\n")
        logger.error(f"Error occurred with file: {wav_path}, Details:\n{e}\n")
        raise
    
    if np.isnan(style_vec).any():
        print("\n")
        logger.warning(f"NaN value found in style vector: {wav_path}")
        raise NaNValueError(f"NaN value found in style vector: {wav_path}")
    np.save(f"{wav_path}.npy", style_vec)


def process_line(line: str):
    wav_path = line.split("|")[0]
    try:
        save_style_vector(wav_path)
        return line, None
    except NaNValueError:
        return line, "nan_error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.style_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.style_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()
    config_path: str = args.config
    num_processes: int = args.num_processes

    hps = HyperParameters.load_from_json(config_path)

    device = config.style_gen_config.device

    training_lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        training_lines.extend(f.readlines())
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        training_results = list(
            tqdm(
                executor.map(process_line, training_lines),
                total=len(training_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )
    ok_training_lines = [line for line, error in training_results if error is None]
    nan_training_lines = [
        line for line, error in training_results if error == "nan_error"
    ]
    if nan_training_lines:
        nan_files = [line.split("|")[0] for line in nan_training_lines]
        logger.warning(
            f"Found NaN value in {len(nan_training_lines)} files: {nan_files}, so they will be deleted from training data."
        )

    val_lines: list[str] = []
    with open(hps.data.validation_files, encoding="utf-8") as f:
        val_lines.extend(f.readlines())

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        val_results = list(
            tqdm(
                executor.map(process_line, val_lines),
                total=len(val_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )
    ok_val_lines = [line for line, error in val_results if error is None]
    nan_val_lines = [line for line, error in val_results if error == "nan_error"]
    if nan_val_lines:
        nan_files = [line.split("|")[0] for line in nan_val_lines]
        logger.warning(
            f"Found NaN value in {len(nan_val_lines)} files: {nan_files}, so they will be deleted from validation data."
        )

    with open(hps.data.training_files, "w", encoding="utf-8") as f:
        f.writelines(ok_training_lines)

    with open(hps.data.validation_files, "w", encoding="utf-8") as f:
        f.writelines(ok_val_lines)

    ok_num = len(ok_training_lines) + len(ok_val_lines)

    logger.info(f"Finished generating style vectors! total: {ok_num} npy files.")