import argparse
import shutil
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Optional

import soundfile as sf
import torch
import torchaudio
# 【修正パッチ】削除された機能をニセモノで復活させてエラーを防ぐ
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
    
from tqdm import tqdm

from config import get_path_config
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


def is_audio_file(file: Path) -> bool:
    supported_extensions = [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"]
    return file.suffix.lower() in supported_extensions


def get_stamps(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    min_silence_dur_ms: int = 700,
    min_sec: float = 2,
    max_sec: float = 12,
):
    """
    min_silence_dur_ms: int (ミリ秒):
        このミリ秒数以上を無音だと判断する。
        逆に、この秒数以下の無音区間では区切られない。
    """

    (get_speech_timestamps, _, read_audio, *_) = utils
    sampling_rate = 16000  # 16kHzか8kHzのみ対応

    min_ms = int(min_sec * 1000)

    wav = read_audio(str(audio_file), sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sampling_rate,
        min_silence_duration_ms=min_silence_dur_ms,
        min_speech_duration_ms=min_ms,
        max_speech_duration_s=max_sec,
    )

    return speech_timestamps


def split_wav(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    target_dir: Path,
    min_sec: float = 2,
    max_sec: float = 12,
    min_silence_dur_ms: int = 700,
    time_suffix: bool = False,
    suffix: str = "", # 識別用サフィックスを追加
) -> tuple[float, int]:
    margin: int = 200  # ミリ秒単位で、音声の前後に余裕を持たせる
    speech_timestamps = get_stamps(
        vad_model=vad_model,
        utils=utils,
        audio_file=audio_file,
        min_silence_dur_ms=min_silence_dur_ms,
        min_sec=min_sec,
        max_sec=max_sec,
    )

    data, sr = sf.read(audio_file)

    total_ms = len(data) / sr * 1000

    file_name = audio_file.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    total_time_ms: float = 0
    count = 0

    # タイムスタンプに従って分割し、ファイルに保存
    for i, ts in enumerate(speech_timestamps):
        start_ms = max(ts["start"] / 16 - margin, 0)
        end_ms = min(ts["end"] / 16 + margin, total_ms)

        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        segment = data[start_sample:end_sample]

        # ファイル名に suffix (-s700 や -s200) を挿入
        if time_suffix:
            file = f"{file_name}{suffix}-{int(start_ms)}-{int(end_ms)}.wav"
        else:
            file = f"{file_name}{suffix}-{i}.wav"
        sf.write(str(target_dir / file), segment, sr)
        total_time_ms += end_ms - start_ms
        count += 1

    return total_time_ms / 1000, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_sec", "-m", type=float, default=2, help="Minimum seconds of a slice"
    )
    parser.add_argument(
        "--max_sec", "-M", type=float, default=12, help="Maximum seconds of a slice"
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="inputs",
        help="Directory of input wav files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The result will be in Data/{model_name}/raw/",
    )
    parser.add_argument(
        "--time_suffix",
        "-t",
        action="store_true",
        help="Make the filename end with -start_ms-end_ms when saving wav.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=3,
        help="Number of processes to use.",
    )
    args = parser.parse_args()

    path_config = get_path_config()
    dataset_root = path_config.dataset_root

    model_name = str(args.model_name)
    input_dir = Path(args.input_dir)
    output_dir = dataset_root / model_name / "raw"
    min_sec: float = args.min_sec
    max_sec: float = args.max_sec
    time_suffix: bool = args.time_suffix
    num_processes: int = args.num_processes

    audio_files = [file for file in input_dir.rglob("*") if is_audio_file(file)]

    logger.info(f"Found {len(audio_files)} audio files.")
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} already exists, deleting...")
        shutil.rmtree(output_dir)

    # モデルをダウンロード
    _ = torch.hub.load(
        repo_or_dir="litagin02/silero-vad",
        model="silero_vad",
        onnx=True,
        trust_repo=True,
    )

    # ワーカーが(Path, int)のタスクを受け取るように変更
    def process_queue(
        q: Queue[Optional[tuple[Path, int]]],
        result_queue: Queue[tuple[float, int]],
        error_queue: Queue[tuple[Path, Exception]],
    ):
        vad_model, utils = torch.hub.load(
            repo_or_dir="litagin02/silero-vad",
            model="silero_vad",
            onnx=True,
            trust_repo=True,
        )
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            
            file, dur = item # ファイルパスと無音時間しきい値(ms)を展開
            try:
                rel_path = file.relative_to(input_dir)
                time_sec, count = split_wav(
                    vad_model=vad_model,
                    utils=utils,
                    audio_file=file,
                    target_dir=output_dir / rel_path.parent,
                    min_sec=min_sec,
                    max_sec=max_sec,
                    min_silence_dur_ms=dur, # 各タスクの持っている値を使用
                    time_suffix=time_suffix,
                    suffix=f"-s{dur}" # 識別用suffixを渡す (-s700, -s200)
                )
                result_queue.put((time_sec, count))
            except Exception as e:
                logger.error(f"Error processing {file} with silence_dur {dur}: {e}")
                error_queue.put((file, e))
                result_queue.put((0, 0))
            finally:
                q.task_done()

    # 700msと200msの2パターンのタスクリストを作成
    tasks = []
    for file in audio_files:
        tasks.append((file, 700))
        tasks.append((file, 200))

    q: Queue[Optional[tuple[Path, int]]] = Queue()
    result_queue: Queue[tuple[float, int]] = Queue()
    error_queue: Queue[tuple[Path, Exception]] = Queue()

    num_processes = min(num_processes, len(tasks))

    threads = [
        Thread(target=process_queue, args=(q, result_queue, error_queue))
        for _ in range(num_processes)
    ]
    for t in threads:
        t.start()

    pbar = tqdm(total=len(tasks), file=SAFE_STDOUT, dynamic_ncols=True)
    for task in tasks:
        q.put(task)

    total_sec = 0
    total_count = 0
    for _ in range(len(tasks)):
        time, count = result_queue.get()
        total_sec += time
        total_count += count
        pbar.update(1)

    q.join()

    for _ in range(num_processes):
        q.put(None)

    for t in threads:
        t.join()

    pbar.close()

    if not error_queue.empty():
        error_str = "Error slicing some files:"
        while not error_queue.empty():
            file, e = error_queue.get()
            error_str += f"\n{file}: {e}"
        raise RuntimeError(error_str)

    logger.info(
        f"Slice done! Total time: {total_sec / 60:.2f} min, {total_count} files."
    )