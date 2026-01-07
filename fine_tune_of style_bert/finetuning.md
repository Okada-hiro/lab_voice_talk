1. uvのインストール(必須ではないですが、速くなります。)
```pip install uv```

2. style-bert-vits2のインストール

```git clone https://github.com/litagin02/Style-Bert-VITS2.git```

```cd Style-Bert-VITS2```

3. 必要な PyTorch バージョンを指定してインストール（GPU対応）
\
Colab では通常 CUDA 12.x 環境なので cu118 → cu121 に変更してOK。

```uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```
```pip install torchcodec```

 4. 依存パッケージをまとめてインストール

```uv pip install -r requirements-colab.txt```

 5. 初期化スクリプトを実行（モデルを自動ダウンロード）

```python initialize.py```

6. 音声ファイル(wav)をStyle-Bert-VITS2/inputsというフォルダに入れる

7. 依存関係的に元々定義されているファイルがうまくいかないため、新しく作ったファイルと置き換えてください

置き換えるファイルは以下のとおりです。

・slice.py

・style_gen.py

・bert_jen.py

・train_ms_jp_extra.py

8. 音声ファイルを適切な長さに分割します。

```python slice.py --model_name Ref_voice```

注意: 短文の音声(2秒未満)は丸ごとカットされてしまいます。最短音声認識時間を短くしましょう。
```!python slice.py --model_name Ref_voice --min_sec 1.0 --min_silence_dur_ms 1000```

9. whisperで文字起こしをします

```pip install faster-whisper```

```python transcribe.py --model_name Ref_voice```

10. ファインチューニングの前処理や、epochなどを決めます

```python preprocess_all.py \
    -m Ref_voice \
    --use_jp_extra \
    -b 8 \
    -e 300 \
    --normalize \
    --trim \
    --yomi_error raise\
    --num_processes 4
```
この時点で、whisperの作った文字に読めない文字がある場合はesd.listを修正してください

11. ファインチューニング
```python train_ms_jp_extra.py```


Style-Bert-VITS2/model_assets/Ref_voiceに三つのファイルができています。これで完了です。