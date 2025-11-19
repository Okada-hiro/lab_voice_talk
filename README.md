# 日本語チャットボット

## 紹介
このレポジトリにあるファイルによってできることは、AIと日本語音声で会話することです。ここにあるファイルで、サーバを立ち上げ、ブラウザで開くとAIと会話をすることができます。タイピングやクリックがほとんど不要で、会話のような体験を実現しています(言い過ぎかもしれませんが)。  
以下の三つのモデルを組み合わせて作ってあります。

1. **whisper_streaming**: 日本語音声からの文字起こし(ASR)を行うモデルです。高速で高精度な文字起こしを行うモデルです。ユーザーからの質問内容を文字起こしします。難しかったため、ストリーミングは使っていません。  
    https://github.com/ufal/whisper_streaming.git

2. **gemini-2.5-flash-mini**: 高速なLLMです。whisper_streamingが文字起こししてできたテキストを入力として受け取り、回答を生成します。

3. **style-bert-vits2**: 高速な日本語音声生成モデルです。gemini-2.5-flash-miniが作った回答テキストをもとにして、日本語音声を生成します。  
    https://github.com/litagin02/Style-Bert-VITS2.git

## 環境
以下はRunPodというGPUが使えてサーバを公開できる環境を想定しています。L4などのpodを作ってください。whisper_streamingが限られたpytorchのバージョンを要求します。具体的には以下の通りです。  
pytorch=2.8.0+cu128, CUDA=12.8, cuDNN=9.10.2/libcudnnn9-cuda-12   
互換性のためにtorchvisionとtorchaudioも同じバージョンにする必要があります。setup_environment.shを実行すれば自動でやってくれます。


## 使い方
RunPodでpodを作り、以下のコマンドを実行してください。webターミナルでも、SSH接続を用いたローカルのターミナルでも大丈夫です。また、予めpodの編集でHTTP PORT 8000を有効にしてください。


1. このレポジトリのクローンをしてください。
```bash
git clone https://okhiro1207@bitbucket.org/concierge_tarou/lab_voice_talk.git
cd lab_voice_talk
```

2. 必要なパッケージのインストールをしてください。
```bash
bash setup_environment.sh
```


3. whisper_stremingのクローンをしてください。
```bash
git clone https://github.com/ufal/whisper_streaming.git
```
4. style_bert_vits2のクローンをしてください。
```bash
git clone https://github.com/litagin02/Style-Bert-VITS2.git
```

5. 名前の変更(-があると扱いづらい)と、__ init __.pyの作成をしてください。
```bash
mv Style-Bert-VITS2 Style_Bert_VITS2
touch Style_Bert_VITS2/__init__.py
```
6. ファインチューニング後のStyle-Bert-VITS2の重みを受け取ってください。
```bash
python tensorfile_import.py
mv Ref_voice/* Style_Bert_VITS2/model_assets
```

7. gemini-2.5-flash-miniを使えるように、APIを設定してください。
```bash
export GOOGLE_API_KEY=your_api_key
```

8. 実行
```bash
python main.py
```
9. ブラウザで、podのPORT8000のURLを検索して、会話を開始することができるはずです。

## 性能

それなりの速度で、それなりの音声認識と音声再生が行われます。

- 速度: ユーザーが話し終えてから大体2秒ぐらいで回答が表示され、4秒ぐらいで音声が再生されます。ただし、1回目の再生にはStyle-Bert-VITS2の起動も含まれるため、遅くなります。  

- 音声認識の正確さ: 現時点で、whisper_streamingの音声認識は人間より劣っています。はっきりと話してあげれば、しっかりと聞き取ってくれます。

- 音声生成の自然さ: 現時点では、ファインチューニングが十分ではないため、生成される日本語音声は、まだギリギリ許容範囲だと感じられるレベルです。  



