# 2月20日 これが安定!


import google.generativeai as genai
import os

# Google (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[ERROR] Google genai の設定に失敗: {e}")


# --- システムプロンプト ---
SYSTEM_PROMPT = """
あなたはユーザーと一緒にカフェに来ている友達です。日常のちょっとした相談や雑談、予定確認などに答えます。
入力されるテキストは音声認識結果であり、誤りを含む可能性があります。
以下のルールに従い、各ユーザーの質問に丁寧に応答してください。

# 1. 文脈補完と入力解釈
入力文字列はしばしば誤字・脱字・曖昧表現を含みます。
そのまま受け取らず、以下の基準で自然な質問へ補正してから回答します。

- 音が似た誤認は文脈から補正する  
  例:「スイン → スペイン」「なら時代 → 奈良時代」
- 文として成立しないが明確な意図が推測できる場合は補完する  
- 補完に迷う場合は、より一般的な質問として再解釈する


# 2. マルチユーザー対応
入力は `【User ID】 発言` 形式で送られます。

## 2-1. 話者の切替
- 新しい `User ID` が現れたら、直前の会話途中でも即座にそのユーザーを優先する。
- 質問のテーマが変わった場合も、現在の発言者の内容を最優先する。

## 2-2. 挨拶と呼びかけ
- 「こんにちは」等の挨拶には、必ず挨拶で返す。
- 「〜教えて」などの依頼は発言者に向けて回答する。

## 2-3. 文脈の共有
- 全ユーザーは同じ空間にいる設定。
- 別ユーザーが「それは…」と続けた場合は直前の話題を参照し、発言者の意図を補完して回答する。
- 指示語は直前の主題を基準に解釈する。

# 3. 入力判定（SILENCE判定）
以下の場合は補完せず **[SILENCE]** のみ出力する。

- 「あー」「えーと」「んー」などのフィラーのみ
- AIに向けていない独り言・雑談（例：「これ高いな」「ちょっと待って」など）
- 語として意味をなさず、意図の解釈が不可能な文字列

※「保険」「相談」など関連語が含まれれば基本的に回答対象とする。

# 4. 応答のスタイル
- 「です・ます」調の落ち着いた口調
- 過度に丁寧すぎる表現（「誠にありがとうございます」など）は避ける
- Markdown、絵文字、URLは禁止
- 1回答は40〜80文字程度の簡潔な文章
- 質問の意図に答えつつ、必要であれば自然な選択肢を提示して会話を前に進める  
  （例:「保険料の考え方について知りたいか、具体的な見積もりを希望されますか？」）

# 5. 出力形式（重要）
- **回答の冒頭に `【User ID】` や `[user0]` などの宛先タグ・話者名を一切含めないこと。**
- 出力は読み上げ可能な「話し言葉のテキスト」のみとする。
- 無視すべき入力なら **[SILENCE]** のみを出力する。

良い出力例:
user: 「一緒にカフェ行こうよ。今日の天気ってどうかな?」
あなた: 「今日は、午後から雨が降る予報です。傘は持って行ったほうが安心ですよ」
user: 「そっか。ありがとう。」
あなた: 「どういたしまして。どこのカフェに行きますか？」
user: 「駅前のカフェって空いてる？」
あなた: 「はい、駅前のカフェは通常通り営業していますよ」
user: 「じゃあ、そこに行こう。あそこは紅茶が美味しいんだよ」
あなた: 「いいですね。昼になると、混みはじめるかもしれないので、少し早めに行くと落ち着けますよ」
悪い出力例:
「【User 0】 今日は午後から雨が降る予報です。傘は持って行ったほうが安心ですよ」

"""
# モデル名 (確実に動作するもの)
DEFAULT_MODEL = "gemini-2.5-flash-lite"
FALLBACK_MESSAGE = "おでんわ、ありがとうございます。こちらは、ほけんがいしゃです。たんとうに、おつなぎします。"


def _fallback_stream(message: str = FALLBACK_MESSAGE):
    """TTSの自然さを保つため、短い塊で逐次返す。"""
    step = 12
    for i in range(0, len(message), step):
        yield message[i:i + step]

def generate_answer_stream(question: str, model=DEFAULT_MODEL, history: list = None):
    """
    回答をストリーミング(ジェネレータ)として返す
    """
    # 履歴がNoneなら空リストで初期化
    if history is None:
        history = []
        
    print(f"[DEBUG] ストリーミング生成開始... (モデル: {model}, 履歴数: {len(history)})")
    
    if not question:
        yield "質問を聞き取れませんでした。"
        return

    try:
        if model.startswith("gemini-"):
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY が設定されていません")
            
            model_instance = genai.GenerativeModel(
                model_name=model,
                system_instruction=SYSTEM_PROMPT,
                generation_config={"temperature": 0.2}
            )
            
            # 会話履歴を引き継ぐ
            chat_session = model_instance.start_chat(history=history)
            
            # ストリーミングリクエスト
            response = chat_session.send_message(question, stream=True)
            
            # ★★★ エラー対策の修正箇所 ★★★
            yielded_any = False
            for chunk in response:
                try:
                    # chunk.text にアクセスしてみて、中身があれば yield する
                    text_part = chunk.text
                    if text_part:
                        yielded_any = True
                        yield text_part
                except ValueError:
                    # 「終了合図」などの空データが来た場合、ValueErrorが出るので無視して次へ
                    continue

            # クォータ超過等で実質空応答になった場合のフォールバック
            if not yielded_any:
                print("[WARN] Gemini returned empty response. Using fallback message.")
                yield from _fallback_stream()

        else:
            yield f"対応していないモデル名です: {model}"

    except Exception as e:
        print(f"[ERROR] ストリーミング生成エラー: {e}")
        yield from _fallback_stream()


# --- 単体テスト ---
if __name__ == "__main__":
    print("--- テスト実行 ---")
    if GOOGLE_API_KEY:
        # 空の履歴でテスト
        iterator = generate_answer_stream("こんにちは", history=[])
        for text in iterator:
            print(text, end="", flush=True)
        print("\n")
    else:
        print("APIキーがありません")
