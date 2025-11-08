# /workspace/answer_generator.py
import openai
import os
import time

# --- OpenAI APIキー設定 ---
# RunPodの環境変数、またはColabのSecretsからAPIキーを読み込みます。

# 環境変数 OPENAI_API_KEY から取得
openai.api_key = os.getenv("OPENAI_API_KEY")

# 確認用（必要なら）
if openai.api_key is None:
    raise ValueError("環境変数 OPENAI_API_KEY が設定されていません")


def generate_answer(question: str, model="gpt-4o-mini") -> str:
    """
    受け取った質問テキストに対してOpenAI LLMで回答を生成する。
    
    Args:
        question (str): 文字起こしされた質問文
        
    Returns:
        str: 生成された回答文
    """
    print(f"[DEBUG] OpenAI回答生成中... 質問: '{question}'")
    
    if not question:
        return "質問を聞き取れませんでした。"

    # APIキーが設定されているか最終チェック
    if not openai.api_key:
        print("[ERROR] OpenAI APIキーが未設定です。ダミー回答を返します。")
        time.sleep(1) # ダミーの処理時間
        return f"「{question}」ですね。（ダミー回答：APIキー未設定）"

    # --- OpenAI APIで回答生成 ---
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは丁寧に質問に答えるアシスタントです。"},
                {"role": "user", "content": question}
            ],
            temperature=0.7,  # 創造性の調整
            max_tokens=1000   # 出力の最大トークン数
        )

        answer = response.choices[0].message.content.strip()
        print(f"[DEBUG] OpenAI回答生成完了: '{answer}'")
        
    except Exception as e:
        print(f"[ERROR] OpenAI APIでの回答生成に失敗しました: {e}")
        answer = f"申し訳ありません、回答を生成中にエラーが発生しました。({e})"

    return answer

# 実行例 (モジュールの単体テスト)
if __name__ == "__main__":
    # ※ このテストを実行するには、Colab userdataまたは環境変数に
    #    OPENAI_API_KEYが設定されている必要があります。
    print("--- OpenAI回答生成 単体テスト ---")
    test_q = "こんにちは、今日の天気はどうですか？"
    
    if not openai.api_key:
        print("APIキーが設定されていないため、テストをスキップします。")
    else:
        ans = generate_answer(test_q)
        print(f"質問: {test_q}")
        print(f"回答: {ans}")