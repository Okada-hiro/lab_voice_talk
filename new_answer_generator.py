# /workspace/answer_generator.py (マルチAPI対応版)

import google.generativeai as genai  # ★ 追加
import os
import time


# Google (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Google APIキーを設定（キーがある場合のみ）
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[ERROR] Google genai の設定に失敗: {e}")


# --- 2. モデル名に応じて処理を分岐する ---

def generate_answer(question: str, model="gemini-2.5-flash-lite") -> str:
    """
    受け取った質問テキストに対し、指定されたモデルで回答を生成する。
    model名のプレフィックスでAPIを自動的に切り替える。
    
    Args:
        question (str): 文字起こしされた質問文
        model (str): 使用するモデル名 
                      ("gpt-...", "gemini-...", "claude-...")
        
    Returns:
        str: 生成された回答文
    """
    print(f"[DEBUG] 回答生成中... (モデル: {model}) 質問: '{question}'")
    
    if not question:
        return "質問を聞き取れませんでした。"

    answer = ""
    system_prompt = "あなたは丁寧に質問に答えるアシスタントです。"

    try:
        

        # --- B. Google (gemini-...) ---
        if model.startswith("gemini-"):
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY が設定されていません")
            
            # Geminiモデルを初期化
            model_instance = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt
            )
            response = model_instance.generate_content(question)
            answer = response.text.strip()

       

        # --- D. 対応モデルなし ---
        else:
            raise ValueError(f"対応していないモデル名です: {model}")

        print(f"[DEBUG] {model} 回答生成完了: '{answer[:30]}...'")

    except Exception as e:
        print(f"[ERROR] {model} での回答生成に失敗しました: {e}")
        answer = f"申し訳ありません、回答を生成中にエラーが発生しました。({e})"

    return answer

# --- 3. 単体テスト (全モデルをテスト) ---
if __name__ == "__main__":
    print("--- マルチAPI回答生成 単体テスト ---")
    test_q = "こんにちは、今日の天気はどうですか？"


    # --- Google/Gemini テスト ---
    if GOOGLE_API_KEY:
        print("\n[Test] gemini-2.5-flash-lite")
        ans_gemini = generate_answer(test_q, model="gemini-2.5-flash-lite")
        print(f"Gemini 回答: {ans_gemini}")
    else:
        print("\n[Skipped] GOOGLE_API_KEY 未設定のため gemini-2.5-flash-lite をスキップ")
    