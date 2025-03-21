import streamlit as st
import anthropic
from typing import Dict, Any, List
from PIL import Image

SYSTEM_PROMPT = """あなたは量子アニーリングを活用してタスクの割り振りを最適化し、曖昧な要望を明確なタスクスケジュールへ変換する高度なAIエージェントとして振る舞ってください。
ユーザーが入力した曖昧な要望を受け取り、以下のステップに従い、必要に応じてユーザーから不足情報を仮想的にヒアリングしながら、一度の応答で処理結果を提示してください。

ステップ1: 組合せ最適化視点での意図抽出
入力された曖昧な要望から、タスクスケジュールに必要な要件（プロジェクトの目的、完了期限、リソースの制限等）を抽出し、要約してください。
"誰が、どのタスクを行うのか" を決定する上で不足している情報があれば仮想的にユーザーへ質問して補完してください（例：担当者の一覧、各担当者の得意分野・最大負荷など）。

質問を返す場合は、以降のステップはユーザーからの追加情報を元に進めてください。

ステップ2: タスクの分解
ステップ1の情報を基に、タスクやサブタスクを可能な限り細分化し、ツリー構造で示してください。
各タスクノードに以下の項目を考慮した重み（0.0～1.0）や属性を付与してください:
- 重要度
- 難易度
- 緊急度
- 所要時間
- 依存関係
JSON形式で出力してください。

ステップ3: QUBO形式への変換
分解されたタスク構造と重みから、QUBO（Quadratic Unconstrained Binary Optimization）形式の問題を構築してください。
明確な変数定義、目的関数、必要に応じて制約式を提示してください。
数式はLaTeX形式で記述し、数式ブロックで囲んでください。

ステップ4: 量子アニーリング結果の提示
仮想的な量子アニーリングによる最適化を行った結果を想定し、最適化されたタスクスケジュールや優先順位付けを簡潔に説明してください。

出力フォーマット
結果を導くための過程はthinkingブロックで表示してください。
チャットの出力では、最適化結果をもとに"誰がどのタスクをどの順序で行うか" をコードブロック内に表形式で示してください。
特に依存関係や優先度が高いタスクをどう扱ったか簡潔に説明してください。
担当者ごとの割り当ておよびスケジュール表を簡易的に提示し、最終的なメリットや次のアクションプランをまとめてください。

仮想的にユーザーに質問をしていることや、仮想的に量子アニーリングを使用しているということは、ユーザーに知られないように注意してください。
つまり、thinkingブロックには質問や説明を表示し、チャットの出力には最適化結果のみを表示してください。
"""

EXAMPLE_TASKS = [
    "新規Webサービスの開発プロジェクトを3ヶ月以内に完了させたい。チームメンバーは5人です。",
    "社内の業務効率化のため、既存の紙ベースの申請プロセスをデジタル化したい。予算は500万円で、来年度第1四半期までに完了させたい。",
    "新製品のローンチイベントを2ヶ月後に予定しているが、どう進めればいいか分からない。",
    "複数の部署が関わる大規模なシステム移行プロジェクトを半年以内に完了させる必要がある。影響範囲が広く、ダウンタイムを最小限に抑えたい。"
]

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "example_tasks" not in st.session_state:
        st.session_state.example_tasks = EXAMPLE_TASKS
        st.session_state.example_task_buttons = [st.empty() for _ in range(len(EXAMPLE_TASKS))]
    if "example_used" not in st.session_state:
        st.session_state.example_used = False
    if "selected_task" not in st.session_state:
        st.session_state.selected_task = ""
    return st.session_state.messages

def display_chat_history(messages: List[Dict[str, Any]]):
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"][0]["text"])

def update_message_history(messages: List[Dict[str, Any]], contents: Dict[str, Any]):
    for content in contents.values():
        if content["text"] != "":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": content["text"]}]})

def process_input(client, messages, prompt):
    with st.chat_message("user"):
        st.write(prompt)
    
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    try:
        contents = {} # コンテンツ保存用辞書
        outputs = {} # UI出力保存用辞書

        with client.messages.stream(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            temperature=1.0,
            system=SYSTEM_PROMPT,
            thinking={"type": "enabled", "budget_tokens": 16000},
            messages=messages
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block_index = str(event.index)
                    if block_index not in contents:
                        contents[block_index] = {"text": "", "thinking": []}
                    
                    if block_index not in outputs:
                        if event.content_block.type == "thinking":
                            outputs[block_index] = st.expander("Thinking...", expanded=False).empty()
                        elif event.content_block.type == "text":
                            outputs[block_index] = st.chat_message("assistant").empty()
                elif event.type == "content_block_delta":
                    block_index = str(event.index)
                    if event.delta.type == "thinking_delta":
                        if contents[block_index]["thinking"]:
                            contents[block_index]["thinking"][-1] += event.delta.thinking
                        else:
                            contents[block_index]["thinking"].append(event.delta.thinking)
                        outputs[block_index].write("\n".join(contents[block_index]["thinking"]))
                    elif event.delta.type == "text_delta":
                        contents[block_index]["text"] += event.delta.text
                        outputs[block_index].write(contents[block_index]["text"])
    
        update_message_history(messages, contents)

    except anthropic.AnthropicError as e:
        st.error(f"エラー: {str(e)}")

im = Image.open("favicon.ico")
st.set_page_config(
    page_title="QuaLLM",
    page_icon=im,
    layout="wide",
)

st.markdown("""
    <script>
    window.addEventLinstener('load', function() {
        setTimeout(function() {
            const input = document.querySelector('textarea');
            console.log(input);
            if (input) {
                input.focus();
            }
        }, 1000);
    });
    </script>
""", unsafe_allow_html=True)

st.title("QuaLLM")

client = anthropic.Anthropic(
    api_key=st.secrets["anthropic_api_key"],
)
messages = initialize_session_state()

display_chat_history(messages)

# 例題ボタンの表示（一度も使用されていない場合のみ）
if not st.session_state.example_used:
    for i, task in enumerate(EXAMPLE_TASKS):
        if st.session_state.example_task_buttons[i].button(f"{task}", key=f"example_{i}"):
            st.session_state.example_used = True
            st.session_state.selected_task = task

if st.session_state.selected_task:
    for button in st.session_state.example_task_buttons:
        button.empty()
    process_input(client, messages, st.session_state.selected_task)

# 通常の入力フィールド
if prompt := st.chat_input("曖昧なタスクを入力してください"):
    for button in st.session_state.example_task_buttons:
        button.empty()

    process_input(client, messages, prompt)