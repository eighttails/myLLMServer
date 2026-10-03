# AGENTS.md — myLLMServer

このファイルは、AI エージェント (Copilot 等) がこのプロジェクトを扱う際のルールと背景知識の定義です。
変更を加える前に必ずこのファイルと README.md を通読してください。

## プロジェクト概要

llama.cpp (llama-server) を Docker コンテナで実行するラッパー。
複数の GGUF モデルを router mode で切り替えながら提供し、Ollama API 互換・OpenAI API 互換エンドポイントを公開する。

- 技術構成: ホスト側は Bash、コンテナ内の制御・モデル同期・preset計算は Python3 (stdlib のみ)。Node.js / Go 等の依存パッケージは存在しない
- ベースイメージ: `ghcr.io/ggml-org/llama.cpp:full-cuda`
- 公開ポート: `11434` (Ollama 互換 `/api/*` + OpenAI 互換 `/v1/*`)

## ディレクトリ構造と役割

| パス                   | 役割                                                                                   |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `launch-container.sh`  | ホスト側起動スクリプト (イメージ再ビルド + コンテナ再作成)                             |
| `reload-model.sh`      | `model_list.yml` を再ロード (モデル・補助ファイルの追加 DL / 不要ファイル削除 / リスト反映) |
| `unload-model.sh`      | ホスト側からモデルを VRAM からアンロード                                               |
| `model_list.yml`       | ユーザー編集対象のモデル・MTP/mmproj/imatrix指定ファイル                         |
| `model_list.example.yml` | `model_list.yml` が無い場合の初期値の元                                      |
| `docker/`              | コンテナイメージの中身 (Dockerfile + Python 制御スクリプト + API プロキシ)              |
| `models/`              | モデル GGUF のダウンロード先 (.gitignore 済み。**この中のファイルは絶対に編集しない**) |
| `continue/config.yaml` | Continue (VS Code) 用の設定サンプル                                                    |

## 変更時の原則 (最重要)

1. **重い処理は起動時に行わない** — KV キャッシュ量子化の選択・`tensor-split` 計算・GGUF メタデータ解析は、すべて「モデルが切り替わった時」にのみ実行する (lazy preset)。これを「起動時に全モデル分まとめて計算」する設計へ改変しない。
2. **出力tok/sを優先しつつ、見積もり不能時は安全側へ戻す** — 既定は単一リクエストを優先する。単一GPUで重み・KV・予約VRAMの収まりを見積もれた場合は `fit = off` / `n-gpu-layers = all` を明示し、見積もりが不成立または不確実な場合は llama-server の `--fit` にフォールバックする。複数GPUの `tensor-split` は実測速度とVRAM予算に基づいて設定する。
3. **Ollama / OpenAI 両互換エンドポイントの API 形状は破壊変更しない** — ollama-vscode / Continue / Copilot Chat といったクライアントがそのまま使えることが前提。
4. **環境変数は README.md とスクリプト両方に反映する** — `launch-container.sh` が環境変数を `-e` でコンテナへ渡すループ (`for var in ...`) と、README の環境変数一覧表の両方に追加する必要がある。
5. **stdlib のみでPythonを書く** — `lazy-llama-proxy.py` は `http.server` / `urllib` 等、標準ライブラリのみ使用。pip 依存パッケージを追加しない。
6. **ホストの UID/GID でコンテナを動かす設計を壊さない** — モデルファイルを root 権限なしで削除可能にしている仕組み (PUID/PGID)。
7. **推論配置は単一リクエストの出力速度を優先する** — 自動コンテキストは最大256Kを目標とし、モデル重みのGPU常駐を優先する。既定の並列スロットは1。`MAX_PARALLEL_SLOTS` の増加は合計スループットを選ぶ明示的な調整とする。
8. **投機的デコードは速度実測に基づき選ぶ** — 既定では無効。明示されたドラフトモデルまたは内蔵MTPヘッドがある場合も、`SPECULATIVE_DECODING=on` が指定されたときだけ使用する。ONにする場合のdraft KVはQ4。
9. **速度比較は公開APIの同一経路で行う** — 同一モデル・同一prompt token数・同一生成上限・同一sampling条件で複数回測り、実際のモデル起動引数と`timings.predicted_per_second`を記録する。直接CLIとプロキシ経由、または異なるコンテキスト長の数値を同条件として比較しない。思考型モデルではこの値に非表示のreasoning tokenが含まれ得るため、ユーザーに見える回答速度と区別する。

## 主要な動作の流れ (変更を検討する前に必ず把握すること)

1. `launch-container.sh` → イメージビルド → コンテナ起動 (`start-llama.sh` は `start-llama.py` の互換ランチャー)
2. `start-llama.py` → `sync-model.py` (モデルの DL / 不要削除) → 軽量 preset 生成 → `lazy-llama-proxy.py` と llama-server (router) を起動
3. リクエストが公開ポート `11434` に到着 → プロキシが `model` を確認 → **前回と異なる場合だけ** `configure-model-preset.py` を実行して preset を再計算し llama-server に reload
4. アイドル `MODEL_IDLE_SECONDS` 超過でモデルが VRAM から自動解放 (`--sleep-idle-seconds`)

## 開発・検証の手順

### ビルド / 起動 / 再起動

```bash
./launch-container.sh          # イメージの再ビルド + コンテナの削除・再作成を毎回行う
```

- スクリプト変更後は必ず再実行する (イメージに焼き込まれるため、コンテナ単体の再起動では反映されない)
- 設定変更時は変更したい環境変数を export してから実行:

```bash
export KV_CACHE_TYPE=q4_0
./launch-container.sh
```

### 動作確認 (API smoke test)

```bash
# Ollama 互換: 登録モデル一覧
curl http://localhost:11434/api/tags

# OpenAI 互換: チャット
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<モデル名>", "messages": [{"role": "user", "content": "hello"}]}'

# リロード / アンロード
curl -X POST http://localhost:11434/models/reload
curl -X POST http://localhost:11434/models/unload -H "Content-Type: application/json" -d '{}'
```

- `<モデル名>` は GGUF ファイル名から拡張子を除いたもの
- GPU 空き状況の確認: `nvidia-smi`
- ログの確認: `docker logs -f my-llm-server`

### スクリプトの検証

- ホスト側 Bash スクリプトは `set -euo pipefail` を維持する。コンテナ内の `.sh` は既存パス互換のための薄い Python 起動ラッパーとする
- 可能な範囲では `bash -n <script>` と `python3 -m py_compile <script>` で構文チェックを行う
- 検証には本物の GPU・モデル DL が要るため、実行結果の確認はユーザーへの依頼を前提とする

## 文書メンテナンスルール

- 機能・環境変数・エンドポイントを変更したら **README.md を必ず同期して更新する**
- 日本語コメント・日本語 README がこのプロジェクトの規約 (house style)。新規コードも日本語コメントを推奨する
- 英語と日本語が混在する場合は、既存の記述 (主に README) に合わせる

## やってはいけないこと

- `models/` 内の GGUF ファイルや `model_list.txt` を勝手に書き換える (ユーザーの作業領域)
- `docker/` 内のスクリプト名を変更する (Dockerfile の `COPY` と相互参照に依存している)
- 公開ポートのデフォルト (`11434`) を変更する (クライアント設定との整合性)
- 起動時の起動時間や VRAM 使用量を悪化させる設計変更 (lazy preset / `--fit` 委譲を壊すような変更)
- 既存の API エンドポイントのパスを変更する

## トラブルシューティングの目安 (ユーザーに回答する際)

| 症状                                | 確認事項                                                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `500 model name=... failed to load` | `docker logs` で `cudaMalloc failed` を確認。他プロセスの GPU 使用中なら `KV_CACHE_TYPE=q4_0` や `MAX_CONTEXT_SIZE` 制限を提案 |
| context size exceeded               | リクエストが `ctx-size` を超えていないか。`CONTEXT_SIZE` / `MAX_CONTEXT_SIZE` で調整                                           |
| モデルがロードされない              | `model_list.yml` の `models[].url` と `models/` 中のファイル名を照合                                     |
