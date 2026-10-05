# myLLMServer

llama.cpp (llama-server) を Ollama API 互換および OpenAI API 互換のエンドポイントとして
Docker コンテナ上で動かすためのラッパーです。
複数の GGUF モデルをルーターモード (router mode) で切り替えながら提供し、一定時間使われていないモデルは
自動的に VRAM から解放されます。

## 特徴

- ホスト環境には Docker 以外の追加インストールが不要(モデルのダウンロード・配置もコンテナ内で完結)
- `model_list.yml` で指定した Hugging Face 上のモデルと補助ファイルを自動ダウンロード・配置
- 投機的デコード(MTP)は既定で無効。出力速度を実測して有効にする場合は、内蔵MTPヘッドまたは YAML の `mtp` (ドラフトモデル) を使用。`mmproj` はモデルごとに指定してllama-serverへ自動適用し、`imatrix` は自動ダウンロードのみ行う(推論時には使用されない)
- リストにないモデル(キャッシュ済みファイル)は起動時に自動削除
- `CUDA_VISIBLE_DEVICES` を参照し、未設定なら全 GPU を使用
- 一定時間(デフォルト30分、環境変数で変更可)アイドルなモデルは VRAM を自動解放
- ホスト側の UID/GID でコンテナを実行するため、ダウンロードしたモデルファイルをホスト側から root 権限なしに削除可能
- モデルの GGUF メタデータから推奨コンテキスト長を自動検出し、GPU の空き VRAM に応じて自動フィット(OOM 回避)
- Ollama API 互換エンドポイントを提供し、ollama-vscode拡張機能から VS Code 上で利用可能
- OpenAI API 互換エンドポイントを提供し、Continue拡張機能から VS Code 上で利用可能

## 構成

```
.
├── docker/
│   ├── Dockerfile        # llama.cpp:full-cuda ベースイメージ + ラッパースクリプト
│   ├── start-llama.py    # コンテナ内のモデル同期・llama-server/proxy起動を制御
│   ├── sync-model.py     # model_list.yml の読み込み、モデル自動ダウンロード・不要モデル削除を行う
│   ├── configure-model-preset.py # モデル切替時に重い preset 計算を行う
│   ├── lazy-llama-proxy.py       # 公開ポートで受け、モデル切替時だけ preset を更新する
│   ├── unload-model.py           # コンテナ内からモデルをアンロードする
│   └── *.sh                     # 既存パス互換の Python 起動ラッパー
├── launch-container.sh    # ホスト側から使う起動スクリプト(ビルド + コンテナ再作成)
├── unload-model.sh        # 外部(ホスト側)からモデルをアンロードするスクリプト
├── reload-model.sh        # model_list.yml を再ロードし、モデル・補助ファイルの追加ダウンロード＆不要ファイル削除を行うスクリプト
├── model_list.yml         # 使用するモデル・補助ファイルの指定ファイル
├── model_list.example.yml # モデル指定ファイルのサンプル
├── continue/
│   └── config.yaml        # Continue (VS Code拡張) 用のモデル設定サンプル
└── models/                # モデルダウンロード先 (.gitignore 済み、初回は空でOK)
```

## 前提条件

- Docker (NVIDIA Container Toolkit導入済み。`docker run --gpus all` が使えること)
- NVIDIA GPU (複数GPU可)

## 使い方

### 1. 起動

初回起動時、`model_list.yml` が存在しない場合は `model_list.example.yml` から自動作成され、そこに記述されたモデルが使用されます。
**自分で使いたいモデルを指定する場合は、`model_list.yml` を編集してください。**

```bash
./launch-container.sh
```

初回実行時、指定モデルが `models/` 配下になければ Hugging Face から自動ダウンロードされます。
KV キャッシュ量子化や `tensor-split` などの重い計算は起動時には全モデル分まとめて実行せず、
リクエストされたモデルが切り替わるタイミングで対象モデルだけ再計算します。

起動後は `http://localhost:11434/v1` が OpenAI 互換の API エンドポイントになります。
また、VS Code/Copilot Chat のローカルモデル検出で使われる Ollama 互換の
`http://localhost:11434/api/tags` でも、登録済みモデル名だけを返します。
チャット送信用に Ollama 互換の `http://localhost:11434/api/chat` (ストリーミング/非ストリーミング両対応)
も実装しており、tool calling の `tools` / `tool_calls` を含めて内部で OpenAI 互換 API に変換し、
`llama-server` へ転送します。`tools` を指定したリクエストでも常にバックエンドとはストリーミングで
通信し、生成中も応答チャンクを継続して送ることでクライアント側の無通信タイムアウトを防ぎます。
ストリーミング応答の最終チャンクには、バックエンドが返す`prompt_eval_count`と`eval_count`を含めます。
`GET /api/ps`は稼働中モデルのスロットあたりの`context_length`も返します。
`tool_calls` はストリーミング中に断片(index単位で分割された name/arguments)を組み立て、
生成完了時にまとめてクライアントへ返します。推論内容や未完成の `tool_calls` 断片は公開せず、
空のcontentチャンクを継続して送ります。バックエンドからSSEが届かない推論区間も15秒間隔で
接続維持チャンクを送るため、長い推論中も接続を維持します。
Ollamaストリーミング応答では、一定長以上の同一ブロックや同一行が連続する明白な生成ループを
検出すると、反復部分の転送を止めてllama-server側の生成もキャンセルします。出力トークン数には
既定の上限を設けないため、反復していない正常な長文生成は継続できます。また、同じtool callと
同じtool結果を含む1～4ステップの周期が3回続いた場合は、次のtool call生成を始める前にAgentループ
として停止します。ポーリングなどで結果が変化している場合は同一ループとは判定しません。
OpenAI互換APIのストリーミング応答も、バックエンドから到着したチャンクを大きな読取バッファが埋まるまで待たずに転送します。

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<モデル名>",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

`model` にはダウンロードした GGUF ファイル名から拡張子を除いたものを指定します。

### ollama-vscode拡張機能から使う場合

本サーバーは、モデル一覧取得の `GET /api/tags` とチャット送信の `POST /api/chat` を実装しています。
ollama-vscode拡張機能の Ollama 接続先を `http://localhost:11434` に設定すると、登録済みモデルを選択して
VS Code 上から利用できます。チャットはストリーミング/非ストリーミングと tool calling に対応しています。

### 2. モデルリストを変更する

自分が使いたいモデルを指定・変更する場合は、`model_list.yml` を編集します。
`models` 配列の各要素に `url` を指定し、必要なら `shards` (分割GGUFの2個目以降)、`mtp` (投機的デコーディング用ドラフトモデル、llama-server の `model-draft` として自動適用)、`mmproj` (マルチモーダル投影、`mmproj` として自動適用) のURLを追加します。投機的デコードは既定で無効です。`SPECULATIVE_DECODING=on` を設定すると、`mtp` 指定時は `draft-simple`、GGUF内に対応する内蔵MTPヘッドがある場合は `draft-mtp` を使用します。有効化時のドラフトKVキャッシュはQ4です。`imatrix` (量子化用データ、再量子化などに使う場合のみ) はダウンロードだけ行い、llama-server の推論設定には反映されません(llama-serverに imatrix を読み込む実行時オプションが無いため)。

```text
# model_list.yml の例
models:
  - url: https://huggingface.co/<owner>/<repo>/resolve/main/<main>.gguf
    mtp: https://huggingface.co/<owner>/<repo>/resolve/main/<draft>.gguf
    mmproj: https://huggingface.co/<owner>/<repo>/resolve/main/mmproj-model-f16.gguf
    imatrix: https://huggingface.co/<owner>/<repo>/resolve/main/imatrix.dat

  # 分割GGUF: url は1個目、shards は2個目以降を順番にすべて指定
  - url: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
    shards:
      - https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
```

`url`のファイル名は`<名前>-00001-of-<合計数>.gguf`形式で、必ず1個目を指定します。
`shards`には`00002`から合計数までを順番に指定してください。全シャードは同一リポジトリに置きます。
同期時に連番・総数・配置を検証して全シャードをダウンロードし、モデルの一部として保持します。分割でないGGUFへの`shards`指定、
シャードの不足・順番違い・別リポジトリ指定は、そのモデル設定のエラーになります。従来の単体GGUF設定はそのまま使えます。

編集後、コンテナを再起動せずに `model_list.yml` を再ロードして変更を即座に反映したい場合は、`./reload-model.sh` を実行します。
このコマンドは起動中のコンテナを前提とし、モデルリストを共有ディレクトリへ反映した後、コンテナ内のPython同期処理を実行します。

```bash
./reload-model.sh
```

このコマンド（または `./launch-container.sh`）を実行すると、以下の処理が自動で行われます:
- `model_list.yml` に新たに追加されたモデル・全シャード・補助ファイルを Hugging Face から自動ダウンロード
- インストール済みだが `model_list.yml` に記載のない（今後使わない）モデル・補助ファイルをディスク（`models/`）から自動削除
- サーバー（llama-server および プロキシ）のモデルリストを即座に更新

また、HTTP API 経由で再ロードをトリガーすることも可能です。

```bash
curl -X POST http://localhost:11434/models/reload
```

一時的に環境変数でモデルを指定したい場合は、`MODEL_NAMES_CSV` を直接指定して起動することも可能です。

```bash
MODEL_NAMES_CSV="<リポジトリ名>/<ファイル名>.gguf" ./launch-container.sh
```

### 3. モデル保存先を変更する

```bash
MODEL_DIR=/path/to/your/models ./launch-container.sh
```

未指定の場合は `./models` が使われます。

### 4. 外部からモデルをアンロードする

アクティブなモデルを VRAM から手動でアンロードしたい場合は、`./unload-model.sh` スクリプトを実行します。

```bash
# 現在ロードされているアクティブモデルをアンロード
./unload-model.sh

# 指定したモデルをアンロード
./unload-model.sh <モデル名>
```

また、HTTP API から直接アンロードエンドポイントを呼び出すことも可能です。

```bash
# 現在アクティブなモデルをアンロード
curl -X POST http://localhost:11434/models/unload -H "Content-Type: application/json" -d '{}'

# 特定のモデルをアンロード
curl -X POST http://localhost:11434/models/unload -H "Content-Type: application/json" -d '{"model": "<モデル名>"}'
```

## 環境変数一覧

`launch-container.sh` 実行前に環境変数を export しておくと、コンテナに引き継がれます。

| 変数名                                                                  | デフォルト               | 説明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `IMAGE_NAME`                                                            | `my-llm-server:latest`   | ビルドする Docker イメージ名                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `CONTAINER_NAME`                                                        | `my-llm-server`          | 作成するコンテナ名                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `MODEL_DIR`                                                             | `./models`               | モデルダウンロード先(ホスト側パス)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `PORT`                                                                  | `11434`                  | 公開ポート                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `LLAMA_ROUTER_PORT`                                                     | `PORT + 1`               | コンテナ内部の llama-server router 用ポート。通常は変更不要                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `PUID` / `PGID`                                                         | 実行ユーザーの uid/gid   | コンテナ内プロセスの実行ユーザー(ダウンロードファイルの権限をホストと一致させる)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `CUDA_VISIBLE_DEVICES`                                                  | (未設定=全GPU)           | 使用する GPU を限定したい場合に指定                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MODEL_LIST_FILE`                                                       | `./model_list.yml`       | YAMLモデルリストのパス                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `HF_ENDPOINT`                                                           | `https://huggingface.co` | モデルダウンロード元エンドポイント                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `http_proxy` / `https_proxy` / `no_proxy` (および大文字版、`all_proxy`) | (未設定=不使用)          | ホスト側のプロキシ設定をコンテナに引き継ぐ。モデルダウンロード(`sync-model.sh` の curl)に使用される。コンテナ内部の通信は常に `127.0.0.1` / `localhost` が `no_proxy` に追加されるためプロキシを回避しない                                                                                                                                                                                                                                                                                                                                                                                  |
| `MODEL_IDLE_SECONDS`                                                    | `1800`                   | この秒数(デフォルト30分)アイドルが続いたモデルは VRAM から解放される                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `CONTEXT_SIZE`                                                          | (未設定=自動検出)        | 全モデル共通のコンテキスト長を固定したい場合に指定。未指定時はモデルの GGUF メタデータ(`<arch>.context_length`)から推奨値を自動検出                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MAX_CONTEXT_SIZE`                                                      | (未設定=自動上限256K)    | コンテキスト長の上限。自動検出時はモデルの対応長と256Kの小さい方を使用。明示すれば256Kを超える設定も可能                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MIN_CONTEXT_SIZE`                                                      | `2048`                   | KV キャッシュが VRAM に収まらず自動でコンテキスト長を切り詰める際の下限。これを下回る場合のみ `--fit` にフォールバックする                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `CONTEXT_SIZE_STEP`                                                     | `1024`                   | コンテキスト長を自動で切り詰める際の丸め単位                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `N_GPU_LAYERS`                                                          | `auto`                   | GPU に載せるレイヤー数。`auto`/`all`/数値を指定可能。`auto` の場合は後述の `--fit` に判断を委ねる                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MODELS_MAX`                                                            | `1`                      | 同時にロードしておくモデル数の上限(router mode)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `MAX_PARALLEL_SLOTS`                                                    | `1`                      | 同時実行スロット数の上限。既定は単一リクエストの出力速度を優先して1。複数同時リクエストの合計tok/sを優先する場合に増やす(同時実行中は1リクエストあたりの速度が下がる場合がある)                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `FLASH_ATTN`                                                            | `on`                     | Flash Attentionの使用設定。`on`/`off`/`auto`を指定可能                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `BATCH_SIZE`                                                            | `1024`                   | prompt処理の論理バッチサイズ。llama.cpp既定値の2048より小さくして一時的なVRAM使用量を抑制                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `UBATCH_SIZE`                                                           | `256`                    | prompt処理の物理バッチサイズ。llama.cpp既定値の512より小さくして計算バッファのVRAM使用量を抑制。`BATCH_SIZE`以下で指定                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `VRAM_RESERVE_MIB`                                                      | `4096`                   | `--fit`と手動`tensor-split`計算でGPUごとに確保する、compute bufferとCUDAワークスペースを含むランタイム用のVRAM余白(MiB)                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MOE_CPU_OFFLOAD`                                                       | `auto`                   | MoE expert重みのCPU配置。GPUに全重みを載せられない場合に限り、`auto`はアクティブexpert比率が閾値以下のモデルで必要最小限の先頭層をCPUへ配置。`all`は比率によらず検討、`off`は無効化                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MOE_ACTIVE_RATIO_THRESHOLD`                                            | `0.125`                  | `MOE_CPU_OFFLOAD=auto`でCPU配置を有効にする`expert_used_count / expert_count`の上限                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MOE_RAM_RESERVE_MIB`                                                   | `8192`                   | MoE expert重み / Dense FFN重みをCPUへ配置した後も残すホストRAMの余白(MiB)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `KV_CACHE_TYPE`                                                         | (未設定=`q4_0`)          | KVキャッシュの型を指定。未指定時に高精度型へ自動引き上げず`q4_0`を維持し、VRAMをモデル重みへ優先する。KとVは同じ型。`f16` / `q8_0`等は明示指定可能。型ごとの出力速度はGPU・モデル・コンテキストに依存するため実測を推奨                                                                                                                                                                                                                                                                                                                                                                      |
| `TENSOR_SPLIT_MODE`                                                     | `auto`                   | 複数 GPU 構成での層分割方法。`auto` の場合、モデル切替時に GPU 毎の生成速度と空き VRAM を実測し、対象モデルの性能比に応じた `tensor-split` を計算して高速化を図る(収まらない場合は自動的に `--fit` 任せへフォールバック)。`off` にすると常に `--fit` 任せの従来動作になる。`SPLIT_MODE` が `layer` 以外の場合はこの計算自体を行わない                                                                                                                                                                                                                                                       |
| `SPLIT_MODE`                                                            | `layer`                  | llama-server の `--split-mode`。複数 GPU 間でのモデル分割方式。`layer`(既定): レイヤー単位でGPUに分割するパイプライン並列。1トークン生成中は常にどれか1枚のGPUのみが計算するため、GPU使用率は50%前後(2GPU時)に留まりやすい仕様。`row`: 各レイヤーの重みを行単位でGPU間に分割するテンソル並列で、全GPUが同時に計算に参加できる。`tensor`: 重みとKVキャッシュの両方を分割(実験的)。`none`: 単一GPUのみ使用。**注意**: `row`/`tensor` は毎レイヤーGPU間の同期が発生するため、NVLink等の高速な相互接続が無い環境(PCIe経由のみ)や性能の異なるGPUの組み合わせでは、`layer` より遅くなることがある |
| `THINKING_MODE`                                                         | `auto`                   | Qwen3.6-35B-A3B-UD 等の思考型モデルにおける推論プロセスの出力制御。`on`: thinking モードを有効化 (推論プロセスの出力を許可)。`off`: thinking モードを無効化 (推論プロセスの出力を抑制)。`auto` (デフォルト): リクエストに thinking パラメータがあればそれを使用、なければ有効。Ollama 互換・OpenAI 互換の両 API で動作し、リクエスト/レスポンス両方から `reasoning_content`, `thinking_content`, `reasoning`, `thoughts` 等のフィールドを自動で除去します。 |
| `GENERATION_LOOP_DETECTION`                                             | `on`                     | Ollamaストリーミング応答で、完全一致する本文の反復ループを検出してバックエンド生成を停止する。`off`で無効化 |
| `GENERATION_LOOP_WINDOW_CHARS`                                          | `16384`                  | 生成ループ検出で保持する末尾文字数。`GENERATION_LOOP_MAX_PATTERN_CHARS * GENERATION_LOOP_REPEAT_COUNT`以上が必要 |
| `GENERATION_LOOP_MIN_PATTERN_CHARS`                                     | `64`                     | 反復ブロックとして判定する最小文字数。短い単語や定型句の通常の再利用を誤検出しないための下限 |
| `GENERATION_LOOP_MAX_PATTERN_CHARS`                                     | `2048`                   | 検出対象とする反復ブロック1周期の最大文字数 |
| `GENERATION_LOOP_REPEAT_COUNT`                                          | `3`                      | 同一ブロックを生成ループと判定する最小連続回数 |
| `GENERATION_LOOP_MIN_REPEATED_CHARS`                                    | `256`                    | 生成ループと判定する反復区間全体の最小文字数 |
| `GENERATION_LOOP_LINE_REPEAT_COUNT`                                     | `6`                      | 16文字以上の同一行を生成ループと判定する最小連続行数。反復区間全体は`GENERATION_LOOP_MIN_REPEATED_CHARS`以上必要 |
| `TOOL_LOOP_DETECTION`                                                   | `on`                     | 同じtool call引数と同じtool結果を含む短周期の反復を受信履歴から検出し、次のバックエンド実行前に停止する。`off`で無効化 |
| `TOOL_LOOP_REPEAT_COUNT`                                                | `3`                      | 同一tool実行周期をAgentループと判定する連続回数 |
| `TOOL_LOOP_MAX_CYCLE_LENGTH`                                            | `4`                      | Agentループとして検査するtool実行周期の最大ステップ数 |
| `SPECULATIVE_DECODING`                                                  | `off`                    | `on`にすると、対応する内蔵MTPヘッドまたは`model_list.yml`で指定したドラフトモデルを使った投機的デコードを有効化。draft KVはQ4 |

## VRAM 管理の仕組み

- **アイドル時の自動解放**: 各モデルプロセスに `--sleep-idle-seconds` を設定しており、`MODEL_IDLE_SECONDS`
  で指定した時間アクセスが無いと自動的に VRAM を解放します。
- **計算バッファと共有KVの省メモリ化**: Flash Attentionを有効にし、`BATCH_SIZE` / `UBATCH_SIZE`を
  llama.cppの既定値より小さくしています。また、`--kv-unified`により並列スロット間で単一のKVバッファを共有します。
  バッチサイズをさらに下げるとVRAMを節約できますが、長いpromptの処理速度は低下します。
- **単一リクエストの出力速度優先**: 既定では`MAX_PARALLEL_SLOTS=1`とし、同時リクエストによるGPU演算・メモリ帯域の
  競合を避けます。複数リクエストの合計tok/sを優先する場合は上限を増やしてください。llama-serverの`--kv-unified`
  ではKVバッファを共有するため、実際のスロット数はVRAM上限と`MAX_PARALLEL_SLOTS`の小さい方になります。
- **低アクティブ率MoEの自動CPU配置**: GGUFの`expert_count`と`expert_used_count`を調べ、既定では
  1トークンあたりのアクティブexpert比率が12.5%以下ならCPU配置を検討します。単一GPUかつ`SPLIT_MODE=layer`では
  GGUFの層ごとのexpert重みを使い、KVキャッシュとVRAM余白を維持できる最小数の先頭層だけを`n-cpu-moe`でCPUへ配置します。
  退避する層の重みと`MOE_RAM_RESERVE_MIB`を空きホストRAM内に確保できない場合は、CPU配置を行わず`--fit`に委ねます。
  複数GPUの自動tensor-splitでは、低アクティブ率MoEのexpert重みをRAMに置ける場合だけCPU配置を候補にします。
  Dense FFN重みはホストRAMに余裕があってもCPUへ退避せず、decode時はGPUに配置します。
  `n-cpu-moe`は層単位の静的配置であり、Strataのような実行時のexpert利用頻度に応じた動的キャッシュではありません。
  CPU配置したexpertの計算・転送により生成速度が低下する可能性があります。
- **Qwen 27Bの出力速度実測**: RTX 5070 Ti + RTX 4060 Ti、同一の88-token prompt、temperature 0、
  192-token生成、256K context、Q4 KVで各3回測定したdecode速度の平均です。

  | 構成 | decode速度 |
  | --- | ---: |
  | 既存構成 (split 26/39、Dense FFNの先頭16層をCPU配置、MTPなし) | 9.01 tok/s |
  | Dense FFNをGPU配置 (split 26/39、MTPなし) | 16.40 tok/s |
  | 5070 Ti側へ寄せたsplit 34/31 (MTPなし) | 17.33 tok/s |
  | split 34/31 + 内蔵MTP、draft KVをQ4化 | 23.95 tok/s |

  この測定ではDense FFNのCPU退避をなくすと約82%、さらにsplitを調整すると約6%、MTPを有効にすると
  MTPなしのsplit 34/31比で約38%向上しました。MTP構成はGPUの空きが最小約548 MiBまで減り、
  VRAM余裕が小さくなります。OOMが発生する場合は`SPECULATIVE_DECODING=off`、`MAX_CONTEXT_SIZE`の縮小、
  または`VRAM_RESERVE_MIB`の増加を検討してください。実際の速度・VRAM消費はモデル、GPU負荷、prompt、
  コンテキスト長により変わります。
- **Qwen GSQでのMTP比較 (過去測定)**: 同一の88-token prompt、temperature 0、192-token生成、256K context、Q4 KVで各3回測定し、
  MTP ONは平均12.63 tok/s (draft受理率52.5%)、OFFは23.15 tok/sでした。ただしONは公開API、OFFは一時起動した
  llama-serverへの直接CLIで測定したため、測定経路が一致していません。厳密なA/B比較ではなく参考値として扱い、
  MTPは既定OFFのままモデルごとに公開API上で再測定してください。
- **Qwen GSQの公開API測定 (2026-10-03)**: RTX 5070 Ti + RTX 4060 Ti、MTP OFF、temperature 0、
  192-token出力 (`finish_reason=length`)、`/v1/chat/completions` の非ストリーミング応答を各3回測定したdecode速度の平均です。
  コンテキスト長は262144、KVはQ4、Flash Attentionは有効、batch/ubatchは1024/256、並列スロットは1。
  現行モデル引数は`--fit on --n-gpu-layers auto`で、手動tensor-splitは成立せず`--fit`へフォールバックしていました。

  | 入力prompt tokens | 平均decode速度 |
  | ---: | ---: |
  | 99 | 22.83 tok/s |
  | 4,372 | 22.25 tok/s |
  | 16,252 | 20.49 tok/s |
  | 64,852 | 15.26 tok/s |

  各promptは同一条件の3連続測定です。ただし`timings.predicted_per_second`はモデルが生成した全completion tokenを数え、
  非表示のreasoning tokenを含む場合があります。実際に99-token promptで192-token上限の応答を調べると、
  OpenAI応答の`message.content`は空で`reasoning_content`が999文字でした。Ollama APIでもthinking既定時に1024 tokenを生成した応答は
  可視contentが空のまま上限に達しました。このため上表はGPU decode能力の測定であり、Ollamaクライアントに表示される回答の
  tok/sを再現する測定ではありません。
- **Qwen GSQでのTHINKING_MODE A/B (2026-10-03)**: 同一のユーザーpromptを`/api/chat`、temperature 0、
  `num_predict=512`で3回ずつ測定しました。

  | 設定 | completion tokens | 可視content | 平均decode速度 |
  | --- | ---: | ---: | ---: |
  | `THINKING_MODE=auto` (既定) | 512 | 0文字 | 23.02 tok/s |
  | `THINKING_MODE=off` | 471 | 2,234文字 | 22.96 tok/s |

  raw decode速度はほぼ同じですが、`auto`では512 tokenを内部thinkingに使い切り、ユーザーに見える回答がありませんでした。
  `off`では推論内容より回答本文を生成し、同じtoken上限で回答が得られました。`off`は回答の仕方を変える設定なので、
  速度だけでなく回答品質を確認してから選択してください。計測後のコンテナは`auto` (環境変数未指定)に戻しています。
- **Qwen GSQでthinkingを抑えた可視回答速度**: `/api/chat`、temperature 0、同一の91-token prompt、`num_predict=1024`で3回測定しました。
  Qwenが対応する`/no_think`指示をprompt先頭に付けると、各回789 completion tokens・1726文字の回答が生成され、
  `eval_duration`から求めた平均decode速度は22.80 tok/s (23.05 / 22.46 / 22.89)でした。これはthinking既定を変更せずに行った診断測定で、
  `/no_think`は回答の仕方を変えるため、品質とのトレードオフを確認せず全リクエストへ適用しないでください。
  報告された約10 tok/sはGPUのraw decodeでは再現できず、Qwenの非表示thinkingが回答開始を遅らせている可能性があります。
  次の最適化ではGPU配置やKV型を先に変えず、実際の会話でthinkingを維持する場合と抑える場合を同一prompt・同一APIで比較してください。
- **投機的デコード(MTP)**: `SPECULATIVE_DECODING=off`が既定です。`on`を指定すると、モデル切替時にGGUFテンソルを確認し、
  対応する内蔵MTPヘッドには`draft-mtp`、`mtp`指定の外部ドラフトモデルには`draft-simple`を設定します。
  draft KVはQ4にして追加VRAMを抑えます。MTPに対応しない通常モデルではオプションを追加しません。
  MTPは追加のVRAMを使い、モデルによっては出力速度が低下するため、モデル別の比較で効果を確認してください。
- **推論時のGPU常駐優先**: 単一GPUではVRAM見積もりが成立すると`fit = off` / `n-gpu-layers = all`を明示し、
  全重みをGPUに配置します。低アクティブ率MoEで全重みが収まらない場合だけ、RAM余裕の範囲内で必要最小限の
  先頭expert層を`n-cpu-moe`へ配置します。容量を見積もれない場合は安全側としてllama-serverの`--fit`へ戻します。
- **モデル切替時の遅延 preset 計算**: 公開ポートでは軽量プロキシがリクエストを受け、`model` が直前のモデルから
  変わった場合だけ対象モデルの preset を再計算して llama-server router に reload します。起動時は全モデルに対して
  KV キャッシュ量子化・GGUF テンソル解析・`tensor-split` 計算を行わないため、モデル数が増えても起動時間が伸びにくくなります。
- **GPU 性能比に基づく tensor-split 自動計算(`TENSOR_SPLIT_MODE=auto`、複数 GPU 時のみ)**: モデル切替時に
  GPU 毎の生成速度(tokens/sec)を `llama-bench` で実測し(結果はキャッシュされ、以後の切替では
  再利用される)その比率に応じてレイヤーを性能の高い GPU に多く割り当てる `tensor-split` を計算します。
  ベンチマークには単一 GPU に確実に収まる小サイズのモデルを使用します(各 GPU の空き VRAM が少なく、対象モデル
  単体ではロードできない場合があるため)。計算時には各 GPU の空き VRAM・GGUF のテンソル情報から求めた
  レヤー毎の重みサイズ・KV キャッシュの必要量を考慮し、OOM しない範囲に収まるように按分します。
  予算内に収まらない場合はログに警告を出したうえで手動指定を諦め、`--fit`へフォールバックします。手動splitが
  成立する場合は`fit = off`として計算した配置を使います。単一GPUの配置は上記のGPU常駐見積もりで処理し、
  `TENSOR_SPLIT_MODE=off`では複数GPUの手動splitを行いません。
- **コンテキスト長の自動検出**: `CONTEXT_SIZE`を指定しない場合はGGUFの最大対応長と256Kの小さい方を上限とし、
  VRAM見積もりが不足する場合のみさらに縮めます。256Kを超える長さが必要な場合は`CONTEXT_SIZE`を明示してください。
- **出力速度を優先したVRAM配分**: モデル切替時に、GGUFメタデータ(`block_count` /
  `attention.head_count_kv` / `attention.key_length` / `attention.value_length`)から必要 KV キャッシュ量を求め、
  GPU ごとの空きVRAMから`VRAM_RESERVE_MIB`を差し引いた量を予算として、次の優先順位で割り当てます。
  `head_count_kv`が層ごとの配列であるSSM/Attentionハイブリッドモデルでは、値が0のSSM層をKV計算から除外します。
  `full_attention_interval` を持つハイブリッドモデルは全長 Attention 層のみを、Sliding Window Attention(SWA)を
  使うモデル(Gemma 等)は SWA 層をウィンドウ分(`sliding_window` × スロット数 + `UBATCH_SIZE`)のみとして計算します。
  また量子化 KV では Flash Attention が K/V を f16 へ展開する作業領域(1層分 × 総トークン数)と KQ マスクが
  compute buffer に載るため、これが `VRAM_RESERVE_MIB` の半分を超える分も KV の所要量として加算します。
  1. **長いコンテキストの維持**: 自動検出では最大256Kを目標にし、ユーザーが指定した`CONTEXT_SIZE`は優先します。
  2. **モデル重みのGPU常駐**: 可能なら全レイヤーをGPUへ配置します。通常のVRAM計算で全重みを載せられない
     低アクティブ率MoEのみ、速度低下を許容できるCPU expert配置を検討します。
  3. **単一リクエストの速度**: 既定は1スロットで、並列リクエストより単一生成の速度を優先します。
  4. **KVキャッシュ型**: 既定の`q4_0`から自動で高精度型へ引き上げません。量子化KVの実速度はモデル/GPUに依存するため、
     比較したい場合は`KV_CACHE_TYPE`を明示して実測してください。

  複数GPUで`TENSOR_SPLIT_MODE=auto`の場合は、各GPUの空きVRAMと`llama-bench`の測定速度を使って`tensor-split`を
  計算します。単一GPUのMoEでRAMが十分でない場合、またはKV/重みの見積もりが不成立の場合は`--fit`へフォールバック
  することがあります。これらは容量確保のための代替経路であり、最大tok/sを実機測定で保証するものではありません。

## Continue (VS Code拡張) との連携

本サーバーは OpenAI API 互換の `GET /v1/models` と `POST /v1/chat/completions` を提供するため、
Continueの `openai` プロバイダから利用できます。
[continue/config.yaml](continue/config.yaml) に本サーバーを OpenAI API 互換プロバイダとして登録するサンプル設定があります。
`~/.continue/config.yaml` にコピーして使用してください。

```yaml
models:
  - name: Local Model (llama-server)
    provider: openai
    apiBase: http://localhost:11434/v1
    apiKey: none
    model: AUTODETECT
    roles:
      - chat
      - edit
      - apply
```

## トラブルシューティング

### `500 model name=... failed to load` (OOM)

- `docker logs my-llm-server` で `cudaMalloc failed: out of memory` が出ていないか確認してください。
- 既定ではQ4 KVを使用し、見積もりに基づいてコンテキスト長やGPU配置を決めますが、他プロセスがGPUを
  使用中で空きVRAMが少ない場合などはそれでも収まらないことがあります。その場合は `VRAM_RESERVE_MIB` を
  増やすか、`MAX_CONTEXT_SIZE` でコンテキスト長自体を制限してください。

### リクエストが `context size exceeded` 的なエラーになる

- 使用中のモデルの `ctx-size` を超えるトークン数をリクエストしていないか確認してください。
- `CONTEXT_SIZE` / `MAX_CONTEXT_SIZE` で明示的に増減できます。
- **個々のリクエストは上限未満なのにこのエラーが出る場合**は、同時実行リクエストの合算で KV キャッシュが
  溢れている可能性があります。`docker logs` でエラー直前の
  `slot release: ... stop processing: n_tokens = N` を複数スロット分合計し、`ctx-size` を超えていないか
  確認してください。本サーバは各スロットにコンテキスト全量を確保できる本数までしかスロットを開かないため
  通常は起きませんが、`MAX_PARALLEL_SLOTS=1` を指定すると確実に逐次実行へ倒せます。

### `llama-server backend request failed` エラーになる

- エラーメッセージの末尾に llama-server が返した理由が付加されます (ログにも
  `[llama-proxy] backend request failed: HTTP Error 400: ...: <理由>` として出力されます)。
- `Cannot have 2 or more assistant messages at the end of the list.` は、クライアントが
  assistant メッセージを連続して送った場合に llama-server が返すエラーです。Ollama 本家は受理するため、
  本プロキシは `/api/chat` と `/v1/chat/completions` の両方で連続する assistant メッセージを 1 つに
  結合してから転送します (ログに `merged N consecutive assistant message(s)` と出力)。

### 長時間のthinking後に `Sorry, no response was returned.` で終わる

- 思考型モデルがreasoningだけを生成し、本文もtool callも返さずに終了すると、クライアント側で
  このエラーになる可能性があります。長時間待ったという事実だけでは、タイムアウトとは断定できません。
- `/api/chat` は本文もtool callもない応答を正常終了にせず、
  `llama-server returned no assistant content or tool calls (finish_reason=...)` を返します。
  ストリーミング開始後はOllama形式の `{"error":"..."}` 行、非ストリーミングはHTTP 502です。
  非表示のreasoningを本文へ転用したり、自動再試行したりはしません。
- ログの `chat result: model=... content_chars=... tool_calls=... finish_reason=...` で
  可視本文とtool callの有無を確認できます。`finish_reason=length` ならクライアントの
  `options.num_predict` (出力上限) を確認してください。出力上限を増やすと待ち時間も増えます。
- thinkingを抑える選択肢は `THINKING_MODE=off ./launch-container.sh` です。ただし全モデルの
  回答の仕方・品質が変わるため、thinkingが必要かを判断したうえで明示的に設定してください。
  修正の反映にはイメージの再ビルド・コンテナ再作成が必要です。

### コンテナの状態確認

```bash
docker logs -f my-llm-server   # 起動ログ・エラーを確認
docker ps -a --filter name=my-llm-server
nvidia-smi                       # GPU の空き VRAM を確認
```

## 再ビルド・再起動

`launch-container.sh` はイメージの再ビルドと既存コンテナの削除・再作成を毎回行います。
スクリプトを修正した場合や設定を変更した場合は、変更したい環境変数を export した上で再実行してください。

```bash
export KV_CACHE_TYPE=q4_0
./launch-container.sh
```
