# myLLMServer

llama.cpp (llama-server) を Ollama API 互換および OpenAI API 互換のエンドポイントとして
Docker コンテナ上で動かすためのラッパーです。
複数の GGUF モデルをルーターモード (router mode) で切り替えながら提供し、一定時間使われていないモデルは
自動的に VRAM から解放されます。

## 特徴

- ホスト環境には Docker 以外の追加インストールが不要(モデルのダウンロード・配置もコンテナ内で完結)
- `model_list.txt` で指定した Hugging Face 上の GGUF モデルを自動ダウンロード・配置
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
│   ├── start-llama.sh    # コンテナ ENTRYPOINT。モデル同期・軽量preset生成・llama-server/proxy起動を行う
│   ├── sync-model.sh     # model_list.txt の読み込み、モデル自動ダウンロード・不要モデル削除を行う
│   ├── configure-model-preset.sh # モデル切替時に重い preset 計算を行う
│   ├── lazy-llama-proxy.py       # 公開ポートで受け、モデル切替時だけ preset を更新する
│   └── unload-model.sh           # コンテナ内からモデルをアンロードするスクリプト
├── launch-container.sh    # ホスト側から使う起動スクリプト(ビルド + コンテナ再作成)
├── unload-model.sh        # 外部(ホスト側)からモデルをアンロードするスクリプト
├── reload-model.sh        # model_list.txt を再ロードし、新規モデルの追加ダウンロード＆不要モデルの削除を行うスクリプト
├── model_list.txt         # 使用するモデルの指定ファイル (初回起動時に model_list.example から自動作成)
├── model_list.example     # モデル指定ファイルのサンプル
├── continue/
│   └── config.yaml        # Continue (VS Code拡張) 用のモデル設定サンプル
└── models/                # モデルダウンロード先 (.gitignore 済み、初回は空でOK)
```

## 前提条件

- Docker (NVIDIA Container Toolkit導入済み。`docker run --gpus all` が使えること)
- NVIDIA GPU (複数GPU可)

## 使い方

### 1. 起動

初回起動時、`model_list.txt` が存在しない場合は `model_list.example` から自動作成され、そこに記述されたモデルが使用されます。
**自分で使いたいモデルを指定する場合は、`model_list.txt` を編集してください。**

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
`llama-server` へ転送します。

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

自分が使いたいモデルを指定・変更する場合は、`model_list.txt` を編集します。
1行につき1つの Hugging Faceのダウンロードリンクを記述します（`#` で始まる行や空行は無視されます）。

```text
# model_list.txt の例
https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
https://huggingface.co/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf/resolve/main/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf
https://huggingface.co/Qwen/Qwen3.8-27B-UD-Q4_K_M.gguf/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf
```

編集後、コンテナを再起動せずに `model_list.txt` を再ロードして変更を即座に反映したい場合は、`./reload-model.sh` を実行します。

```bash
./reload-model.sh
```

このコマンド（または `./launch-container.sh`）を実行すると、以下の処理が自動で行われます:
- `model_list.txt` に新たに追加されたモデルを Hugging Face から自動ダウンロード
- インストール済みだが `model_list.txt` に記載のない（今後使わない）モデルをディスク（`models/`）から自動削除
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

| 変数名                       | デフォルト               | 説明                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `IMAGE_NAME`                 | `my-llm-server:latest`   | ビルドする Docker イメージ名                                                                                                                                                                                                                                                                                                                                                          |
| `CONTAINER_NAME`             | `my-llm-server`          | 作成するコンテナ名                                                                                                                                                                                                                                                                                                                                                                    |
| `MODEL_DIR`                  | `./models`               | モデルダウンロード先(ホスト側パス)                                                                                                                                                                                                                                                                                                                                                    |
| `PORT`                       | `11434`                  | 公開ポート                                                                                                                                                                                                                                                                                                                                                                            |
| `LLAMA_ROUTER_PORT`          | `PORT + 1`               | コンテナ内部の llama-server router 用ポート。通常は変更不要                                                                                                                                                                                                                                                                                                                           |
| `PUID` / `PGID`              | 実行ユーザーの uid/gid   | コンテナ内プロセスの実行ユーザー(ダウンロードファイルの権限をホストと一致させる)                                                                                                                                                                                                                                                                                                      |
| `CUDA_VISIBLE_DEVICES`       | (未設定=全GPU)           | 使用する GPU を限定したい場合に指定                                                                                                                                                                                                                                                                                                                                                   |
| `MODEL_LIST_FILE`            | `./model_list.txt`       | モデルリストを指定するテキストファイルのパス                                                                                                                                                                                                                                                                                                                                          |
| `MODEL_NAMES_CSV`            | (未設定)                 | `リポジトリ/ファイル名.gguf` のカンマ区切りリスト。指定すると `model_list.txt` より優先されます                                                                                                                                                                                                                                                                                       |
| `HF_ENDPOINT`                | `https://huggingface.co` | モデルダウンロード元エンドポイント                                                                                                                                                                                                                                                                                                                                                    |
| `http_proxy` / `https_proxy` / `no_proxy` (および大文字版、`all_proxy`) | (未設定=不使用)           | ホスト側のプロキシ設定をコンテナに引き継ぐ。モデルダウンロード(`sync-model.sh` の curl)に使用される。コンテナ内部の通信は常に `127.0.0.1` / `localhost` が `no_proxy` に追加されるためプロキシを绕回しない                                                                                                                                                                                                                       |
| `MODEL_IDLE_SECONDS`         | `1800`                   | この秒数(デフォルト30分)アイドルが続いたモデルは VRAM から解放される                                                                                                                                                                                                                                                                                                                  |
| `CONTEXT_SIZE`               | (未設定=自動検出)        | 全モデル共通のコンテキスト長を固定したい場合に指定。未指定時はモデルの GGUF メタデータ(`<arch>.context_length`)から推奨値を自動検出                                                                                                                                                                                                                                                   |
| `MAX_CONTEXT_SIZE`           | (未設定=上限なし)        | 自動検出したコンテキスト長に上限をかけたい場合に指定(VRAM保護用)                                                                                                                                                                                                                                                                                                                      |
| `MIN_CONTEXT_SIZE`           | `2048`                   | KV キャッシュが VRAM に収まらず自動でコンテキスト長を切り詰める際の下限。これを下回る場合のみ `--fit` にフォールバックする                                                                                                                                                                                                                                                            |
| `CONTEXT_SIZE_STEP`          | `1024`                   | コンテキスト長を自動で切り詰める際の丸め単位                                                                                                                                                                                                                                                                                                                                          |
| `N_GPU_LAYERS`               | `auto`                   | GPU に載せるレイヤー数。`auto`/`all`/数値を指定可能。`auto` の場合は後述の `--fit` に判断を委ねる                                                                                                                                                                                                                                                                                     |
| `MODELS_MAX`                 | `1`                      | 同時にロードしておくモデル数の上限(router mode)                                                                                                                                                                                                                                                                                                                                       |
| `FLASH_ATTN`                 | `on`                     | Flash Attentionの使用設定。`on`/`off`/`auto`を指定可能                                                                                                                                                                                                                                                                                                                                |
| `BATCH_SIZE`                 | `1024`                   | prompt処理の論理バッチサイズ。llama.cpp既定値の2048より小さくして一時的なVRAM使用量を抑制                                                                                                                                                                                                                                                                                             |
| `UBATCH_SIZE`                | `256`                    | prompt処理の物理バッチサイズ。llama.cpp既定値の512より小さくして計算バッファのVRAM使用量を抑制。`BATCH_SIZE`以下で指定                                                                                                                                                                                                                                                                |
| `VRAM_RESERVE_MIB`           | `4096`                   | `--fit`と手動`tensor-split`計算でGPUごとに確保する、compute bufferとCUDAワークスペースを含むランタイム用のVRAM余白(MiB)                                                                                                                                                                                                                                                               |
| `MOE_CPU_OFFLOAD`            | `auto`                   | MoE expert重みのCPU配置。`auto`はアクティブexpert比率が閾値以下の場合、KV確保後に収まらないexpert層だけCPUへ配置。`all`はすべてのMoEモデルで同じ調整を有効化、`off`は無効化                                                                                                                                                                                                           |
| `MOE_ACTIVE_RATIO_THRESHOLD` | `0.125`                  | `MOE_CPU_OFFLOAD=auto`でCPU配置を有効にする`expert_used_count / expert_count`の上限                                                                                                                                                                                                                                                                                                   |
| `MOE_RAM_RESERVE_MIB`        | `8192`                   | MoE expert重みをCPUへ配置した後も残すホストRAMの余白(MiB)                                                                                                                                                                                                                                                                                                                             |
| `KV_CACHE_TYPE`              | (未設定=自動選択)        | KVキャッシュの量子化タイプを固定したい場合に指定。KとVは常に同じ型(`cache-type-k` == `cache-type-v`)になるように保証されます。未指定時はモデル切替時に空き VRAM と対象モデルの GGUF メタデータから必要な KV キャッシュ量を見積もり、収まる範囲でなるべく精度の高いタイプ(`f16` → `q8_0` → `q4_0` の順)を自動選択する。allowed: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` |
| `TENSOR_SPLIT_MODE`          | `auto`                   | 複数 GPU 構成での層分割方法。`auto` の場合、モデル切替時に GPU 毎の生成速度と空き VRAM を実測し、対象モデルの性能比に応じた `tensor-split` を計算して高速化を図る(収まらない場合は自動的に `--fit` 任せへフォールバック)。`off` にすると常に `--fit` 任せの従来動作になる。`SPLIT_MODE` が `layer` 以外の場合はこの計算自体を行わない                                                     |
| `SPLIT_MODE`                 | `layer`                  | llama-server の `--split-mode`。複数 GPU 間でのモデル分割方式。`layer`(既定): レイヤー単位でGPUに分割するパイプライン並列。1トークン生成中は常にどれか1枚のGPUのみが計算するため、GPU使用率は50%前後(2GPU時)に留まりやすい仕様。`row`: 各レイヤーの重みを行単位でGPU間に分割するテンソル並列で、全GPUが同時に計算に参加できる。`tensor`: 重みとKVキャッシュの両方を分割(実験的)。`none`: 単一GPUのみ使用。**注意**: `row`/`tensor` は毎レイヤーGPU間の同期が発生するため、NVLink等の高速な相互接続が無い環境(PCIe経由のみ)や性能の異なるGPUの組み合わせでは、`layer` より遅くなることがある |

## VRAM 管理の仕組み

- **アイドル時の自動解放**: 各モデルプロセスに `--sleep-idle-seconds` を設定しており、`MODEL_IDLE_SECONDS`
  で指定した時間アクセスが無いと自動的に VRAM を解放します。
- **計算バッファと共有KVの省メモリ化**: Flash Attentionを有効にし、`BATCH_SIZE` / `UBATCH_SIZE`を
  llama.cppの既定値より小さくしています。また、`--kv-unified`により並列スロット間で単一のKVバッファを共有します。
  バッチサイズをさらに下げるとVRAMを節約できますが、長いpromptの処理速度は低下します。
- **低アクティブ率MoEの自動CPU配置**: GGUFの`expert_count`と`expert_used_count`を調べ、既定では
  1トークンあたりのアクティブexpert比率が12.5%以下なら、まず全expertをCPUへ置ける前提でKVキャッシュを確保します。
  その後`n-cpu-moe`を0から順に試し、KVキャッシュとVRAM余白を維持したまま成立する最小値を採用します。
  したがって、残ったVRAMには可能な限り多くのexpert重みが戻され、収まらない先頭側のexpert層だけがCPUへ配置されます。
  KVキャッシュはGPU間で均等と仮定せず、各GPUへ割り当てられるAttention層のKV head数に応じて計上します。
  必要なホストRAMを確保できない場合は自動的にCPU配置を見送ります。CPU配置したexpertの計算・転送により、
  生成速度が低下する可能性があります。
- **OOM 回避 (`--fit`)**: 基本方針として `tensor-split` や `n-gpu-layers` は固定値指定を避け、llama-server 側の
  自動フィット機能(`--fit`, デフォルト有効)に GPU 間のレイヤー配置やコンテキストサイズの調整を委ねています。
  これは、固定値を指定すると `--fit` が「ユーザー指定済み」と判断して調整を放棄し、VRAM に収まらない場合に
  OOM で起動失敗することがあるためです。
- **モデル切替時の遅延 preset 計算**: 公開ポートでは軽量プロキシがリクエストを受け、`model` が直前のモデルから
  変わった場合だけ対象モデルの preset を再計算して llama-server router に reload します。起動時は全モデルに対して
  KV キャッシュ量子化・GGUF テンソル解析・`tensor-split` 計算を行わないため、モデル数が増えても起動時間が伸びにくくなります。
- **GPU 性能比に基づく tensor-split 自動計算(`TENSOR_SPLIT_MODE=auto`、複数 GPU 時のみ)**: モデル切替時に
  GPU 毎の生成速度(tokens/sec)を `llama-bench` で実測し(結果はキャッシュされ、以後の切替では
  再利用される)その比率に応じてレイヤーを性能の高い GPU に多く割り当てる `tensor-split` を計算します。
  ベンチマークには単一 GPU に確実に収まる小サイズのモデルを使用します(各 GPU の空き VRAM が少なく、対象モデル
  単体ではロードできない場合があるため)。計算時には各 GPU の空き VRAM・GGUF のテンソル情報から求めた
  レヤー毎の重みサイズ・KV キャッシュの必要量を考慮し、OOM しない範囲に収まるように按分します。
  予算内に収まらない場合はログに警告を出したうえで手動指定を諦め、通常通り `--fit` 任せの自動調整に
  フォールバックします(preset ファイル内で `fit = off` / `fit = on` を切り替えることで実現しており、
  コマンドライン引数側では固定しません)。単一 GPU の場合や `TENSOR_SPLIT_MODE=off` の場合はこの計算は行わず、
  常に `--fit` 任せになります。
- **コンテキスト長の自動検出**: `CONTEXT_SIZE` を指定しない場合、モデル切替時に `gguf-dump` を使って対象モデルの GGUF メタデータから
  `<arch>.context_length`(モデルが学習時にサポートする最大コンテキスト長)を読み取り、`ctx-size` に設定します。
- **KV キャッシュ量子化の自動選択**: `KV_CACHE_TYPE` を指定しない場合、モデル切替時に `nvidia-smi` で取得した空き VRAM 合計から
  モデルファイルサイズを差し引いた「予算」を計算し、モデルの GGUF メタデータ(`block_count` /
  `attention.head_count_kv` / `attention.key_length` / `attention.value_length`)から算出した必要 KV キャッシュ量と
  比較して、予算に収まる範囲でなるべく精度の高いタイプ(`f16` → `q8_0` → `q4_0` の順)を自動選択します。
  `head_count_kv`が層ごとの配列であるSSM/Attentionハイブリッドモデルでは、値が0のSSM層をKV計算から除外します。
- **コンテキスト長の自動切り詰め**: 最も軽い `q4_0` でも KV キャッシュが予算に収まらない場合は、`--fit` に
  切り替えるのではなく `q4_0` のまま予算に収まるところまで `ctx-size` を切り詰めます(`CONTEXT_SIZE_STEP`
  の倍数に丸め、`MIN_CONTEXT_SIZE` を下限とします)。KV キャッシュの概算では収まっていても、GPU ごとの
  重み配置と固定予約を含めると `tensor-split` が成立しない場合も、成立する最大のコンテキスト長を探索します。
  モデルの重み自体が空き VRAM に収まらない場合のみ、KV キャッシュ・`tensor-split` の手動指定を諦めて
  `--fit` 任せにフォールバックします。

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
- 通常は KV キャッシュ量子化が空き VRAM から自動選択されるため発生しにくいですが、他プロセスが GPU を
  使用中で空き VRAM が少ない場合などはそれでも収まらないことがあります。その場合は `KV_CACHE_TYPE=q4_0`
  を明示指定するか、`MAX_CONTEXT_SIZE` でコンテキスト長自体を制限してください。

### リクエストが `context size exceeded` 的なエラーになる

- 使用中のモデルの `ctx-size` を超えるトークン数をリクエストしていないか確認してください。
- `CONTEXT_SIZE` / `MAX_CONTEXT_SIZE` で明示的に増減できます。

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
