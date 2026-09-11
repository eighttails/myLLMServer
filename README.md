# myLLMServer

llama.cpp (llama-server) を OpenAI API 互換のエンドポイントとして Docker コンテナ上で動かすためのラッパーです。
複数の GGUF モデルをルーターモード (router mode) で切り替えながら提供し、一定時間使われていないモデルは
自動的に VRAM から解放されます。

## 特徴

- ホスト環境には Docker 以外の追加インストールが不要(モデルのダウンロード・配置もコンテナ内で完結)
- `MODEL_NAMES_CSV` で指定した Hugging Face 上の GGUF モデルをコマンドラインだけでダウンロード・配置
- リストにないモデル(キャッシュ済みファイル)は起動時に自動削除
- `CUDA_VISIBLE_DEVICES` を参照し、未設定なら全 GPU を使用
- 一定時間(デフォルト5分、環境変数で変更可)アイドルなモデルは VRAM を自動解放
- ホスト側の UID/GID でコンテナを実行するため、ダウンロードしたモデルファイルをホスト側から root 権限なしに削除可能
- モデルの GGUF メタデータから推奨コンテキスト長を自動検出し、GPU の空き VRAM に応じて自動フィット(OOM 回避)

## 構成

```
.
├── docker/
│   ├── Dockerfile        # llama.cpp:full-cuda ベースイメージ + ラッパースクリプト
│   ├── start-llama.sh    # コンテナ ENTRYPOINT。モデル同期・軽量preset生成・llama-server/proxy起動を行う
│   ├── configure-model-preset.sh # モデル切替時に重い preset 計算を行う
│   └── lazy-llama-proxy.py       # 公開ポートで受け、モデル切替時だけ preset を更新する
├── run-llama.sh           # ホスト側から使う起動スクリプト(ビルド + コンテナ再作成)
├── continue/
│   └── config.yaml        # Continue (VS Code拡張) 用のモデル設定サンプル
└── models/                # モデルダウンロード先 (.gitignore 済み、初回は空でOK)
```

## 前提条件

- Docker (NVIDIA Container Toolkit導入済み。`docker run --gpus all` が使えること)
- NVIDIA GPU (複数GPU可)

## 使い方

### 1. モデルを指定して起動

デフォルトでは [docker/start-llama.sh](docker/start-llama.sh) 内の `MODEL_NAMES` に定義された以下の2モデルを使用します。

- `bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`

```bash
./run-llama.sh
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
    "model": "Qwen3.8-27B-UD-Q4_K_M",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

`model` にはダウンロードした GGUF ファイル名から拡張子を除いたものを指定します
(例: `Llama-3.2-3B-Instruct-Q4_K_M`, `Qwen3.8-27B-UD-Q4_K_M`)。

### VS Code Copilot Chat で使う場合

Copilot Chat では **Custom endpoint** として `http://localhost:11434/v1` (OpenAI 互換) を
登録するのが基本です。`Local` / Ollama プロバイダを選ぶ場合は、モデル一覧の検出に
`http://localhost:11434/api/tags`、チャット送信に `http://localhost:11434/api/chat`
(Ollama 互換、内部で OpenAI 互換 API に変換) を使用できます。

### 2. モデルリストを変更する

`MODEL_NAMES_CSV` 環境変数で、`リポジトリ名/ファイル名.gguf` をカンマ区切りで指定すると、
[docker/start-llama.sh](docker/start-llama.sh) 内の `MODEL_NAMES` を上書きできます。
このリストにないモデルファイルが `models/` にキャッシュされていた場合は起動時に自動削除されます。

```bash
MODEL_NAMES_CSV="bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf" ./run-llama.sh
```

### 3. モデル保存先を変更する

```bash
MODEL_DIR=/path/to/your/models ./run-llama.sh
```

未指定の場合は `./models` が使われます。

## 環境変数一覧

`run-llama.sh` 実行前に環境変数を export しておくと、コンテナに引き継がれます。

| 変数名 | デフォルト | 説明 |
|---|---|---|
| `IMAGE_NAME` | `my-llama-server:latest` | ビルドする Docker イメージ名 |
| `CONTAINER_NAME` | `my-llama-server` | 作成するコンテナ名 |
| `MODEL_DIR` | `./models` | モデルダウンロード先(ホスト側パス) |
| `PORT` | `11434` | 公開ポート |
| `LLAMA_ROUTER_PORT` | `PORT + 1` | コンテナ内部の llama-server router 用ポート。通常は変更不要 |
| `PUID` / `PGID` | 実行ユーザーの uid/gid | コンテナ内プロセスの実行ユーザー(ダウンロードファイルの権限をホストと一致させる) |
| `CUDA_VISIBLE_DEVICES` | (未設定=全GPU) | 使用する GPU を限定したい場合に指定 |
| `MODEL_NAMES_CSV` | (未設定) | `リポジトリ/ファイル名.gguf` のカンマ区切りリスト。指定するとスクリプト内蔵の `MODEL_NAMES` を上書き |
| `HF_ENDPOINT` | `https://huggingface.co` | モデルダウンロード元エンドポイント |
| `MODEL_IDLE_SECONDS` | `300` | この秒数アイドルが続いたモデルは VRAM から解放される |
| `CONTEXT_SIZE` | (未設定=自動検出) | 全モデル共通のコンテキスト長を固定したい場合に指定。未指定時はモデルの GGUF メタデータ(`<arch>.context_length`)から推奨値を自動検出 |
| `MAX_CONTEXT_SIZE` | (未設定=上限なし) | 自動検出したコンテキスト長に上限をかけたい場合に指定(VRAM保護用) |
| `MIN_CONTEXT_SIZE` | `2048` | KV キャッシュが VRAM に収まらず自動でコンテキスト長を切り詰める際の下限。これを下回る場合のみ `--fit` にフォールバックする |
| `CONTEXT_SIZE_STEP` | `1024` | コンテキスト長を自動で切り詰める際の丸め単位 |
| `N_GPU_LAYERS` | `auto` | GPU に載せるレイヤー数。`auto`/`all`/数値を指定可能。`auto` の場合は後述の `--fit` に判断を委ねる |
| `MODELS_MAX` | `1` | 同時にロードしておくモデル数の上限(router mode) |
| `KV_CACHE_TYPE` | (未設定=自動選択) | KVキャッシュの量子化タイプを固定したい場合に指定。未指定時はモデル切替時に空き VRAM と対象モデルの GGUF メタデータから必要な KV キャッシュ量を見積もり、収まる範囲でなるべく精度の高いタイプ(`f16` → `q8_0` → `q4_0` の順)を自動選択する。allowed: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` |
| `TENSOR_SPLIT_MODE` | `auto` | 複数 GPU 構成での層分割方法。`auto` の場合、モデル切替時に GPU 毎の生成速度と空き VRAM を実測し、対象モデルの性能比に応じた `tensor-split` を計算して高速化を図る(収まらない場合は自動的に `--fit` 任せへフォールバック)。`off` にすると常に `--fit` 任せの従来動作になる |

## VRAM 管理の仕組み

- **アイドル時の自動解放**: 各モデルプロセスに `--sleep-idle-seconds` を設定しており、`MODEL_IDLE_SECONDS`
  で指定した時間アクセスが無いと自動的に VRAM を解放します。
- **OOM 回避 (`--fit`)**: 基本方針として `tensor-split` や `n-gpu-layers` は固定値指定を避け、llama-server 側の
  自動フィット機能(`--fit`, デフォルト有効)に GPU 間のレイヤー配置やコンテキストサイズの調整を委ねています。
  これは、固定値を指定すると `--fit` が「ユーザー指定済み」と判断して調整を放棄し、VRAM に収まらない場合に
  OOM で起動失敗することがあるためです。
- **モデル切替時の遅延 preset 計算**: 公開ポートでは軽量プロキシがリクエストを受け、`model` が直前のモデルから
  変わった場合だけ対象モデルの preset を再計算して llama-server router に reload します。起動時は全モデルに対して
  KV キャッシュ量子化・GGUF テンソル解析・`tensor-split` 計算を行わないため、モデル数が増えても起動時間が伸びにくくなります。
- **GPU 性能比に基づく tensor-split 自動計算(`TENSOR_SPLIT_MODE=auto`、複数 GPU 時のみ)**: モデル切替時に対象
  モデルを使って GPU 毎の生成速度(tokens/sec)を `llama-bench` で実測し(結果はキャッシュされ、以後の切替では
  再利用されます)、その比率に応じてレイヤーを性能の高い GPU に多く割り当てる `tensor-split` を計算します。
  計算時には各 GPU の空き VRAM・GGUF のテンソル情報から求めたレイヤー毎の重みサイズ・KV キャッシュの
  必要量を考慮し、OOM しない範囲に収まるように按分します。予算内に収まらない場合はログに警告を出したうえで
  手動指定を諦め、通常通り `--fit` 任せの自動調整にフォールバックします(preset ファイル内で `fit = off` /
  `fit = on` を切り替えることで実現しており、コマンドライン引数側では固定しません)。
  単一 GPU の場合や `TENSOR_SPLIT_MODE=off` の場合はこの計算は行わず、常に `--fit` 任せになります。
- **コンテキスト長の自動検出**: `CONTEXT_SIZE` を指定しない場合、モデル切替時に `gguf-dump` を使って対象モデルの GGUF メタデータから
  `<arch>.context_length`(モデルが学習時にサポートする最大コンテキスト長)を読み取り、`ctx-size` に設定します。
- **KV キャッシュ量子化の自動選択**: `KV_CACHE_TYPE` を指定しない場合、モデル切替時に `nvidia-smi` で取得した空き VRAM 合計から
  モデルファイルサイズを差し引いた「予算」を計算し、モデルの GGUF メタデータ(`block_count` /
  `attention.head_count_kv` / `attention.key_length` / `attention.value_length`)から算出した必要 KV キャッシュ量と
  比較して、予算に収まる範囲でなるべく精度の高いタイプ(`f16` → `q8_0` → `q4_0` の順)を自動選択します。
- **コンテキスト長の自動切り詰め**: 最も軽い `q4_0` でも KV キャッシュが予算に収まらない場合は、`--fit` に
  切り替えるのではなく `q4_0` のまま予算に収まるところまで `ctx-size` を切り詰めます(`CONTEXT_SIZE_STEP`
  の倍数に丸め、`MIN_CONTEXT_SIZE` を下限とします)。KV キャッシュの概算では収まっていても、GPU ごとの
  重み配置と固定予約を含めると `tensor-split` が成立しない場合も、成立する最大のコンテキスト長を探索します。
  モデルの重み自体が空き VRAM に収まらない場合のみ、KV キャッシュ・`tensor-split` の手動指定を諦めて
  `--fit` 任せにフォールバックします。

## Continue (VS Code拡張) との連携

[continue/config.yaml](continue/config.yaml) に本サーバーを OpenAI 互換プロバイダとして登録するサンプル設定があります。
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

- `docker logs my-llama-server` で `cudaMalloc failed: out of memory` が出ていないか確認してください。
- 通常は KV キャッシュ量子化が空き VRAM から自動選択されるため発生しにくいですが、他プロセスが GPU を
  使用中で空き VRAM が少ない場合などはそれでも収まらないことがあります。その場合は `KV_CACHE_TYPE=q4_0`
  を明示指定するか、`MAX_CONTEXT_SIZE` でコンテキスト長自体を制限してください。

### リクエストが `context size exceeded` 的なエラーになる

- 使用中のモデルの `ctx-size` を超えるトークン数をリクエストしていないか確認してください。
- `CONTEXT_SIZE` / `MAX_CONTEXT_SIZE` で明示的に増減できます。

### コンテナの状態確認

```bash
docker logs -f my-llama-server   # 起動ログ・エラーを確認
docker ps -a --filter name=my-llama-server
nvidia-smi                       # GPU の空き VRAM を確認
```

## 再ビルド・再起動

`run-llama.sh` はイメージの再ビルドと既存コンテナの削除・再作成を毎回行います。
スクリプトを修正した場合や設定を変更した場合は、変更したい環境変数を export した上で再実行してください。

```bash
export KV_CACHE_TYPE=q4_0
./run-llama.sh
```
