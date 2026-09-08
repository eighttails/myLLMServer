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
│   └── start-llama.sh    # コンテナ ENTRYPOINT。モデル同期・preset生成・llama-server起動を行う
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

初回実行時、指定モデルが `models/` 配下になければ Hugging Face から自動ダウンロードされ、
`llama-bench` によるベンチマークが一度だけ実行されます(結果は `models/llama-bench-*.json` にキャッシュ)。

起動後は `http://localhost:8080/v1` が OpenAI 互換の API エンドポイントになります。

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-UD-Q4_K_M",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

`model` にはダウンロードした GGUF ファイル名から拡張子を除いたものを指定します
(例: `Llama-3.2-3B-Instruct-Q4_K_M`, `Qwen3.8-27B-UD-Q4_K_M`)。

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
| `PORT` | `8080` | 公開ポート |
| `PUID` / `PGID` | 実行ユーザーの uid/gid | コンテナ内プロセスの実行ユーザー(ダウンロードファイルの権限をホストと一致させる) |
| `CUDA_VISIBLE_DEVICES` | (未設定=全GPU) | 使用する GPU を限定したい場合に指定 |
| `MODEL_NAMES_CSV` | (未設定) | `リポジトリ/ファイル名.gguf` のカンマ区切りリスト。指定するとスクリプト内蔵の `MODEL_NAMES` を上書き |
| `HF_ENDPOINT` | `https://huggingface.co` | モデルダウンロード元エンドポイント |
| `MODEL_IDLE_SECONDS` | `300` | この秒数アイドルが続いたモデルは VRAM から解放される |
| `CONTEXT_SIZE` | (未設定=自動検出) | 全モデル共通のコンテキスト長を固定したい場合に指定。未指定時はモデルの GGUF メタデータ(`<arch>.context_length`)から推奨値を自動検出 |
| `MAX_CONTEXT_SIZE` | (未設定=上限なし) | 自動検出したコンテキスト長に上限をかけたい場合に指定(VRAM保護用) |
| `N_GPU_LAYERS` | `auto` | GPU に載せるレイヤー数。`auto`/`all`/数値を指定可能。`auto` の場合は後述の `--fit` に判断を委ねる |
| `MODELS_MAX` | `1` | 同時にロードしておくモデル数の上限(router mode) |
| `KV_CACHE_TYPE` | (未設定=f16相当) | KVキャッシュの量子化タイプ。`q8_0`(約1/2)や`q4_0`(約1/4)を指定すると大きなコンテキスト長でも VRAM 使用量を大幅に削減できる。allowed: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` |

## VRAM 管理の仕組み

- **アイドル時の自動解放**: 各モデルプロセスに `--sleep-idle-seconds` を設定しており、`MODEL_IDLE_SECONDS`
  で指定した時間アクセスが無いと自動的に VRAM を解放します。
- **OOM 回避 (`--fit`)**: `tensor-split` や `n-gpu-layers` を固定値で指定すると、llama-server 側の自動フィット機能
  (`--fit`, デフォルト有効) が「ユーザー指定済み」と判断して調整を放棄してしまい、VRAM に収まらない場合に
  OOM で起動失敗することがあります。そのため本ラッパーではこれらを固定せず、`--fit on` に委ねて
  GPU 間のレイヤー配置やコンテキストサイズを実行時の空き VRAM に応じて自動調整させています。
- **コンテキスト長の自動検出**: `CONTEXT_SIZE` を指定しない場合、`gguf-dump` を使って各モデルの GGUF メタデータから
  `<arch>.context_length`(モデルが学習時にサポートする最大コンテキスト長)を読み取り、`ctx-size` に設定します。
  VRAM が少ない環境では `MAX_CONTEXT_SIZE` や `KV_CACHE_TYPE=q4_0` などと組み合わせて調整してください。

## Continue (VS Code拡張) との連携

[continue/config.yaml](continue/config.yaml) に本サーバーを OpenAI 互換プロバイダとして登録するサンプル設定があります。
`~/.continue/config.yaml` にコピーして使用してください。

```yaml
models:
  - name: Local Model (llama-server)
    provider: openai
    apiBase: http://localhost:8080/v1
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
- `KV_CACHE_TYPE=q4_0` を設定するとコンテキスト長あたりの VRAM 使用量を約1/4に抑えられます。
- それでも収まらない場合は `MAX_CONTEXT_SIZE` でコンテキスト長自体を制限してください。

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
