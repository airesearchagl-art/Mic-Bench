# Mic-Bench 🎤

**Mic-Bench** は、複数のマイク入力の精度をリアルタイムで比較・検証するためのベンチマークツールです。
特に音声認識（Whisper）の精度、S/N 比、レイテンシを同一条件で横並び評価することに特化しています。

## 🌟 主な機能

- **最大4デバイス同時キャプチャ** — Windows のオーディオデバイスを並列で録音・解析
- **リアルタイム文字起こし** — `faster-whisper` を使用し、デバイスごとの認識精度を視覚化
- **定量指標の算出**:
  - **SNR (S/N 比)** — 独自のノイズフロア推定による信号対雑音比
  - **Confidence** — AI による文字起こしの確信度（%）
  - **Latency** — 音声入力から文字化までの推論時間
- **レポート出力** — セッション統計データをタイムスタンプ付き CSV でエクスポート
- **GPU 自動選択** — CUDA 対応 GPU があれば自動でGPU推論（なければ CPU/int8 へ自動フォールバック）

## 🖥 動作環境

| 項目 | 要件 |
| :--- | :--- |
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.10 以上 |
| GPU | CUDA 対応 GPU（任意。なくても CPU で動作） |

## 🛠 セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

> **PyAudio のインストールが失敗する場合**
> Windows 環境では公式バイナリが提供されないため、`pipwin` 経由でインストールしてください。
>
> ```bash
> pip install pipwin && pipwin install pyaudio
> ```

### 2. 起動

```bash
python mic_bench.py
```

## 📦 主な依存パッケージ

| パッケージ | 役割 |
| :--- | :--- |
| `PyQt6` | UI フレームワーク |
| `PyAudio` | マイク入力ストリーミング |
| `faster-whisper` | 音声認識（CTranslate2 バックエンド） |
| `scipy` | 音声リサンプリング（`resample_poly`） |
| `numpy` | 数値演算・RMS 計算 |

## 📖 詳細ガイド

操作手順・各指標の読み方・比較のコツ・トラブルシューティングは [Manual.md](Manual.md) を参照してください。
