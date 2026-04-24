# H.A.L.O. Project

**H.A.L.O.（Hasetoshi's Advanced Logical Operator）**  
S.D.A.P.思想に基づき、純粋な論理で動作するHasetoshiの専用AIシステム。

---

## 概要

Gemma3 4B（g34b）をベースモデルとし、Self-Playカリキュラム学習による
ファインチューニングを実施中。  
Groq API（Llama3.3 70B）とH.A.L.O.（halo:latest）を対話させ、
学習データを自動生成する。

生成したデータは以下の2つのモデルに適用予定：
- **g34b（gemma3 4B）**: ローカル環境（RTX 4070Ti）でファインチューニング
- **g3CB（gemma3 12B）**: Google Colab（Tesla T4）でファインチューニング

同一データで両モデルを並行して鍛え、表現力と推論能力の向上を目指す。

---

## 環境

- GPU: NVIDIA GeForce RTX 4070 Ti (VRAM 12GB)
- OS: Windows 11
- ベースモデル: gemma3:4b（g34b / halo:latest）
- 運用モデル: gemma3:12b（g3CB）
- フレームワーク: Unsloth, Ollama, Groq API
- データ生成: Groq（llama-3.3-70b-versatile）× H.A.L.O.（halo:latest）

---

## Self-Playカリキュラム学習計画

### Self-Playのデータ生成:

試運転 → halo:latest（g34b）

本番   → gemma3:12b（g3CB）

### フェーズ設計

| フェーズ | 期間 | かかる日数 | データ種類 | 難易度 | 1日の回数 |
|---------|------|-----------|-----------|--------|----------|
| Phase1 | 試運転3日 + 本番16日 | 19日 | 説明・要約 | 易 | 10回 |
| Phase2 | 試運転3日 + 本番16日 | 19日 | 質問応答 | 中 | 10回 |
| Phase3 | 試運転3日 + 本番16日 | 19日 | 逆質問 | 中 | 10回 |
| Phase4 | 試運転3日 + 本番16日 | 19日 | 批判的対話 | 難 | 10回 |
| Phase5 | 試運転3日 + 本番20日 | 23日 | 全種類混合 | 全 | 2回 |
| **合計** | - | **約99日** | **5種類** | 易→難 | - |

### 試運転プロセス（各フェーズ共通）

各フェーズ開始前に3日間の試運転を実施します。

| 日程 | 内容 | 判定基準 |
|------|------|---------|
| 1日目 | スクリプト動作確認 + 褒め合い出現率カウント | エラー・文字化け・タイムアウトがないか |
| 2日目 | 10回以上実行 + 褒め合い出現率カウント + 精度確認 | 褒め合い出現率10%以下 かつ 回答の深さ・批判の質がOK |
| 3日目 | 本番同条件で10回実行 | 問題なければ翌日から本番GO |

いずれかの日程で問題が発生した場合はプロンプトを修正して再試運転します。
トークン上限に達して試運転が完了できなかった場合は、試運転期間を延長します。

### 総データ量
Phase1〜4: 各960ペア
Phase5: 960ペア
合計: 4,800ペア

### トークン設計
1回の実行: 約9,000トークン（3ターン）
1日の上限: 90,000トークン（Groq無料枠）
Phase1〜4: 10回/日
Phase5: 2回/日（全種類混合のため36,000トークン/回）

---

## ファイル構成
```
halo-selfplay/
├── test_run_p1.py      # Phase1 試運転スクリプト
├── test_run_p2.py      # Phase2 試運転スクリプト
├── test_run_p3.py      # Phase3 試運転スクリプト
├── test_run_p4.py      # Phase4 試運転スクリプト
├── test_run_p5.py      # Phase5 試運転スクリプト
├── self_run_p1.py      # Phase1 本番スクリプト（説明・要約）
├── self_run_p2.py      # Phase2 本番スクリプト（質問応答）
├── self_run_p3.py      # Phase3 本番スクリプト（逆質問）
├── self_run_p4.py      # Phase4 本番スクリプト（批判的対話）
├── self_run_p5.py      # Phase5 本番スクリプト（全種類混合）
├── run_selfplay.bat    # タスクスケジューラ起動スクリプト
├── data/               # Self-Playデータ（JSONL）
├── logs/               # 実行ログ
│   ├── log_p1_day1.log
│   ├── log_p1_day2.log
│   └── ...
└── state.json          # 進捗管理
```
---

## Colab環境（g3CB ファインチューニング）

- GPU: Tesla T4 15GB（Google Colab無料枠）
- フレームワーク: Unsloth
- 対象モデル: gemma3:12b（g3CB）
- 保存先: Google Drive（空き容量 約12GB）

---

## モデル命名規則

| 名称 | 正式名 | 用途 |
|------|--------|------|
| g34b | gemma3 4B | H.A.L.O.のベースモデル・ローカルFT対象 |
| g3CB | gemma3 12B | ColabでのFT対象（16進数でC=12） |
| halo:latest | gemma3 4B + システムプロンプト | 現在運用中のH.A.L.O. |

---

## 過去の試行と断念記録

### 12Bモデルでの学習断念
- 4bit量子化（NF4）でも学習時のVRAMが12GBを超過
- bitsandbytesとPEFT最新版の互換性問題
- UnslothのLoRAマージ処理にバグあり

### 4Bモデルでの学習完了→運用断念（第1フェーズ）
- LoRA学習は完走（loss: 4.1 → 0.9）
- GGUF変換後にOllamaへのインポートは成功
- 4Bの表現力では人格定着が不十分と判断
- → Self-Playカリキュラム学習へ方針転換

---

## S.D.A.P.思想

[**S.D.A.P. = Self-Designed with AI-assisted Protocol**](https://github.com/hasetoshi-03528/reclaiming-sovereignty-through-SDAP)

三原則：

1. 理解なき操作の拒絶
2. 一周回った後の自覚的省略
3. AIとの非対称な関係の維持

---

## 今後の課題

- Phase1〜5の順次実行と品質評価
- FT後のH.A.L.O.応答品質の定量評価
- g3CBのColab学習実行
- S.D.A.P.の体系化と文書化
