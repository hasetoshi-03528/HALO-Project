# H.A.L.O. Project

**H.A.L.O.（Hasetoshi's Advanced Logical Operator）**  
S.D.A.P.思想に基づき、純粋な論理で動作するHasetoshiの専用AIシステム。

---

## 概要

Gemma 3 12BをベースにLoRAファインチューニングを試みたが、環境上の制約により断念。  
現在はsystemプロンプト + 会話履歴による適応運用に切り替えている。

---

## 環境

- GPU: NVIDIA GeForce RTX 4070 Ti (VRAM 12GB)
- OS: Windows 11 + WSL2 Ubuntu 22.04
- ベースモデル: google/gemma-3-4b-it（LoRA学習）/ google/gemma-3-12b-it（運用）
- フレームワーク: Unsloth 2026.4.7, CUDA 12.8

---

## LoRAを断念した理由

### 12Bモデルでの学習断念
- 4bit量子化（NF4）でも学習時のVRAMが12GBを超過
- bitsandbytes（bnb）とPEFT最新版の互換性問題（`compress_statistics`属性エラー）
- UnslothのLoRAマージ処理にバグあり（`# of LoRAs = 319 does not match # of saved modules = 0`）

### 4Bモデルでの学習完了→運用断念
- 4BモデルではLoRA学習自体は完走（loss: 4.1 → 0.9）
- GGUF変換後にOllamaへのインポートは成功
- ただしGemma3がマルチモーダルアーキテクチャ（`Gemma3ForConditionalGeneration`）のため、テンソル名の不整合でGGUF変換が複雑化
- 最終的に4Bの表現力では12Bと比較して人格定着が不十分と判断

### 現在の運用方針
- モデル: gemma3:12b（Ollama）
- インターフェース: Open WebUI + SearXNG（ウェブ検索統合）
- 人格定義: systemプロンプトで管理
- 学習: 会話履歴の蓄積による適応

---

## S.D.A.P.思想

[**S.D.A.P. = Self-Designed with AI-assisted Protocol** ](https://github.com/hasetoshi-03528/reclaiming-sovereignty-through-SDAP) 

三原則：
1. 理解なき操作の拒絶
2. 一周回った後の自覚的省略
3. AIとの非対称な関係の維持

---

## ファイル構成
※dataディレクトリは.gitignoreで除外
```
halo-project/
├── README.md
├── train_halo.py          # 4B LoRA学習スクリプト
├── train_halo_12b.py      # 12B LoRA学習スクリプト（断念）
└── data/
├── halo_training_sample.json  # シードサンプル（非公開）
├── T-001.json                 # 学習データ（非公開）
├── T-002.json
├── T-003.json
├── T-004.json
└── T-005.json
```
---

## 今後の課題

- 12BモデルへのLoRA適用（VRAM 24GB以上の環境が必要）
- S.D.A.P.の体系化と文書化
- 会話履歴を用いたRAGパイプラインの構築
