# UPW 強化學習再生排程系統

這是在本機從 0 建構的 UPW 強化學習再生排程系統。專案以 `化工廠.py` 作為主要執行入口，核心流程放在 `upw_rl_system/` package 中，用於從 UPW 製程歷史 CSV 訓練數位雙生模型，接著用 PPO 強化學習產生混床再生排程。

## 系統結構

```text
upw_rl_system/
  config.py       # 訓練、模擬與輸出設定
  data.py         # CSV 驗證、篩選、標準化與 transition 建立
  model.py        # PINN/LSTM 數位雙生模型
  environment.py  # Gymnasium 模擬環境
  pipeline.py     # 端到端訓練與排程輸出
  cli.py          # 命令列介面
  __main__.py     # python -m upw_rl_system 入口
化工廠.py          # 專案主程式入口
```

## 安裝

建議使用虛擬環境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## CSV 欄位需求

輸入資料必須包含以下欄位：

| 欄位 | 說明 |
| --- | --- |
| `arm` | 製程線名稱，預設篩選 `Legacy_2passRO_MB` |
| `MB_regen` | 歷史再生動作，值為 `Yes` 時轉成 action `1`，其他值轉成 action `0` |
| `feed_SDI` | 進水 SDI |
| `feed_conductivity_uScm` | 進水導電度 |
| `resistivity_MOhm_cm` | 電阻率 |
| `TOC_ppb` | TOC |
| `cation_leak_ppb` | 陽離子洩漏 |
| `anion_leak_ppb` | 陰離子洩漏 |
| `deltaP_bar` | 壓差 |

缺少欄位時，系統會回報缺少的欄位名稱。特徵欄位中的缺值會先 forward fill，再以 `0` 補齊。

## 使用方式

使用專案主入口：

```powershell
python .\化工廠.py --csv .\data\upw_history.csv
```

或使用 package entry point：

```powershell
python -m upw_rl_system --csv .\data\upw_history.csv
```

常用參數：

```powershell
python .\化工廠.py `
  --csv .\data\upw_history.csv `
  --arm Legacy_2passRO_MB `
  --output-dir .\outputs `
  --pinn-epochs 250 `
  --ppo-timesteps 40000 `
  --max-steps 52 `
  --seed 42
```

## 輸出

預設輸出到 `outputs/`：

| 檔案 | 說明 |
| --- | --- |
| `digital_twin.pt` | PINN/LSTM 數位雙生模型權重與設定 |
| `preprocessing.pkl` | 標準化器、特徵名稱與資料設定 |
| `ppo_policy.zip` | Stable-Baselines3 PPO 策略模型 |
| `schedule.csv` | 模擬排程，包含 week、action、real_toc、reward、no_maint_weeks |

如果只想在畫面輸出結果、不保存 artifacts，可以加上：

```powershell
python .\化工廠.py --csv .\data\upw_history.csv --no-save
```

## 工作流程

1. 讀取 CSV 並篩選指定 `arm`。
2. 將製程特徵標準化，並把 `MB_regen` 轉成二元動作。
3. 建立一週一步的 state transition 訓練資料。
4. 訓練 PINN/LSTM 數位雙生模型預測下一週製程狀態。
5. 用數位雙生模型建立 Gymnasium 環境。
6. 用 PPO 訓練再生策略。
7. 以 deterministic policy rollout 輸出模擬再生週次。

## 限制

- 這是資料驅動的模擬與決策輔助系統，不是可直接上線控制設備的安全控制系統。
- 模型品質取決於 CSV 歷史資料品質、資料量與製程狀態覆蓋範圍。
- 建議先用短 epoch 與短 timesteps 做 smoke test，再提高訓練量。
