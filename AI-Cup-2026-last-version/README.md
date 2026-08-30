# AI-Cup-2026 code

### 執行方式
- baseline_code.py
```shell
python .\baseline_code.py --epochs 10
```
- baseline_modify.py
```shell
python .\baseline_modify.py --epochs 30 --emb 32 --layers 2 --out "output/baseline_modify_2.csv"
```
☆ 混合方法
1. 訓練 BiLSTM -> 預測下一拍的球種 & 最終勝負
```shell
python .\try_BiLSTM_2.py
```
2. 訓練 XGBoost -> 預測球下一拍的落點
```shell
python .\train_point_only.py
```
3. 融合兩者的預測結果 (合併 csv)
```shell
python .\combine_csv.py
```

### 檔案結構
```
CODE
|
├── baseline_code.py
├── baseline_modify.py
├── try_BiLSTM_2.py
├── train_point_only.py
├── combine_csv.py
│
├── dataset/
│   ├── Reference_Only_Old_Test_Data
│   │   ├── README.txt
│   │   └── test.csv
│   ├── sample_submission.csv
│   ├── test_new.csv
│   └── train.csv
│
└── output/
    ├── baseline_modify.csv
    ├── submission_lstm_baseline.csv
    ├── baseline_bilstm2_act_pt_rly.csv
    ├── train_point_only.csv
    └── BaseMod3_TPO_2.csv
```
