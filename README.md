# 2026_AI_CUP
重建環境變數（anaconda，我是照hw4建的）：
conda env create -f environment.yml
conda activate myenv

# 使用run.bat跑模型
可以指定參數，如果沒有指定，會使用預設
- 參數：

    `--train`（預設: `inputs/train.csv`）  
        說明：訓練資料檔案路徑，用於載入模型訓練所需的資料。

    `--test`（預設: `inputs/test_new.csv`）  
        說明：測試資料檔案路徑，用於推論或產生提交檔的輸入資料。

    `--sample`（預設: `result/sample_submission.csv`）  
        說明：範例 submission 檔案路徑（用於參考格式）。

    `--out`（預設: `result/submission.csv`）  
        說明：輸出 submission 檔案路徑，模型訓練或推論後產生的最終提交結果會寫入此檔案。

    `--epochs`（預設: `9`）  
        說明：訓練輪數（epochs），控制整個訓練資料被完整迭代的次數。

    `--batch`（預設: `64`）  
        說明：批次大小（batch size），每次參數更新所使用的樣本數。

    `--emb`（預設: `16`）  
        說明：嵌入向量維度（embedding size），用於文字或類別特徵的向量化維度。

    `--hidden`（預設: `128`）  
        說明：隱藏層單元數（hidden size），例如 RNN 或全連接層的神經元數量。

    `--layers`（預設: `1`）  
        說明：模型層數（例如 RNN/Transformer 的堆疊層數）。

    `--drop`（預設: `0.2`）  
        說明：Dropout 比例（0~1），用於正則化以降低過擬合風險。

    `--lr`（預設: `1e-4`）  
        說明：學習率（learning rate），控制參數更新步長。

    `--val_size`（預設: `0.10`）  
        說明：驗證集比例，從訓練資料中劃分作為驗證集的比率（0~1）。

    `--max_len`（預設: `512`）  
        說明：序列最大長度（token/字數上限），超過此長度時會截斷或填充。

    `--cap`（預設: `11`）  
        說明：整數型上限參數（例如截斷、分箱或類別數限制等用途），具體用途依實作而定。
- 使用範例：
    run.bat
    run.bat --train "inputs/train.csv" --test "inputs/test_new.csv" --epochs 9 --batch 16


# 使用save.bat紀錄版本
會自動建立backup 資料夾，他會把目標程式碼（預設是"baseline code.py"、"result/submission.csv"、 upload_result.txt）自動寫到history裡面，當作一次上傳版本的紀錄
- 參數：
    - `--codeName`, `-c`（預設: `baseline code.py`）  
        說明：要備份或上傳的主要程式檔案路徑／檔名。預設為專案根目錄下的 "baseline code.py"。如果檔名或路徑包含空格，請用引號包起來。

    - `--submission`, `-s`（預設: `result/submission.csv`）  
        說明：要上傳或紀錄的 submission 檔案路徑，預設放在 `result/submission.csv`。

    - `--history`, `-H`（預設: `history`）  
        說明：備份記錄存放的資料夾名稱或路徑。執行 save 動作時會將指定的檔案複製到此資料夾下作為版本歷史。

    - `--upload`, `-u`（預設: `0`）  
        說明：寫入到 `upload_result.txt` 的數字，用來表示上傳或版本標記的狀態／識別碼。預設為 `0`（表示未上傳或預設狀態），如果你有上傳可以寫結果在這裡。
- 使用範例：
    save.bat
    save.bat --codeName "baseline code.py" --submission "result/submission.csv" --history "./history" --upload 0
    


