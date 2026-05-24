@echo off
set PYTHON=python
@REM set SCRIPT="baseline code.py"
set SCRIPT="independent_task.py"

set TRAIN_PATH=inputs/train.csv
set TEST_PATH=inputs/test_new.csv
set SAMPLE_PATH=result/sample_submission.csv
set OUTPUT_PATH=result/submission.csv

REM --- 在下面宣告 script 的參數（留空表示不傳遞，程式會使用其內部預設值） ---
set LR=
set EPOCHS=
set BATCH_SIZE=
set MODEL=
set SEED=
REM 增加或修改為 baseline code.py 裡實際的參數名稱

REM --- 根據有無設定來組合參數字串 ---
set ARGS=
if defined LR set ARGS=%ARGS% --lr %LR%
if defined EPOCHS set ARGS=%ARGS% --epochs %EPOCHS%
if defined BATCH_SIZE set ARGS=%ARGS% --batch_size %BATCH_SIZE%
if defined MODEL set ARGS=%ARGS% --model %MODEL%
if defined SEED set ARGS=%ARGS% --seed %SEED%
REM 繼續為所有參數加入 if defined 行

if "%~1"=="--direct" (
    shift
    REM 把後面所有參數直接傳給 baseline code.py，例如：run.bat --direct --lr 0.001 --epochs 10
    %PYTHON% %SCRIPT% %*
) else (
    REM 預設行為：使用設定好的路徑與組合出的 ARGS
    %PYTHON% %SCRIPT% --train %TRAIN_PATH% --test %TEST_PATH% --sample %SAMPLE_PATH% --out %OUTPUT_PATH% %ARGS%
)
pause
