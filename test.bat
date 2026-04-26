@echo off
set CMD=%1
set DATASET=%2

if "%CMD%"=="" (
    echo Usage: test [spx^|webp^|jxl^|bench] [dataset]
    echo Example: test bench gold
    exit /b 1
)

if "%DATASET%"=="" (
    set DATASET=gold
)

:: Validate CMD
if "%CMD%"=="spx" goto run
if "%CMD%"=="webp" goto run
if "%CMD%"=="jxl" goto run
if "%CMD%"=="bench" goto run

echo Error: Unknown command "%CMD%". Valid: spx, webp, jxl, bench.
exit /b 1

:run
python -m spx.test_suite %CMD% %DATASET% %3 %4 %5 %6 %7 %8 %9
