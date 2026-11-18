@echo off
set PYTHON=.venv\python.exe
set PYTEST=.venv\python.exe -m pytest

if "%1"=="" goto all
if "%1"=="all" goto all
if "%1"=="data" goto data
if "%1"=="test" goto test
if "%1"=="evaluate" goto evaluate
if "%1"=="report" goto report

:all
%PYTHON% -m src.run_all
goto end

:data
%PYTHON% -m src.data.prepare_data
goto end

:test
%PYTEST% tests/
goto end

:evaluate
%PYTHON% -m src.eval.run_benchmark
goto end

:report
%PYTHON% -m src.eval.generate_reports
goto end

:end
