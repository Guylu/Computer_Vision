@echo off
cd /d %~dp0/..
call conda activate wis-cv
call jupyter notebook ex4-alignment.ipynb
