@echo off
CHCP 65001 >nul
if exist models/%1/setup.bat call models/%1/setup.bat
java %GPT_JAVA_ARGS% -jar app/target/demo-gpt-app.jar %*
