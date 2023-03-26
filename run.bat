@echo off
CHCP 65001 >nul
if exist models/%1/setup.bat call models/%1/setup.bat
java %GPT_JAVA_ARGS% -cp target/demo-gpt-java-1.0.jar ai.demo.gpt.App %*

