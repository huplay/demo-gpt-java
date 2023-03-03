@echo off
CHCP 65001
java -Xmx2G -Xms2G -cp target/demo-gpt-java-1.0.jar ai.demo.gpt.Application model=MEDIUM maxLength=10

