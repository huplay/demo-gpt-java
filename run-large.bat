@echo off
CHCP 65001
java -Xmx4G -Xms4G -cp target/demo-gpt-java-1.0.jar ai.demo.gpt.Application model=LARGE maxLength=10

