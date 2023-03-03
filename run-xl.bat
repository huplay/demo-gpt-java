@echo off
CHCP 65001
java -Xmx7G -Xms7G -cp target/demo-gpt-java-1.0.jar ai.demo.gpt.Application model=XL maxLength=10

