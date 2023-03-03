@echo off
CHCP 65001
java -Xmx1024m -Xms1024m -cp target/demo-gpt-java-1.0.jar ai.demo.gpt.Application model=SMALL maxLength=25

