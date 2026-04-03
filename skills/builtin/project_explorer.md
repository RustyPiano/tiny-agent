## 项目探索策略
- 接到任务先执行：run_bash("find . -type f | grep -v __pycache__ | grep -v .git | head -60")
- 从入口文件开始读（main.py / index.ts / README.md），再读核心模块
- 读大文件时用 start_line/end_line 分段，每次不超过 150 行
- 在修改前用注释记录你对现有代码的理解
