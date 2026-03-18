# 智能广告文案生成器 - Streamlit 前端

基于 BLIP + GPT-2 的两阶段模型，实现从产品图片自动生成广告文案。

## 工作流程

```
产品图片 → [BLIP] → 产品描述 → [GPT-2] → 广告文案
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中打开（默认 http://localhost:8501）

## 使用步骤

1. 打开浏览器访问应用
2. 上传时尚产品图片（JPG/PNG）
3. 点击"生成广告文案"按钮
4. 查看生成的产品描述和广告文案

## 项目结构

```
5240DP/
├── app.py                 # Streamlit 前端主程序
├── requirements.txt       # 依赖包
├── README_APP.md         # 使用说明
├── step1_blip_finetuning_(1).ipynb  # BLIP 模型训练
└── step2_gpt2_finetuning.ipynb      # GPT-2 模型训练
```

## 模型说明

- **BLIP**: Salesforce/blip-image-captioning-base，用于图像描述生成
- **GPT-2**: gpt2，用于广告文案生成

## 注意事项

- 首次运行时会自动下载模型（约 1.5GB）
- 建议使用 GPU 以获得更快的推理速度
- 适用于时尚服饰类产品图片
