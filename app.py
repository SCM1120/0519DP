import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

# 页面配置
st.set_page_config(
    page_title="智能广告文案生成器",
    page_icon="📝",
    layout="centered",
)

# CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .caption-text {
        font-size: 1.1rem;
        color: #333;
        font-style: italic;
    }
    .ad-text {
        font-size: 1.2rem;
        color: #1f77b4;
        font-weight: 500;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """加载BLIP和GPT-2模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载BLIP模型
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_model.eval()

    # 加载GPT-2模型
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_model.eval()

    return blip_processor, blip_model, gpt2_tokenizer, gpt2_model, device


def generate_caption(image, blip_processor, blip_model, device):
    """使用BLIP生成图片标题"""
    inputs = blip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = blip_model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True
        )

    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption


def generate_ad(caption, gpt2_tokenizer, gpt2_model, device):
    """使用GPT-2生成广告文案"""
    # 构建提示
    prompt = f"Product: {caption}\nDescription: {caption}\nAd:"

    inputs = gpt2_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = gpt2_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            eos_token_id=gpt2_tokenizer.eos_token_id,
        )

    full_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 提取广告部分
    if "Ad:" in full_text:
        ad = full_text.split("Ad:")[-1].strip().split("\n")[0]
    else:
        ad = full_text.strip()

    return ad


def main():
    # 标题
    st.markdown('<div class="main-header">📝 智能广告文案生成器</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">上传产品图片，AI自动生成吸引人的广告文案</div>', unsafe_allow_html=True)

    # 加载模型
    with st.spinner("正在加载模型，请稍候..."):
        blip_processor, blip_model, gpt2_tokenizer, gpt2_model, device = load_models()
    st.success("✅ 模型加载完成！")

    # 图片上传
    uploaded_file = st.file_uploader(
        "上传产品图片",
        type=["jpg", "jpeg", "png"],
        help="支持 JPG、JPEG、PNG 格式"
    )

    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="上传的产品图片", use_container_width=True)

        # 生成按钮
        if st.button("🚀 生成广告文案", type="primary", use_container_width=True):
            # 步骤1: 生成标题
            with st.spinner("🔍 正在分析图片内容..."):
                caption = generate_caption(image, blip_processor, blip_model, device)

            # 显示标题
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown("**📝 产品描述：**")
            st.markdown(f"<div class='caption-text'>\"{caption}\"</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # 步骤2: 生成广告文案
            with st.spinner("✨ 正在创作广告文案..."):
                ad = generate_ad(caption, gpt2_tokenizer, gpt2_model, device)

            # 显示广告文案
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown("**📢 生成的广告文案：**")
            st.markdown(f"<div class='ad-text'>\"{ad}\"</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # 复制按钮
            st.code(ad, language="text")

    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        **工作流程：**
        1. 📤 上传时尚产品图片（服装、配饰等）
        2. 🔍 BLIP模型分析图片并生成产品描述
        3. ✨ GPT-2模型基于描述生成吸引人的广告文案

        **提示：**
        - 建议使用清晰的产品图片
        - 图片中的产品应占据主要画面
        - 支持时尚服饰类产品效果最佳
        """)

    # 页脚
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #999;'>Powered by BLIP + GPT-2</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
