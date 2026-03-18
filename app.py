import streamlit as st

# 页面配置必须在最前面
st.set_page_config(
    page_title="智能广告文案生成器",
    page_icon="📝",
    layout="centered",
)

# 检查依赖是否安装
try:
    from PIL import Image
    import torch
    from transformers import (
        BlipProcessor,
        BlipForConditionalGeneration,
        GPT2Tokenizer,
        GPT2LMHeadModel,
    )
except ImportError as e:
    st.error(f"❌ 缺少依赖包: {e}")
    st.info("请确保 requirements.txt 已正确配置并重新部署")
    st.stop()

import gc

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


@st.cache_resource(show_spinner=False)
def load_blip_model():
    """加载BLIP模型"""
    device = "cpu"  # Streamlit Cloud 使用 CPU
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.eval()
        return processor, model, device
    except Exception as e:
        st.error(f"加载 BLIP 模型失败: {e}")
        raise


@st.cache_resource(show_spinner=False)
def load_gpt2_model():
    """加载GPT-2模型"""
    device = "cpu"  # Streamlit Cloud 使用 CPU
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"加载 GPT-2 模型失败: {e}")
        raise


def generate_caption(image, blip_processor, blip_model, device):
    """使用BLIP生成图片标题"""
    inputs = blip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output_ids = blip_model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=2,
            early_stopping=True
        )

    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption


def generate_ad(caption, gpt2_tokenizer, gpt2_model, device):
    """使用GPT-2生成广告文案"""
    prompt = f"Product: {caption}\nDescription: {caption}\nAd:"
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = gpt2_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )

    full_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "Ad:" in full_text:
        ad = full_text.split("Ad:")[-1].strip().split("\n")[0]
    else:
        ad = full_text.strip()

    return ad


def main():
    # 标题
    st.markdown('<div class="main-header">📝 智能广告文案生成器</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">上传产品图片，AI自动生成吸引人的广告文案</div>', unsafe_allow_html=True)

    # 初始化 session state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

    # 加载模型
    if not st.session_state.models_loaded:
        with st.spinner("⏳ 正在加载 AI 模型（首次需要下载，约1-2分钟）..."):
            try:
                blip_processor, blip_model, device1 = load_blip_model()
                gpt2_tokenizer, gpt2_model, device2 = load_gpt2_model()
                st.session_state.blip_processor = blip_processor
                st.session_state.blip_model = blip_model
                st.session_state.gpt2_tokenizer = gpt2_tokenizer
                st.session_state.gpt2_model = gpt2_model
                st.session_state.device = device1
                st.session_state.models_loaded = True
                st.success("✅ 模型加载完成！")
            except Exception as e:
                st.error(f"模型加载失败: {str(e)}")
                st.stop()
    else:
        blip_processor = st.session_state.blip_processor
        blip_model = st.session_state.blip_model
        gpt2_tokenizer = st.session_state.gpt2_tokenizer
        gpt2_model = st.session_state.gpt2_model
        device = st.session_state.device

    # 图片上传
    uploaded_file = st.file_uploader(
        "📤 上传产品图片",
        type=["jpg", "jpeg", "png"],
        help="支持 JPG、JPEG、PNG 格式，建议图片大小不超过 2MB"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="上传的产品图片", use_container_width=True)

            if st.button("🚀 生成广告文案", type="primary", use_container_width=True):
                # 步骤1: 生成标题
                with st.spinner("🔍 正在分析图片内容..."):
                    caption = generate_caption(image, blip_processor, blip_model, device)

                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown("**📝 产品描述：**")
                st.markdown(f"<div class='caption-text'>\"{caption}\"</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # 步骤2: 生成广告文案
                with st.spinner("✨ 正在创作广告文案..."):
                    ad = generate_ad(caption, gpt2_tokenizer, gpt2_model, device)

                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown("**📢 生成的广告文案：**")
                st.markdown(f"<div class='ad-text'>\"{ad}\"</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # 复制按钮
                st.code(ad, language="text")

                # 清理内存
                gc.collect()

        except Exception as e:
            st.error(f"处理图片时出错: {str(e)}")

    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        **工作流程：**
        1. 📤 上传时尚产品图片（服装、配饰等）
        2. 🔍 BLIP模型分析图片并生成产品描述
        3. ✨ GPT-2模型基于描述生成吸引人的广告文案

        **提示：**
        - 建议使用清晰的产品图片
        - 图片大小建议不超过 2MB
        - 适用于时尚服饰类产品
        """)

    # 页脚
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #999;'>Powered by BLIP + GPT-2</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
