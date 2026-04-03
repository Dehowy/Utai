"""Model category mapping for separation models.

Categories are determined by a combination of model filename patterns,
primary/secondary stems, and flags from models_info.json.
"""

from __future__ import annotations

from typing import Any

# Category definitions: id -> display info
CATEGORIES = {
    "vocal": {
        "label_zh": "提取人声",
        "label_ja": "ボーカル抽出",
        "label_en": "Extract Vocals",
        "color": "var(--accent-green)",
        "order": 1,
    },
    "instrumental": {
        "label_zh": "提取伴奏",
        "label_ja": "伴奏抽出",
        "label_en": "Extract Instrumental",
        "color": "var(--accent-blue)",
        "order": 2,
    },
    "denoise": {
        "label_zh": "去噪",
        "label_ja": "ノイズ除去",
        "label_en": "Denoise",
        "color": "var(--accent-cyan)",
        "order": 3,
    },
    "dereverb": {
        "label_zh": "去混响",
        "label_ja": "リバーブ除去",
        "label_en": "Dereverb",
        "color": "var(--accent-purple)",
        "order": 4,
    },
    "karaoke": {
        "label_zh": "卡拉OK",
        "label_ja": "カラオケ",
        "label_en": "Karaoke",
        "color": "var(--accent-pink)",
        "order": 5,
    },
    "multi_stem": {
        "label_zh": "多轨道",
        "label_ja": "マルチトラック",
        "label_en": "Multi-stem",
        "color": "var(--accent-orange)",
        "order": 6,
    },
    "other": {
        "label_zh": "其他",
        "label_ja": "その他",
        "label_en": "Other",
        "color": "var(--accent-gold)",
        "order": 7,
    },
}

# Architecture types for sub-grouping within categories
ARCHITECTURES = {
    "VR": {"label": "VR", "order": 1},
    "MDX": {"label": "MDX-Net", "order": 2},
    "Demucs": {"label": "Demucs", "order": 3},
    "MDXC": {"label": "MDXC / Roformer", "order": 4},
}


def get_model_arch(filename: str, info: dict[str, Any]) -> str:
    """Determine the architecture type of a model."""
    model_class = info.get("model_class", "")
    fn_lower = filename.lower()

    if model_class == "VR_Models":
        return "VR"
    if filename.endswith(".onnx"):
        return "MDX"
    if filename.endswith(".th") or filename.endswith(".yaml"):
        return "Demucs"
    # .ckpt models: distinguish MDX23C vs Roformer etc.
    return "MDXC"


def get_model_arch_detail(filename: str, info: dict[str, Any]) -> str:
    """Get a more specific architecture sub-type for display."""
    fn_lower = filename.lower()
    model_class = info.get("model_class", "")

    if model_class == "VR_Models":
        return "VR"
    if filename.endswith(".onnx"):
        return "MDX-Net"
    if filename.endswith(".th") or filename.endswith(".yaml"):
        return "Demucs"

    # MDXC sub-types
    if "bs_roformer" in fn_lower or "bs-roformer" in fn_lower:
        return "BS-Roformer"
    if "mel_band_roformer" in fn_lower or "melband_roformer" in fn_lower or "melbandroformer" in fn_lower:
        return "Mel-Roformer"
    if "mdx23c" in fn_lower:
        return "MDX23C"
    if "swin" in fn_lower:
        return "Swin"
    if "scnet" in fn_lower:
        return "SCNet"
    if "segm" in fn_lower:
        return "SegM"
    if "bandit" in fn_lower:
        return "Bandit"
    if "apollo" in fn_lower:
        return "Apollo"
    if "htdemucs" in fn_lower:
        return "Demucs"

    return "MDXC"


# Brief model descriptions keyed by exact filename
# Format: (zh, ja, en)
MODEL_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    # --- VR v5 ---
    "1_HP-UVR.pth": ("HP系列 伴奏提取 SDR 13.7", "HP系伴奏抽出 SDR 13.7", "HP series, inst SDR 13.7"),
    "2_HP-UVR.pth": ("HP系列变体 伴奏SDR 13.5", "HP系変体 伴奏SDR 13.5", "HP variant, inst SDR 13.5"),
    "3_HP-Vocal-UVR.pth": ("HP系列 人声提取 SDR 8.2", "HP系人声抽出 SDR 8.2", "HP vocal, SDR 8.2"),
    "4_HP-Vocal-UVR.pth": ("HP系列 最佳人声 SDR 8.3", "HP系最良人声 SDR 8.3", "HP best vocal, SDR 8.3"),
    "5_HP-Karaoke-UVR.pth": ("卡拉OK 强力去人声", "カラオケ 強力人声除去", "Karaoke, aggressive vocal removal"),
    "6_HP-Karaoke-UVR.pth": ("卡拉OK改进版 伴奏SDR 13.0", "カラオケ改善版 伴奏SDR 13.0", "Karaoke v2, inst SDR 13.0"),
    "7_HP2-UVR.pth": ("HP2升级版", "HP2アップグレード版", "HP2 upgrade"),
    "8_HP2-UVR.pth": ("HP2变体", "HP2変体", "HP2 variant"),
    "9_HP2-UVR.pth": ("HP2最佳伴奏 SDR 13.7", "HP2最良伴奏 SDR 13.7", "HP2 best inst SDR 13.7"),
    "10_SP-UVR-2B-32000-1.pth": ("SP系列 2B参数 32kHz", "SP系 2Bパラメータ 32kHz", "SP 2B params 32kHz"),
    "11_SP-UVR-2B-32000-2.pth": ("SP系列 最佳伴奏 SDR 13.8", "SP系最良伴奏 SDR 13.8", "SP best inst SDR 13.8"),
    "12_SP-UVR-3B-44100.pth": ("SP系列 3B参数 44.1kHz", "SP系 3Bパラメータ 44.1kHz", "SP 3B params 44.1kHz"),
    "13_SP-UVR-4B-44100-1.pth": ("SP系列 4B参数", "SP系 4Bパラメータ", "SP 4B params"),
    "14_SP-UVR-4B-44100-2.pth": ("SP系列 4B改进版", "SP系 4B改善版", "SP 4B improved"),
    "15_SP-UVR-MID-44100-1.pth": ("SP MID系列", "SP MID系", "SP MID series"),
    "16_SP-UVR-MID-44100-2.pth": ("SP MID改进版", "SP MID改善版", "SP MID improved"),
    "17_HP-Wind_Inst-UVR.pth": ("分离木管乐器", "木管楽器の分離", "Woodwind separation"),
    # --- VR audio processing ---
    "UVR-De-Echo-Aggressive.pth": ("激进去回声", "アグレッシブエコー除去", "Aggressive de-echo"),
    "UVR-De-Echo-Normal.pth": ("标准去回声", "標準エコー除去", "Normal de-echo"),
    "UVR-DeEcho-DeReverb.pth": ("回声+混响同时去除", "エコー＋リバーブ同時除去", "Combined de-echo & de-reverb"),
    "UVR-DeNoise-Lite.pth": ("轻量去噪", "軽量ノイズ除去", "Lightweight denoise"),
    "UVR-DeNoise.pth": ("标准去噪", "標準ノイズ除去", "Standard denoise"),
    "UVR-DeReverb-aufr33-jarredou_4band_v4_ms_fullband.pth": ("去混响 4band全频段", "リバーブ除去 4band全帯域", "Dereverb 4band fullband"),
    # --- VR v4 ---
    "MGM_HIGHEND_v4.pth": ("高频处理 SDR 12.3", "高周波処理 SDR 12.3", "High-end processing SDR 12.3"),
    "MGM_LOWEND_A_v4.pth": ("低频处理A SDR 13.0", "低周波処理A SDR 13.0", "Low-end A, SDR 13.0"),
    "MGM_LOWEND_B_v4.pth": ("低频处理B SDR 13.1", "低周波処理B SDR 13.1", "Low-end B, SDR 13.1"),
    "MGM_MAIN_v4.pth": ("MGM主模型 SDR 12.4", "MGMメインモデル SDR 12.4", "MGM main, SDR 12.4"),
    # --- VR other ---
    "UVR-BVE-4B_SN-44100-1.pth": ("BVE系列 4B参数", "BVE系 4Bパラメータ", "BVE 4B params"),
    "Harmonic_Noise_Separation_yxlllc.pth": ("谐波/噪声分离", "ハーモニック/ノイズ分離", "Harmonic/noise separation"),
    # --- Demucs ---
    "HTDemucs4.th": ("Demucs标准版 4轨道", "Demucs標準版 4トラック", "Demucs standard, 4 stems"),
    "HTDemucs4_6stems.th": ("6轨道: 人声/鼓/贝斯/吉他/钢琴/其他", "6トラック: ボーカル/ドラム/ベース/ギター/ピアノ/他", "6 stems: vocal/drums/bass/guitar/piano/other"),
    "HTDemucs4_FT_bass.th": ("微调版 最佳贝斯SDR 12.0", "FT版 最良ベースSDR 12.0", "Fine-tuned, best bass SDR 12.0"),
    "HTDemucs4_FT_drums.th": ("微调版 鼓分离", "FT版 ドラム分離", "Fine-tuned drums"),
    "HTDemucs4_FT_other.th": ("微调版 其他分离", "FT版 その他分離", "Fine-tuned other"),
    "HTDemucs4_FT_vocals_official.th": ("微调版 人声SDR 10.8", "FT版 人声SDR 10.8", "Fine-tuned vocal SDR 10.8"),
    # --- MDXC: BS-Roformer ---
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt": ("★最佳人声 SDR 12.97", "★最良人声 SDR 12.97", "★ Best vocal SDR 12.97"),
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt": ("人声提取 SDR 12.96", "人声抽出 SDR 12.96", "Vocal SDR 12.96"),
    "model_bs_roformer_ep_937_sdr_10.5309.ckpt": ("去鼓去贝斯", "ドラム＋ベース除去", "Remove drums & bass"),
    "BS-Roformer_LargeV1.ckpt": ("大参数版 人声提取", "大パラメータ版 人声抽出", "Large model, vocal extraction"),
    "bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt": ("男声/女声分离 SDR 7.29", "男声/女声分離 SDR 7.29", "Male/female separation SDR 7.29"),
    "bs_roformer_4stems_ft.ckpt": ("4轨道微调版", "4トラック FT版", "4-stem fine-tuned"),
    # --- MDXC: Mel-Roformer ---
    "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt": ("Mel-Roformer基础版 SDR 11.44", "Mel-Roformer基本版 SDR 11.44", "Mel-Roformer base SDR 11.44"),
    "Kim_MelBandRoformer.ckpt": ("Kim训练 人声提取", "Kim訓練 人声抽出", "Kim-trained vocal"),
    "kimmel_unwa_ft.ckpt": ("Kim微调版", "Kim FT版", "Kim fine-tuned"),
    "melband_roformer_inst_v2.ckpt": ("★最佳伴奏 SDR 16.1", "★最良伴奏 SDR 16.1", "★ Best instrumental SDR 16.1"),
    "melband_roformer_instvox_duality_v2.ckpt": ("人声+伴奏双输出 SDR 16.1/11.0", "人声＋伴奏デュアル出力", "Dual vocal+inst output"),
    "inst_v1e.ckpt": ("伴奏V1E SDR 15.8", "伴奏V1E SDR 15.8", "Inst V1E SDR 15.8"),
    "big_beta5e.ckpt": ("大参数Beta5e 人声SDR 12.4", "大パラメータBeta5e 人声SDR 12.4", "Big Beta5e vocal SDR 12.4"),
    "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt": ("去除人群噪音 SDR 8.71", "群衆ノイズ除去 SDR 8.71", "Crowd noise removal SDR 8.71"),
    "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt": ("卡拉OK专用 SDR 10.20", "カラオケ専用 SDR 10.20", "Karaoke SDR 10.20"),
    "mel_band_roformer_vocals_becruily.ckpt": ("Becruily人声版", "Becruily人声版", "Becruily vocal"),
    "mel_band_roformer_instrumental_becruily.ckpt": ("Becruily伴奏版", "Becruily伴奏版", "Becruily instrumental"),
    # --- MDXC: MDX23C ---
    "model_mdx23c_ep_168_sdr_7.0207.ckpt": ("MDX23C ep168 SDR 7.02", "MDX23C ep168 SDR 7.02", "MDX23C ep168 SDR 7.02"),
    "model_mdx23c_ep_271_l1_freq_72.2383.ckpt": ("MDX23C L1频率优化", "MDX23C L1周波数最適化", "MDX23C L1 freq optimized"),
    "model_vocals_mdx23c_sdr_10.17.ckpt": ("MDX23C人声 SDR 10.17", "MDX23C人声 SDR 10.17", "MDX23C vocal SDR 10.17"),
    # --- MDXC: Dereverb ---
    "dereverb_mdx23c_sdr_6.9096.ckpt": ("MDX23C去混响 SDR 6.91", "MDX23Cリバーブ除去 SDR 6.91", "MDX23C dereverb SDR 6.91"),
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt": ("★最佳去混响 SDR 19.17", "★最良リバーブ除去 SDR 19.17", "★ Best dereverb SDR 19.17"),
    "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt": ("温和去混响 SDR 18.81", "穏やかリバーブ除去 SDR 18.81", "Less aggressive dereverb SDR 18.81"),
    "deverb_bs_roformer_8_256dim_8depth.ckpt": ("BS-Roformer去混响 8depth", "BS-Roformerリバーブ除去 8depth", "BS-Roformer dereverb 8depth"),
    "deverb_bs_roformer_8_384dim_10depth.ckpt": ("BS-Roformer去混响 10depth", "BS-Roformerリバーブ除去 10depth", "BS-Roformer dereverb 10depth"),
    "deverb_mel_band_roformer_ep_27_sdr_10.4567.ckpt": ("Mel-Roformer去混响 SDR 10.46", "Mel-Roformerリバーブ除去 SDR 10.46", "Mel-Roformer dereverb SDR 10.46"),
    "de_big_reverb_mbr_ep_362.ckpt": ("大参数去混响", "大パラメータリバーブ除去", "Big model dereverb"),
    "dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt": ("混合去混响+回声", "ハイブリッドリバーブ＋エコー除去", "Fused dereverb+echo removal"),
    # --- MDXC: Denoise ---
    "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt": ("★最佳去噪 SDR 27.99", "★最良ノイズ除去 SDR 27.99", "★ Best denoise SDR 27.99"),
    "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt": ("激进去噪 SDR 27.98", "アグレッシブノイズ除去 SDR 27.98", "Aggressive denoise SDR 27.98"),
    # --- MDXC: Other ---
    "aspiration_mel_band_roformer_sdr_18.9845.ckpt": ("呼吸声分离 SDR 18.98", "ブレス分離 SDR 18.98", "Breath separation SDR 18.98"),
    "aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt": ("温和呼吸声分离 SDR 18.12", "穏やかブレス分離 SDR 18.12", "Less aggressive breath sep SDR 18.12"),
    "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt": ("鼓分离 6种鼓类型 SDR 10.81", "ドラム分離 6種 SDR 10.81", "Drum sep, 6 drum types SDR 10.81"),
    "model_drumsep.th": ("Demucs鼓分离", "Demucsドラム分離", "Demucs drum separation"),
    "Apollo_LQ_MP3_restoration.ckpt": ("MP3低质量音频修复", "MP3低品質修復", "Low-quality MP3 restoration"),
    "apollo_model_uni.ckpt": ("通用音频修复", "汎用オーディオ修復", "Universal audio restoration"),
    # --- MDXC: Swin / SCNet / Others ---
    "model_swin_upernet_ep_56_sdr_10.6703.ckpt": ("Swin人声 SDR 10.67", "Swin人声 SDR 10.67", "Swin vocal SDR 10.67"),
    "model_scnet_sdr_9.3244.ckpt": ("SCNet多轨道 SDR 9.32", "SCNetマルチトラック SDR 9.32", "SCNet multi-stem SDR 9.32"),
    "scnet_checkpoint_musdb18.ckpt": ("SCNet MUSDB18基准", "SCNet MUSDB18ベンチマーク", "SCNet MUSDB18 benchmark"),
    "model_bandit_plus_dnr_sdr_11.47.chpt": ("Bandit+ 去噪+去混响 SDR 11.47", "Bandit+ ノイズ＋リバーブ除去", "Bandit+ denoise+dereverb SDR 11.47"),
    "model_vocals_htdemucs_sdr_8.78.ckpt": ("HTDemucs人声 SDR 8.78", "HTDemucs人声 SDR 8.78", "HTDemucs vocal SDR 8.78"),
    "model_vocals_mel_band_roformer_sdr_8.42.ckpt": ("Mel-Roformer人声 SDR 8.42", "Mel-Roformer人声 SDR 8.42", "Mel-Roformer vocal SDR 8.42"),
    "model_vocals_segm_models_sdr_9.77.ckpt": ("SegM人声 SDR 9.77", "SegM人声 SDR 9.77", "SegM vocal SDR 9.77"),
    # --- New MDX23C ---
    "MDX23C-8KFFT-InstVoc_HQ.ckpt": ("MDX23C HQ 人声+伴奏 SDR 15.8", "MDX23C HQ ボーカル＋伴奏 SDR 15.8", "MDX23C HQ vocal+inst SDR 15.8"),
    "MDX23C-8KFFT-InstVoc_HQ_2.ckpt": ("★MDX23C HQ2 最佳平衡 SDR 15.9", "★MDX23C HQ2 最良バランス SDR 15.9", "★ MDX23C HQ2 best balance SDR 15.9"),
    "MDX23C_D1581.ckpt": ("MDX23C D1581 SDR 15.5", "MDX23C D1581 SDR 15.5", "MDX23C D1581 SDR 15.5"),
    # --- New Mel-Roformer inst ---
    "melband_roformer_inst_v1.ckpt": ("伴奏V1 SDR 15.9", "伴奏V1 SDR 15.9", "Inst V1 SDR 15.9"),
    "melband_roformer_instvoc_duality_v1.ckpt": ("人声+伴奏双输出V1", "人声＋伴奏デュアルV1", "Dual vocal+inst V1"),
    # --- Big/SYHFT ---
    "MelBandRoformerBigSYHFTV1.ckpt": ("大参数SYHFT V1 SDR 12.3", "大パラメータSYHFT V1 SDR 12.3", "Big SYHFT V1 SDR 12.3"),
    "melband_roformer_big_beta4.ckpt": ("大参数Beta4 SDR 12.5", "大パラメータBeta4 SDR 12.5", "Big Beta4 SDR 12.5"),
    "MelBandRoformerSYHFT.ckpt": ("SYHFT基础版 SDR 8.0", "SYHFT基本版 SDR 8.0", "SYHFT base SDR 8.0"),
    "MelBandRoformerSYHFTV2.ckpt": ("SYHFT V2 SDR 8.6", "SYHFT V2 SDR 8.6", "SYHFT V2 SDR 8.6"),
    "MelBandRoformerSYHFTV2.5.ckpt": ("SYHFT V2.5 SDR 8.5", "SYHFT V2.5 SDR 8.5", "SYHFT V2.5 SDR 8.5"),
    "MelBandRoformerSYHFTV3Epsilon.ckpt": ("SYHFT V3 SDR 9.5", "SYHFT V3 SDR 9.5", "SYHFT V3 SDR 9.5"),
    # --- Dereverb+echo ---
    "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt": ("去混响+回声 SDR 10.02", "リバーブ＋エコー除去 SDR 10.02", "Dereverb+echo SDR 10.02"),
    "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt": ("去混响+回声V2 SDR 13.48", "リバーブ＋エコー除去V2 SDR 13.48", "Dereverb+echo V2 SDR 13.48"),
    # --- Bleed suppressor ---
    "mel_band_roformer_bleed_suppressor_v1.ckpt": ("抑制人声泄漏", "ボーカル漏れ抑制", "Vocal bleed suppression"),
    # --- Gabox vocal ---
    "mel_band_roformer_vocals_gabox.ckpt": ("Gabox人声V1", "Gaboxボーカル V1", "Gabox vocal V1"),
    "mel_band_roformer_vocals_v2_gabox.ckpt": ("Gabox人声V2", "GaboxボーカルV2", "Gabox vocal V2"),
    "mel_band_roformer_vocals_fv1_gabox.ckpt": ("Gabox人声FV1", "GaboxボーカルFV1", "Gabox vocal FV1"),
    "mel_band_roformer_vocals_fv2_gabox.ckpt": ("Gabox人声FV2", "GaboxボーカルFV2", "Gabox vocal FV2"),
    "mel_band_roformer_vocals_fv3_gabox.ckpt": ("Gabox人声FV3", "GaboxボーカルFV3", "Gabox vocal FV3"),
    "mel_band_roformer_vocals_fv4_gabox.ckpt": ("Gabox人声FV4", "GaboxボーカルFV4", "Gabox vocal FV4"),
    "mel_band_roformer_vocals_fv5_gabox.ckpt": ("Gabox人声FV5", "GaboxボーカルFV5", "Gabox vocal FV5"),
    "mel_band_roformer_vocals_fv6_gabox.ckpt": ("Gabox人声FV6", "GaboxボーカルFV6", "Gabox vocal FV6"),
    # --- Gabox instrumental ---
    "mel_band_roformer_instrumental_gabox.ckpt": ("Gabox伴奏V1", "Gabox伴奏V1", "Gabox inst V1"),
    "mel_band_roformer_instrumental_2_gabox.ckpt": ("Gabox伴奏V2", "Gabox伴奏V2", "Gabox inst V2"),
    "mel_band_roformer_instrumental_3_gabox.ckpt": ("Gabox伴奏V3", "Gabox伴奏V3", "Gabox inst V3"),
    "mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt": ("Gabox无泄漏伴奏V1", "Gaboxリークレス伴奏V1", "Gabox bleedless inst V1"),
    "mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt": ("Gabox无泄漏伴奏V2", "Gaboxリークレス伴奏V2", "Gabox bleedless inst V2"),
    "mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt": ("Gabox无泄漏伴奏V3", "Gaboxリークレス伴奏V3", "Gabox bleedless inst V3"),
    "mel_band_roformer_instrumental_fullness_v1_gabox.ckpt": ("Gabox饱满伴奏V1", "Gaboxフルネス伴奏V1", "Gabox fullness inst V1"),
    "mel_band_roformer_instrumental_fullness_v2_gabox.ckpt": ("Gabox饱满伴奏V2", "Gaboxフルネス伴奏V2", "Gabox fullness inst V2"),
    "mel_band_roformer_instrumental_fullness_v3_gabox.ckpt": ("Gabox饱满伴奏V3", "Gaboxフルネス伴奏V3", "Gabox fullness inst V3"),
    "mel_band_roformer_instrumental_instv5_gabox.ckpt": ("Gabox伴奏INSTV5", "Gabox伴奏INSTV5", "Gabox inst V5"),
    "mel_band_roformer_instrumental_instv6_gabox.ckpt": ("Gabox伴奏INSTV6", "Gabox伴奏INSTV6", "Gabox inst V6"),
    "mel_band_roformer_instrumental_instv7_gabox.ckpt": ("Gabox伴奏INSTV7", "Gabox伴奏INSTV7", "Gabox inst V7"),
    "mel_band_roformer_instrumental_instv8_gabox.ckpt": ("Gabox伴奏INSTV8", "Gabox伴奏INSTV8", "Gabox inst V8"),
}


def get_model_description(filename: str, lang: str = "zh") -> str:
    """Get a brief description for a model. Returns empty string if not found."""
    desc = MODEL_DESCRIPTIONS.get(filename)
    if not desc:
        return ""
    idx = {"zh": 0, "ja": 1, "en": 2}.get(lang, 0)
    return desc[idx]


def categorize_model(filename: str, info: dict[str, Any]) -> str:
    """Determine the category of a model based on its filename and metadata."""
    fn_lower = filename.lower()
    model_class = info.get("model_class", "")
    primary_stem = info.get("primary_stem", "")
    secondary_stem = info.get("secondary_stem", "")

    # --- Karaoke (check first, before vocal/instrumental) ---
    if info.get("is_karaoke"):
        return "karaoke"
    if "karaoke" in fn_lower or "kara" in fn_lower:
        return "karaoke"

    # --- Denoise ---
    if "denoise" in fn_lower or "noise" in fn_lower:
        return "denoise"
    if primary_stem == "Noise" or secondary_stem == "Noise":
        return "denoise"
    if primary_stem == "No Noise" or secondary_stem == "No Noise":
        return "denoise"

    # --- Dereverb / De-echo ---
    if any(kw in fn_lower for kw in ("dereverb", "deverb", "de_reverb", "de-reverb",
                                      "de_echo", "de-echo", "deecho", "reverb")):
        return "dereverb"
    if primary_stem in ("Dry", "No Reverb", "No Echo") or secondary_stem in ("Reverb", "Echo"):
        return "dereverb"

    # --- Multi-stem ---
    if model_class == "multi_stem_models":
        # Drum separation models go to "other"
        if "drumsep" in fn_lower or "drum_sep" in fn_lower:
            return "other"
        return "multi_stem"

    # --- Aspiration / breath ---
    if "aspiration" in fn_lower:
        return "other"
    if primary_stem == "No Aspiration" or secondary_stem == "Aspiration":
        return "other"

    # --- Crowd noise ---
    if "crowd" in fn_lower:
        return "other"

    # --- Male/female separation ---
    if "male_female" in fn_lower or "chorus" in fn_lower:
        return "other"

    # --- Woodwinds ---
    if "wind_inst" in fn_lower or "woodwind" in fn_lower:
        return "other"
    if primary_stem == "No Woodwinds" or secondary_stem == "Woodwinds":
        return "other"

    # --- Restoration / Apollo ---
    if "restoration" in fn_lower or "apollo" in fn_lower:
        return "other"

    # --- Bleed suppressor ---
    if "bleed" in fn_lower:
        return "other"

    # --- Single-stem vocal/instrumental extraction from demucs ---
    if model_class == "single_stem_models":
        if "vocal" in fn_lower:
            return "vocal"
        if "drum" in fn_lower or "bass" in fn_lower or "other" in fn_lower:
            return "multi_stem"
        # Default single_stem to vocal category
        return "vocal"

    # --- Instrumental-focused models ---
    if "inst" in fn_lower and "instvoc" not in fn_lower and "instvox" not in fn_lower:
        return "instrumental"

    # --- VR models: check stems ---
    if model_class == "VR_Models":
        if primary_stem == "Vocals":
            return "vocal"
        if primary_stem == "Instrumental":
            return "instrumental"

    # --- Vocal models (model_class) ---
    if model_class == "vocal_models":
        # Some vocal_models are actually instrumental-focused
        if "inst" in fn_lower and "instvoc" not in fn_lower and "instvox" not in fn_lower:
            return "instrumental"
        return "vocal"

    # Fallback
    return "vocal"
