"""
精简特征 CSV 生成脚本

根据医生结论，从完整数据集中提取以下9个特征：
1. 基础血清促黄体生成激素（LH）
2. 基础血清卵泡刺激素（FSH）
3. 骨龄(岁)
4. 骨龄与实际年龄比值
5. 子宫长（cm）
6. 子宫厚（cm）
7. 左卵巢体积
8. 右卵巢体积
9. LH/FSH比值

使用方法：
    python convert_simplified.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def find_column(df: pd.DataFrame, target: str) -> str | None:
    """查找列名（忽略空格差异）"""
    target_normalized = target.replace(" ", "").replace("\u3000", "")
    for col in df.columns:
        col_normalized = col.replace(" ", "").replace("\u3000", "")
        if target_normalized == col_normalized:
            return col
    return None


def extract_simplified_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """提取精简特征"""
    result = pd.DataFrame()
    missing_cols = []

    for target_col in feature_cols:
        found_col = find_column(df, target_col)
        if found_col:
            result[target_col] = df[found_col]
        else:
            missing_cols.append(target_col)

    if missing_cols:
        print(f"警告: 未找到以下列: {missing_cols}")

    return result


def calculate_lh_fsh_ratio(df: pd.DataFrame, original_df: pd.DataFrame) -> pd.Series:
    """计算 LH/FSH 比值（从原始数据获取FSH）"""
    lh_col = "基础血清促黄体生成激素（LH）"
    fsh_col = "基础血清卵泡刺激素（FSH）"

    if lh_col not in df.columns:
        print(f"  警告: 缺少LH列，跳过LH/FSH比值计算")
        return pd.Series([np.nan] * len(df))

    lh = df[lh_col]

    # 从原始数据获取FSH
    fsh_found = find_column(original_df, fsh_col)
    if fsh_found is None:
        print(f"  警告: 原始数据中缺少FSH列，跳过LH/FSH比值计算")
        return pd.Series([np.nan] * len(df))

    fsh = original_df[fsh_found].iloc[df.index].replace(0, np.nan)
    return lh / fsh


def main():
    # 路径配置
    input_dir = Path("./input")
    disease_file = input_dir / "激发试验确诊性早熟组数据_new.csv"
    normal_file = input_dir / "性早熟数据激发试验正常组_new.csv"

    # 输出文件
    disease_output = input_dir / "激发试验确诊性早熟组数据_simplified.csv"
    normal_output = input_dir / "性早熟数据激发试验正常组_simplified.csv"

    # 定义精简特征列（包含SHAP前5重要特征）
    simplified_feature_cols = [
        "患者编号",
        "年龄",  # 用于年龄过滤
        # === LH/FSH 相关 ===
        "基础血清促黄体生成激素（LH）",  # SHAP #6
        # "基础血清卵泡刺激素（FSH）",
        # === 骨龄相关 ===
        # "骨龄(岁)",
        # "骨龄与实际年龄比值",
        # === 子宫 ===
        "子宫长（cm）",  # SHAP #1
        "子宫厚（cm）",  # SHAP #3
        # === 卵巢 ===
        "最大卵泡直径直径",  # SHAP #2 (新增)
        # "左卵巢体积（长X宽X厚X0.5233）",
        # "右卵巢体积（长X宽X厚X0.5233）",  # SHAP #4
        "卵巢体积平均值",  # SHAP #5 (新增)
    ]

    print("=" * 60)
    print("精简特征 CSV 生成脚本")
    print("=" * 60)

    # 读取数据
    print(f"\n读取早熟组数据: {disease_file}")
    disease_df = pd.read_csv(disease_file)
    print(f"  样本数: {len(disease_df)}, 原始特征数: {len(disease_df.columns)}")

    print(f"\n读取正常组数据: {normal_file}")
    normal_df = pd.read_csv(normal_file)
    print(f"  样本数: {len(normal_df)}, 原始特征数: {len(normal_df.columns)}")

    # 过滤年龄：只保留4-8岁（4 <= 年龄 < 9）
    if "年龄" in disease_df.columns:
        disease_before = len(disease_df)
        disease_df = disease_df[
            (disease_df["年龄"] >= 4) & (disease_df["年龄"] < 9)
        ].reset_index(drop=True)
        print(f"  年龄过滤(4-8岁): {disease_before} -> {len(disease_df)}")

    if "年龄" in normal_df.columns:
        normal_before = len(normal_df)
        normal_df = normal_df[
            (normal_df["年龄"] >= 4) & (normal_df["年龄"] < 9)
        ].reset_index(drop=True)
        print(f"  年龄过滤(4-8岁): {normal_before} -> {len(normal_df)}")

    # 提取精简特征
    print("\n提取精简特征...")
    disease_simple = extract_simplified_features(disease_df, simplified_feature_cols)
    normal_simple = extract_simplified_features(normal_df, simplified_feature_cols)

    # 计算 LH/FSH 比值（如果LH列存在）
    if "基础血清促黄体生成激素（LH）" in disease_simple.columns:
        print("计算 LH/FSH 比值...")
        disease_simple["LH/FSH比值"] = calculate_lh_fsh_ratio(
            disease_simple, disease_df
        )
        normal_simple["LH/FSH比值"] = calculate_lh_fsh_ratio(normal_simple, normal_df)
    else:
        print("跳过 LH/FSH 比值计算（LH列不在特征列表中）")

    # 删除年龄列（仅用于过滤，不作为特征）
    if "年龄" in disease_simple.columns:
        disease_simple = disease_simple.drop(columns=["年龄"])
    if "年龄" in normal_simple.columns:
        normal_simple = normal_simple.drop(columns=["年龄"])

    # 不过滤缺失特征，保留所有样本
    print(f"\n保留所有样本（不过滤缺失特征）")
    print(f"  早熟组: {len(disease_simple)}")
    print(f"  正常组: {len(normal_simple)}")

    # 保存
    print(f"\n保存早熟组精简数据: {disease_output}")
    disease_simple.to_csv(disease_output, index=False)
    print(f"  特征数: {len(disease_simple.columns)}, 样本数: {len(disease_simple)}")

    print(f"\n保存正常组精简数据: {normal_output}")
    normal_simple.to_csv(normal_output, index=False)
    print(f"  特征数: {len(normal_simple.columns)}, 样本数: {len(normal_simple)}")

    # 打印特征列表
    print("\n" + "=" * 60)
    print(
        "精简特征列表 (共 {} 个特征):".format(len(disease_simple.columns) - 1)
    )  # 减去患者编号
    print("=" * 60)
    for i, col in enumerate(disease_simple.columns):
        if col != "患者编号":
            print(f"  {i}. {col}")

    # 数据统计
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    if "LH/FSH比值" in disease_simple.columns:
        print(f"\n早熟组 LH/FSH 比值:")
        print(f"  有效值: {disease_simple['LH/FSH比值'].notna().sum()}")
        print(f"  均值: {disease_simple['LH/FSH比值'].mean():.4f}")
        print(f"  中位数: {disease_simple['LH/FSH比值'].median():.4f}")

        print(f"\n正常组 LH/FSH 比值:")
        print(f"  有效值: {normal_simple['LH/FSH比值'].notna().sum()}")
        print(f"  均值: {normal_simple['LH/FSH比值'].mean():.4f}")
        print(f"  中位数: {normal_simple['LH/FSH比值'].median():.4f}")

    print("\n完成！")


if __name__ == "__main__":
    main()
