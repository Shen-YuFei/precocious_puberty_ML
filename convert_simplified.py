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


def calculate_lh_fsh_ratio(df: pd.DataFrame) -> pd.Series:
    """计算 LH/FSH 比值"""
    lh_col = "基础血清促黄体生成激素（LH）"
    fsh_col = "基础血清卵泡刺激素（FSH）"

    lh = df[lh_col]
    fsh = df[fsh_col].replace(0, np.nan)  # 避免除零

    return lh / fsh


def main():
    # 路径配置
    input_dir = Path("./input")
    disease_file = input_dir / "激发试验确诊性早熟组数据_new.csv"
    normal_file = input_dir / "性早熟数据激发试验正常组_new.csv"

    # 输出文件
    disease_output = input_dir / "激发试验确诊性早熟组数据_simplified.csv"
    normal_output = input_dir / "性早熟数据激发试验正常组_simplified.csv"

    # 定义精简特征列（根据医生结论，9个特征）
    simplified_feature_cols = [
        "患者编号",
        # === LH/FSH 相关 ===
        "基础血清促黄体生成激素（LH）",  # 1
        "基础血清卵泡刺激素（FSH）",  # 2
        # === 骨龄相关 ===
        "骨龄(岁)",  # 3
        "骨龄与实际年龄比值",  # 4
        # === 子宫 ===
        "子宫长（cm）",  # 6
        "子宫厚（cm）",  # 8
        # === 卵巢体积 ===
        "左卵巢体积（长X宽X厚X0.5233）",  # 13
        "右卵巢体积（长X宽X厚X0.5233）",  # 17
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

    # 提取精简特征
    print("\n提取精简特征...")
    disease_simple = extract_simplified_features(disease_df, simplified_feature_cols)
    normal_simple = extract_simplified_features(normal_df, simplified_feature_cols)

    # 计算 LH/FSH 比值
    print("计算 LH/FSH 比值...")
    disease_simple["LH/FSH比值"] = calculate_lh_fsh_ratio(disease_simple)
    normal_simple["LH/FSH比值"] = calculate_lh_fsh_ratio(normal_simple)

    # 筛选完整样本（所有特征非NaN）
    print("\n筛选完整样本（删除含NaN的行）...")
    disease_before = len(disease_simple)
    normal_before = len(normal_simple)

    disease_simple = disease_simple.dropna()
    normal_simple = normal_simple.dropna()

    print(
        f"  早熟组: {disease_before} -> {len(disease_simple)} (删除 {disease_before - len(disease_simple)} 行)"
    )
    print(
        f"  正常组: {normal_before} -> {len(normal_simple)} (删除 {normal_before - len(normal_simple)} 行)"
    )

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
