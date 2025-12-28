#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import re

# ===== 关键列定义（统一使用不带后缀的列名）=====
CRITICAL_COLUMNS = {
    "基本临床资料": [
        "患者编号",
        "年龄",
        "身高（cm）",
        "体重（kg）",
        "BMI （体重Kg÷身高m）",
        "Tanner分期",
        "乳晕色素沉着",
        "乳核",
        "有无阴毛",
        "有无腋毛",
    ],
    "实验室检验": [
        "基础血清促黄体生成激素（LH）",
        "基础血清卵泡刺激素（FSH）",
        "LH/FSH比值",
    ],
    "超声报告": [
        "子宫长（cm）",
        "子宫宽（cm）",
        "子宫厚（cm）",
        "子宫体积（长X宽X厚X0.5236）",
        "最大卵泡直径直径",
        "左卵巢长（cm）",
        "左卵巢宽（cm）",
        "左卵巢厚（cm）",
        "左卵巢体积（长X宽X厚X0.5233）",
        "右卵巢长（cm）",
        "右卵巢宽（cm）",
        "右卵巢厚（cm）",
        "右卵巢体积（长X宽X厚X0.5233）",
        "卵巢长平均值（cm）",
        "卵巢宽平均值（cm）",
        "卵巢厚平均值（cm）",
        "卵巢体积平均值",
        "左乳腺体宽（cm）",
        "左乳腺体厚（cm）",
        "右乳腺体宽（cm）",
        "右乳腺体厚（cm）",
        "乳腺体宽平均值（cm）",
        "乳腺体厚平均值（cm）",
    ],
    "左手X线检查": [
        "骨龄(岁)",
        "骨龄与实际年龄比值",
        "生物年龄和骨龄之间的差异",
        "按CHN法测算，左手、腕骨发育成熟度评分",
    ],
}

# ===== 计算列定义=====
COMPUTED_COLUMNS = {
    "BMI （体重Kg÷身高m）": {
        "formula": lambda df: df["体重（kg）"] / (df["身高（cm）"] / 100) ** 2,
        "depends": ["身高（cm）", "体重（kg）"],
    },
    "子宫体积（长X宽X厚X0.5236）": {
        "formula": lambda df: df["子宫长（cm）"]
        * df["子宫宽（cm）"]
        * df["子宫厚（cm）"]
        * 0.5236,
        "depends": ["子宫长（cm）", "子宫宽（cm）", "子宫厚（cm）"],
    },
    "骨龄与实际年龄比值": {
        "formula": lambda df: (df["骨龄(岁)"] / df["年龄"]).round(2),
        "depends": ["骨龄(岁)", "年龄"],
    },
    "生物年龄和骨龄之间的差异": {
        "formula": lambda df: df["骨龄(岁)"] - df["年龄"],
        "depends": ["年龄", "骨龄(岁)"],
    },
    "卵巢长平均值（cm）": {
        "formula": lambda df: smart_avg(df["左卵巢长（cm）"], df["右卵巢长（cm）"]),
        "depends": ["左卵巢长（cm）", "右卵巢长（cm）"],
    },
    "卵巢宽平均值（cm）": {
        "formula": lambda df: smart_avg(df["左卵巢宽（cm）"], df["右卵巢宽（cm）"]),
        "depends": ["左卵巢宽（cm）", "右卵巢宽（cm）"],
    },
    "卵巢厚平均值（cm）": {
        "formula": lambda df: smart_avg(df["左卵巢厚（cm）"], df["右卵巢厚（cm）"]),
        "depends": ["左卵巢厚（cm）", "右卵巢厚（cm）"],
    },
    "卵巢体积平均值": {
        "formula": lambda df: smart_avg(
            df["左卵巢体积（长X宽X厚X0.5233）"], df["右卵巢体积（长X宽X厚X0.5233）"]
        ),
        "depends": ["左卵巢体积（长X宽X厚X0.5233）", "右卵巢体积（长X宽X厚X0.5233）"],
    },
    "乳腺体宽平均值（cm）": {
        "formula": lambda df: smart_avg(df["左乳腺体宽（cm）"], df["右乳腺体宽（cm）"]),
        "depends": ["左乳腺体宽（cm）", "右乳腺体宽（cm）"],
    },
    "乳腺体厚平均值（cm）": {
        "formula": lambda df: smart_avg(df["左乳腺体厚（cm）"], df["右乳腺体厚（cm）"]),
        "depends": ["左乳腺体厚（cm）", "右乳腺体厚（cm）"],
    },
    "LH/FSH比值": {
        "formula": lambda df: (
            df["基础血清促黄体生成激素（LH）"] / df["基础血清卵泡刺激素（FSH）"]
        ).round(4),
        "depends": ["基础血清促黄体生成激素（LH）", "基础血清卵泡刺激素（FSH）"],
    },
}


def smart_avg(left, right):
    """智能平均值：双侧有值取平均，单侧有值取该侧值"""
    return np.where(
        left.notna() & right.notna(),
        (left + right) / 2,
        np.where(left.notna(), left, np.where(right.notna(), right, np.nan)),
    )


# 0值应视为缺失的列
ZERO_AS_NA_COLS = [
    "左卵巢体积（长X宽X厚X0.5233）",
    "右卵巢体积（长X宽X厚X0.5233）",
    "子宫厚（cm）",
]

# 需要计算多次测量平均值的列（早熟组有多次测量）
MULTI_MEASUREMENT_COLS = [
    "子宫长（cm）",
    "子宫宽（cm）",
    "子宫厚（cm）",
    "最大卵泡直径直径",
    "左卵巢长（cm）",
    "左卵巢宽（cm）",
    "左卵巢厚（cm）",
    "左卵巢体积（长X宽X厚X0.5233）",
    "右卵巢长（cm）",
    "右卵巢宽（cm）",
    "右卵巢厚（cm）",
    "右卵巢体积（长X宽X厚X0.5233）",
    "左乳腺体宽（cm）",
    "左乳腺体厚（cm）",
    "右乳腺体宽（cm）",
    "右乳腺体厚（cm）",
]


def get_multi_measurement_avg(df, base_col):
    """计算多次测量的平均值（skipna=True，0值视为缺失）"""
    related_cols = [
        c for c in df.columns if re.match(rf"^{re.escape(base_col)}[123]$", c)
    ]
    if not related_cols:
        return None

    zero_as_na = base_col in ZERO_AS_NA_COLS
    numeric_data = []
    for col in related_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if zero_as_na:
            series = series.replace(0, np.nan)
        numeric_data.append(series)

    return pd.concat(numeric_data, axis=1).mean(axis=1, skipna=True)


def get_all_columns():
    """获取所有关键列"""
    return [col for cols in CRITICAL_COLUMNS.values() for col in cols]


def convert_xlsx_to_csv(xlsx_file, csv_file, skip_rows=1, is_disease=True):
    """转换XLSX到CSV"""
    try:
        print(f"\n读取: {xlsx_file}")
        df = pd.read_excel(xlsx_file, skiprows=skip_rows)
        print(f"  原始: {df.shape[0]} 行 × {df.shape[1]} 列")

        all_cols = get_all_columns()
        computed_col_names = set(COMPUTED_COLUMNS.keys())

        # 早熟组：计算多次测量平均值，存入不带后缀的列名
        if is_disease:
            for base_col in MULTI_MEASUREMENT_COLS:
                avg_result = get_multi_measurement_avg(df, base_col)
                if avg_result is not None:
                    df[base_col] = avg_result  # 直接用不带后缀的列名
                    print(f"  多测量均值: {base_col}")

            # 早熟组特殊列映射（带后缀1的列 -> 不带后缀）
            suffix_mapping = {
                "Tanner分期1": "Tanner分期",
                "乳晕色素沉着1": "乳晕色素沉着",
                "乳核1": "乳核",
                "有无阴毛1": "有无阴毛",
                "有无腋毛1": "有无腋毛",
                "基础血清促黄体生成激素（LH）1": "基础血清促黄体生成激素（LH）",
                "基础血清卵泡刺激素（FSH）1": "基础血清卵泡刺激素（FSH）",
                "骨龄(岁)1": "骨龄(岁)",
                "按CHN法测算，左手、腕骨发育成熟度评分1": "按CHN法测算，左手、腕骨发育成熟度评分",
            }
            for old_name, new_name in suffix_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df[new_name] = df[old_name]

        # 提取需要的列
        source_cols = [col for col in all_cols if col not in computed_col_names]
        existing_cols = [col for col in source_cols if col in df.columns]
        missing_cols = [col for col in source_cols if col not in df.columns]
        if missing_cols:
            print(f"  警告: 缺失 {len(missing_cols)} 列: {missing_cols[:3]}...")

        df = df[existing_cols].copy()

        # Tanner分期处理
        if "Tanner分期" in df.columns:
            df["Tanner分期"] = (
                df["Tanner分期"].astype(str).str.replace(r"^B", "", regex=True)
            )
            df["Tanner分期"] = df["Tanner分期"].replace("", "1")
            df["Tanner分期"] = pd.to_numeric(df["Tanner分期"], errors="coerce")
            print(f"  处理: Tanner分期 提取数字（B2->2）")

        # LH检测下限处理
        if "基础血清促黄体生成激素（LH）" in df.columns:
            count = (df["基础血清促黄体生成激素（LH）"] == "<0.1").sum()
            if count > 0:
                df.loc[
                    df["基础血清促黄体生成激素（LH）"] == "<0.1",
                    "基础血清促黄体生成激素（LH）",
                ] = 0.05
                print(
                    f"  处理: 基础血清促黄体生成激素（LH） 中 {count} 个 '<0.1' -> 0.05"
                )

        # FSH检测下限处理
        if "基础血清卵泡刺激素（FSH）" in df.columns:
            count = (df["基础血清卵泡刺激素（FSH）"] == "<0.1").sum()
            if count > 0:
                df.loc[
                    df["基础血清卵泡刺激素（FSH）"] == "<0.1",
                    "基础血清卵泡刺激素（FSH）",
                ] = 0.05
                print(f"  处理: 基础血清卵泡刺激素（FSH） 中 {count} 个 '<0.1' -> 0.05")

        # 乳核编码
        if "乳核" in df.columns:
            df["乳核"] = (
                df["乳核"]
                .astype(str)
                .apply(lambda x: 1 if "有" in x else (0 if "无" in x else pd.NA))
            )
            print(f"  编码: 乳核 (有->1, 无->0)")

        # 乳晕色素沉着编码
        if "乳晕色素沉着" in df.columns:

            def encode_ruyun(x):
                x = str(x)
                if "无" in x or "未见" in x or "未" in x:
                    return 0
                elif "稍" in x or "轻微" in x or "少许" in x:
                    return 1
                elif "色素沉着" in x or "着色" in x or "黑" in x or "深" in x:
                    return 2
                return pd.NA

            df["乳晕色素沉着"] = df["乳晕色素沉着"].apply(encode_ruyun)
            print(f"  编码: 乳晕色素沉着 (无->0, 稍有->1, 有->2)")

        # 有无阴毛/腋毛编码
        for col in ["有无阴毛", "有无腋毛"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(lambda x: 1 if "有" in x else (0 if "无" in x else pd.NA))
                )
                print(f"  编码: {col} (有->1, 无->0)")

        # 对 ZERO_AS_NA_COLS 中的列，将 0 值转换为 NaN（在计算派生列之前）
        for col in ZERO_AS_NA_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)

        # 计算派生列
        for col_name, config in COMPUTED_COLUMNS.items():
            depends = config["depends"]
            if all(d in df.columns for d in depends):
                for d in depends:
                    df[d] = pd.to_numeric(df[d], errors="coerce")
                df[col_name] = config["formula"](df)
                df[col_name] = df[col_name].replace([np.inf, -np.inf], pd.NA)
                print(f"  计算: {col_name}")

        # 按定义顺序重排列
        final_cols = [col for col in all_cols if col in df.columns]
        df = df[final_cols]

        # 空值处理
        df = df.replace("", pd.NA)
        missing_count = df.isna().sum().sum()
        print(f"  空值: {missing_count} 个（保留为NaN）")

        # 过滤年龄：只保留4-8岁（4 <= 年龄 < 9）
        if "年龄" in df.columns:
            before_filter = len(df)
            df["年龄"] = pd.to_numeric(df["年龄"], errors="coerce")
            df = df[(df["年龄"] >= 4) & (df["年龄"] < 9)].reset_index(drop=True)
            print(f"  年龄过滤(4-8岁): {before_filter} -> {len(df)} 行")

        # 过滤乳腺数据缺失样本：删除任一乳腺特征缺失的样本
        breast_cols = [
            "左乳腺体宽（cm）",
            "左乳腺体厚（cm）",
            "右乳腺体宽（cm）",
            "右乳腺体厚（cm）",
        ]
        breast_cols_exist = [c for c in breast_cols if c in df.columns]
        if breast_cols_exist:
            before_filter = len(df)
            df = df.dropna(subset=breast_cols_exist, how="any").reset_index(drop=True)
            print(f"  乳腺缺失过滤: {before_filter} -> {len(df)} 行")

        # 过滤阴毛缺失样本
        if "有无阴毛" in df.columns:
            before_filter = len(df)
            df = df.dropna(subset=["有无阴毛"], how="any").reset_index(drop=True)
            print(f"  阴毛缺失过滤: {before_filter} -> {len(df)} 行")

        # 过滤Tanner分期缺失样本
        if "Tanner分期" in df.columns:
            before_filter = len(df)
            df = df.dropna(subset=["Tanner分期"], how="any").reset_index(drop=True)
            print(f"  Tanner缺失过滤: {before_filter} -> {len(df)} 行")

        print(f"  保留: {len(df)} 行 × {len(df.columns)} 列")

        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"  保存: {csv_file}")
        return True

    except Exception as e:
        print(f"  失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("XLSX转CSV - 提取关键列")
    print("=" * 60)

    print(f"关键列: {len(get_all_columns())} 个")

    files = [
        (
            "../input/性早熟数据激发试验正常组.xlsx",
            "../input/性早熟数据激发试验正常组_new.csv",
            False,
        ),
        (
            "../input/激发试验确诊性早熟组数据.xlsx",
            "../input/激发试验确诊性早熟组数据_new.csv",
            True,
        ),
    ]

    for xlsx, csv, is_disease in files:
        if os.path.exists(xlsx):
            convert_xlsx_to_csv(xlsx, csv, is_disease=is_disease)
        else:
            print(f"\n文件不存在: {xlsx}")

    print("\n" + "=" * 60)
    print("转换完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
