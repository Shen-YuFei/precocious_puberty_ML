#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XLSX转CSV（严格模式）
只保留关键列，删除任何关键列有缺失值的行
"""

import pandas as pd
import os
import re

# ===== 早熟组关键列定义=====
CRITICAL_COLUMNS_DISEASE = {
    "基本临床资料": [
        "患者编号",
        "年龄",
        "性别",
        "民族",
        "身高（cm）",
        "体重（kg）",
        "BMI （体重Kg÷身高m）",
        "Tanner分期1",
    ],
    "实验室检验": [
        "基础血清促黄体生成激素（LH）1",
        "基础血清卵泡刺激素（FSH）1",
    ],
    "超声报告": [
        "子宫长（cm）1",
        "子宫宽（cm）1",
        "子宫厚（cm）1",
        "子宫体积（长X宽X厚X0.5236）1",
        "最大卵泡直径直径1",
        "左卵巢长（cm）1",
        "左卵巢宽（cm）1",
        "左卵巢厚（cm）1",
        "左卵巢体积（长X宽X厚X0.5233）1",
        "右卵巢长（cm）1",
        "右卵巢宽（cm）1",
        "右卵巢厚（cm）1",
        "右卵巢体积（长X宽X厚X0.5233）1",
    ],
    "左手X线检查": [
        "骨龄(岁)1",
        "骨龄与实际年龄比值",
        "生物年龄和骨龄之间的差异1",
        "按CHN法测算，左手、腕骨发育成熟度评分1",
    ],
}

# ===== 早熟组人工添加列定义 =====
COMPUTED_COLUMNS_DISEASE = {
    "BMI （体重Kg÷身高m）": {
        "formula": lambda df: df["体重（kg）"] / (df["身高（cm）"] / 100) ** 2,
        "depends": ["身高（cm）", "体重（kg）"],
        "category": "基本临床资料",
    },
    "子宫体积（长X宽X厚X0.5236）1": {
        "formula": lambda df: df["子宫长（cm）1"]
        * df["子宫宽（cm）1"]
        * df["子宫厚（cm）1"]
        * 0.5236,
        "depends": ["子宫长（cm）1", "子宫宽（cm）1", "子宫厚（cm）1"],
        "category": "超声报告",
    },
    "左卵巢体积（长X宽X厚X0.5233）1": {
        "formula": lambda df: df["左卵巢长（cm）1"]
        * df["左卵巢宽（cm）1"]
        * df["左卵巢厚（cm）1"]
        * 0.5233,
        "depends": ["左卵巢长（cm）1", "左卵巢宽（cm）1", "左卵巢厚（cm）1"],
        "category": "超声报告",
    },
    "右卵巢体积（长X宽X厚X0.5233）1": {
        "formula": lambda df: df["右卵巢长（cm）1"]
        * df["右卵巢宽（cm）1"]
        * df["右卵巢厚（cm）1"]
        * 0.5233,
        "depends": ["右卵巢长（cm）1", "右卵巢宽（cm）1", "右卵巢厚（cm）1"],
        "category": "超声报告",
    },
    "骨龄与实际年龄比值": {
        "formula": lambda df: (df["骨龄(岁)1"] / df["年龄"]).round(2),
        "depends": ["骨龄(岁)1", "年龄"],
        "category": "左手X线检查",
    },
    "生物年龄和骨龄之间的差异1": {
        "formula": lambda df: df["年龄"] - df["骨龄(岁)1"],
        "depends": ["年龄", "骨龄(岁)1"],
        "category": "左手X线检查",
    },
}


def remove_suffix_1(col_name):
    """去除列名末尾的数字1"""
    return re.sub(r"1$", "", col_name)


def get_normal_columns():
    """获取正常组列名（去除数字后缀1）"""
    cols = {}
    for cat, col_list in CRITICAL_COLUMNS_DISEASE.items():
        cols[cat] = [remove_suffix_1(c) for c in col_list]
    return cols


def get_normal_computed_columns():
    """获取正常组计算列定义（去除数字后缀1）"""
    return {
        "BMI （体重Kg÷身高m）": {
            "formula": lambda df: df["体重（kg）"] / (df["身高（cm）"] / 100) ** 2,
            "depends": ["身高（cm）", "体重（kg）"],
            "category": "基本临床资料",
        },
        "子宫体积（长X宽X厚X0.5236）": {
            "formula": lambda df: df["子宫长（cm）"]
            * df["子宫宽（cm）"]
            * df["子宫厚（cm）"]
            * 0.5236,
            "depends": ["子宫长（cm）", "子宫宽（cm）", "子宫厚（cm）"],
            "category": "超声报告",
        },
        "左卵巢体积（长X宽X厚X0.5233）": {
            "formula": lambda df: df["左卵巢长（cm）"]
            * df["左卵巢宽（cm）"]
            * df["左卵巢厚（cm）"]
            * 0.5233,
            "depends": ["左卵巢长（cm）", "左卵巢宽（cm）", "左卵巢厚（cm）"],
            "category": "超声报告",
        },
        "右卵巢体积（长X宽X厚X0.5233）": {
            "formula": lambda df: df["右卵巢长（cm）"]
            * df["右卵巢宽（cm）"]
            * df["右卵巢厚（cm）"]
            * 0.5233,
            "depends": ["右卵巢长（cm）", "右卵巢宽（cm）", "右卵巢厚（cm）"],
            "category": "超声报告",
        },
        "骨龄与实际年龄比值": {
            "formula": lambda df: (df["骨龄(岁)"] / df["年龄"]).round(2),
            "depends": ["骨龄(岁)", "年龄"],
            "category": "左手X线检查",
        },
        "生物年龄和骨龄之间的差异": {
            "formula": lambda df: df["年龄"] - df["骨龄(岁)"],
            "depends": ["年龄", "骨龄(岁)"],
            "category": "左手X线检查",
        },
    }


def get_all_critical_columns(is_disease=True):
    """获取所有关键列（包括计算列，按定义顺序）"""
    if is_disease:
        return [col for cols in CRITICAL_COLUMNS_DISEASE.values() for col in cols]
    else:
        normal_cols = get_normal_columns()
        return [col for cols in normal_cols.values() for col in cols]


def convert_xlsx_to_csv_dropna(xlsx_file, csv_file, skip_rows=1, is_disease=True):
    """转换XLSX到CSV，只保留关键列，删除有缺失值的行"""
    try:
        print(f"\n读取: {xlsx_file}")
        df = pd.read_excel(xlsx_file, skiprows=skip_rows)
        print(f"  原始: {df.shape[0]} 行 × {df.shape[1]} 列")

        # 根据数据类型选择列定义
        if is_disease:
            critical_columns = CRITICAL_COLUMNS_DISEASE
            computed_columns = COMPUTED_COLUMNS_DISEASE
        else:
            critical_columns = get_normal_columns()
            computed_columns = get_normal_computed_columns()

        # 获取所有列（按顺序）和计算列名
        all_cols = [col for cols in critical_columns.values() for col in cols]
        computed_col_names = set(computed_columns.keys())

        # 只保留存在的非计算列
        source_cols = [col for col in all_cols if col not in computed_col_names]
        existing_cols = [col for col in source_cols if col in df.columns]
        missing_cols = [col for col in source_cols if col not in df.columns]

        if missing_cols:
            print(f"  警告: 缺失 {len(missing_cols)} 列: {missing_cols[:3]}...")

        df = df[existing_cols].copy()

        # Tanner分期值替换：将"B"替换为"B1"
        tanner_col = "Tanner分期1" if is_disease else "Tanner分期"
        if tanner_col in df.columns:
            df[tanner_col] = df[tanner_col].replace("B", "B1")
            print(f"  替换: {tanner_col} 中的 'B' -> 'B1'")

        # 计算人工添加列
        for col_name, config in computed_columns.items():
            depends = config["depends"]
            if all(d in df.columns for d in depends):
                for d in depends:
                    df[d] = pd.to_numeric(df[d], errors="coerce")
                df[col_name] = config["formula"](df)
                print(f"  计算: {col_name}")

        # 按定义顺序重排列
        final_cols = [col for col in all_cols if col in df.columns]
        df = df[final_cols]

        # 删除有缺失值的行
        df = df.replace("", pd.NA)
        original_count = len(df)
        df = df.dropna(how="any")

        print(f"  删除: {original_count - len(df)} 行有缺失值")
        print(f"  保留: {len(df)} 行 × {len(df.columns)} 列")

        # 保存
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"  保存: {csv_file}")
        return True

    except Exception as e:
        print(f"  失败: {e}")
        return False


def main():
    print("=" * 60)
    print("XLSX转CSV（严格模式）- 删除缺失值行")
    print("=" * 60)

    # 显示关键列
    print(f"早熟组关键列: {len(get_all_critical_columns(is_disease=True))} 个")
    print(f"正常组关键列: {len(get_all_critical_columns(is_disease=False))} 个")

    # 文件映射: (xlsx, csv, is_disease)
    files = [
        (
            "./input/性早熟数据激发试验正常组.xlsx",
            "./input/性早熟数据激发试验正常组_dropna.csv",
            False,
        ),
        (
            "./input/激发试验确诊性早熟组数据.xlsx",
            "./input/激发试验确诊性早熟组数据_dropna.csv",
            True,
        ),
    ]

    for xlsx, csv, is_disease in files:
        if os.path.exists(xlsx):
            convert_xlsx_to_csv_dropna(xlsx, csv, is_disease=is_disease)
        else:
            print(f"\n文件不存在: {xlsx}")

    print("\n" + "=" * 60)
    print("转换完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
