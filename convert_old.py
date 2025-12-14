#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XLSX转CSV
对数值型数据计算多次测量的平均值，对分类型数据使用第一次测量
"""

import pandas as pd
import re
import os
from pathlib import Path
import json

# 0通常表示“未增厚”或“未测到”的列，需要在统计时将其视为缺失
STRUCTURAL_ZERO_COLUMNS = {
    "内膜厚度（备注：没有增厚就写0）",
    "右卵巢体积（长X宽X厚X0.5233）",
    "左卵巢体积（长X宽X厚X0.5233）",
    "子宫厚（cm）",
}

# 激素/生化类指标在临床表格中若记录为0通常表示未检测到，需要忽略
HORMONE_ZERO_COLUMNS = {
    "基础血清促黄体生成激素（LH）",
    "基础血清卵泡刺激素（FSH）",
    "基础血清卵泡刺激素（FSH）峰值",
    "血清促黄体生成激素（LH）峰值",
    "LH峰值/FSH峰值",
    "雌二醇",
    "总睾酮",
    "泌乳素",
    "孕酮",
}


def _normalize_measurement_name(column_name):
    """去除末尾的测量序号（1/2/3），但保留小数形式的列名。"""

    name = str(column_name)
    if re.search(r"\.\d+$", name):
        return name
    return re.sub(r"\d+$", "", name)


def _zero_handling_type(column_name):
    """返回列名的零值处理类别（structural/hormone/None）。"""

    normalized = _normalize_measurement_name(column_name)
    if normalized in STRUCTURAL_ZERO_COLUMNS:
        return "structural"
    if normalized in HORMONE_ZERO_COLUMNS:
        return "hormone"
    return None


def calculate_multi_measurement_averages(df, target_col):
    """
    计算多次测量的平均值或选择第一次测量的值

    Args:
        df: 原始DataFrame
        target_col: 目标列名

    Returns:
        Series: 计算得到的平均值序列
    """
    # 查找所有相关的测量列（后缀1、2、3等）
    # 精确匹配：目标列名 + 数字后缀
    pattern = re.compile(rf"^{re.escape(str(target_col))}[123]$")
    related_cols = [col for col in df.columns if pattern.match(str(col))]
    related_cols.sort(key=lambda col: int(str(col)[-1]))

    if not related_cols:
        return None

    # 实验室检验数据列表（使用第一次测量，不计算均值）
    lab_test_fields = [
        "基础血清促黄体生成激素（LH）",
        "基础血清卵泡刺激素（FSH）",
        "血清促黄体生成激素（LH）峰值",
        "基础血清卵泡刺激素（FSH）峰值",
        "LH峰值/FSH峰值",
        "雌二醇",
        "总睾酮",
        "泌乳素",
        "孕酮",
    ]

    # 检查是否为实验室检验数据
    if target_col in lab_test_fields:
        # 实验室检验数据：使用第一次血检数据
        first_col = next(
            (col for col in related_cols if col.endswith("1")), related_cols[0]
        )
        print(f"    {target_col}: 使用第一次血检数据 ({first_col})")
        return df[first_col]

    # 数值型数据列表（需要计算平均值）
    # 注意：血检相关关键词（血清、激素、LH、FSH等）保留在此处用于模式识别
    # 但实际的实验室检验字段已在lab_test_fields中明确列出，会优先使用第一次测量
    numeric_keywords = [
        "宽（cm）",
        "厚（cm）",
        "长（cm）",
        "体积",
        "直径",
        "血清",
        "激素",
        "LH",
        "FSH",
        "孕酮",
        "睾酮",
        "雌二醇",
        "泌乳素",
        "骨龄",
        "评分",
        "差异",
        "BMI",
        "身高",
        "体重",
        "内膜厚度",
    ]

    # 判断是否为数值型数据
    is_numeric = any(keyword in target_col for keyword in numeric_keywords)

    zero_handling = _zero_handling_type(target_col)

    if is_numeric:
        # 数值型数据：计算平均值
        numeric_cols = []
        for col in related_cols:
            # 转换为数值型
            numeric_series = pd.to_numeric(df[col], errors="coerce")

            if zero_handling:
                numeric_series = numeric_series.mask(numeric_series == 0)

            if not numeric_series.isna().all():  # 如果不是全部为空
                numeric_cols.append(numeric_series)

        if numeric_cols:
            # 计算平均值（忽略NaN）
            result = pd.concat(numeric_cols, axis=1).mean(axis=1, skipna=True)
            suffix = "（0值已排除）" if zero_handling else ""
            print(f"    {target_col}: 使用 {len(numeric_cols)} 次测量的平均值{suffix}")
            return result

    # 非数值型数据：使用第一次测量
    first_col = next(
        (col for col in related_cols if col.endswith("1")), related_cols[0]
    )
    print(f"    {target_col}: 使用第一次测量值 ({first_col})")
    return df[first_col]


def standardize_disease_group_columns(df, target_columns):
    """
    标准化早熟组的列名，确保能匹配正常组
    对数值型数据计算多次测量的平均值，对分类型数据使用第一次测量

    Args:
        df: 早熟组DataFrame
        target_columns: 目标列名列表（正常组的列名）

    Returns:
        标准化后的DataFrame
    """
    print("  开始处理多次测量数据...")

    # 创建结果DataFrame
    df_result = pd.DataFrame(index=df.index)

    for target_col in target_columns:
        if target_col in df.columns:
            # 如果目标列直接存在，直接使用
            df_result[target_col] = df[target_col]
            if _zero_handling_type(target_col):
                numeric_series = pd.to_numeric(df_result[target_col], errors="coerce")
                zero_mask = numeric_series == 0
                zero_count = zero_mask.sum()
                if zero_count > 0:
                    numeric_series = numeric_series.mask(zero_mask)
                    print(f"    {target_col}: 直接列中 {zero_count} 个0值已替换为NaN")
                df_result[target_col] = numeric_series
            print(f"    {target_col}: 直接使用原列")
        else:
            # 尝试计算多次测量的平均值
            avg_result = calculate_multi_measurement_averages(df, target_col)
            if avg_result is not None:
                df_result[target_col] = avg_result
            else:
                df_result[target_col] = None
                print(f"    {target_col}: 未找到相关测量数据")

    print(f"  处理完成，生成 {len(target_columns)} 个标准化列")
    return df_result


def align_normal_group_columns(df, target_columns):
    """
    对齐正常组的列，确保顺序和名称一致

    Args:
        df: 正常组DataFrame
        target_columns: 目标列名列表

    Returns:
        对齐后的DataFrame
    """

    # 检查缺失的列
    missing_cols = set(target_columns) - set(df.columns)
    if missing_cols:
        print(f"  正常组缺少以下列: {missing_cols}")
        # 为缺少的列添加空列
        for col in missing_cols:
            df[col] = None

    # 按照目标列的顺序重新排列
    df_final = df[target_columns].copy()

    for col in df_final.columns:
        zero_handling = _zero_handling_type(col)
        if zero_handling:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
            zero_mask = df_final[col] == 0
            zero_count = zero_mask.sum()
            if zero_count > 0:
                df_final.loc[zero_mask, col] = pd.NA
                print(f"  正常组: {col} 中的 {zero_count} 个0值已替换为NaN")

    return df_final


def add_bilateral_averages(df):
    """
    为左右对称的数据添加平均值列（不包括卵泡个数）

    Args:
        df: DataFrame

    Returns:
        添加了平均值列的DataFrame
    """
    print("  添加左右对称数据的平均值列...")

    # 定义左右对称的数据对（不包括卵泡个数）
    bilateral_pairs = [
        # 卵巢相关
        ("左卵巢长（cm）", "右卵巢长（cm）", "卵巢长平均值（cm）"),
        ("左卵巢宽（cm）", "右卵巢宽（cm）", "卵巢宽平均值（cm）"),
        ("左卵巢厚（cm）", "右卵巢厚（cm）", "卵巢厚平均值（cm）"),
        (
            "左卵巢体积（长X宽X厚X0.5233）",
            "右卵巢体积（长X宽X厚X0.5233）",
            "卵巢体积平均值",
        ),
        # 乳腺体相关
        ("左乳腺体宽（cm）", "右乳腺体宽（cm）", "乳腺体宽平均值（cm）"),
        ("左乳腺体厚（cm）", "右乳腺体厚（cm）", "乳腺体厚平均值（cm）"),
    ]

    for left_col, right_col, avg_col in bilateral_pairs:
        # 检查左右列是否存在
        if left_col in df.columns and right_col in df.columns:
            # 转换为数值型
            left_values = pd.to_numeric(df[left_col], errors="coerce")
            right_values = pd.to_numeric(df[right_col], errors="coerce")

            # 计算平均值（忽略NaN）
            df[avg_col] = left_values.combine(
                right_values,
                lambda x, y: (
                    (x + y) / 2
                    if pd.notna(x) and pd.notna(y)
                    else x if pd.notna(x) else y if pd.notna(y) else pd.NA
                ),
                fill_value=pd.NA,
            )

            # 统计有效计算的数量
            both_valid = left_values.notna() & right_values.notna()
            left_only = left_values.notna() & right_values.isna()
            right_only = left_values.isna() & right_values.notna()

            print(
                f"    {avg_col}: 双侧={both_valid.sum()}, 仅左={left_only.sum()}, 仅右={right_only.sum()}"
            )
        else:
            print(f"    {avg_col}: 跳过（列不存在）")

    return df


def convert_xlsx_to_csv_improved(
    xlsx_file,
    csv_file,
    skip_rows=0,
    is_disease_group=False,
    target_columns=None,
    exclude_diagnosis_cols=True,
    columns_to_exclude=None,
):
    """
    改进的XLSX转CSV函数，确保列名完全匹配

    Args:
        xlsx_file: 输入的XLSX文件路径
        csv_file: 输出的CSV文件路径
        skip_rows: 跳过的行数（默认0）
        is_disease_group: 是否是早熟组数据
        target_columns: 目标列名列表（标准化后的列名）
        exclude_diagnosis_cols: 是否排除诊断报告相关列
        columns_to_exclude: 要排除的列名列表

    Returns:
        bool: 转换成功返回True，失败返回False
    """

    try:
        # 读取Excel文件
        print(f"读取XLSX文件: {xlsx_file}")
        if skip_rows > 0:
            data = pd.read_excel(xlsx_file, skiprows=skip_rows)
            print(f"  - 跳过前 {skip_rows} 行")
        else:
            data = pd.read_excel(xlsx_file)

        print(f"  - 原始数据: {data.shape[0]} 行 × {data.shape[1]} 列")

        # 过滤患者编号为空的行（第一列）
        if len(data.columns) >= 1:
            patient_id_col = data.columns[0]
            original_count = len(data)
            data = data[data[patient_id_col].notna() & (data[patient_id_col] != "")]
            filtered_count = original_count - len(data)
            if filtered_count > 0:
                print(f"  - 删除患者编号为空的行: {filtered_count} 行")
                print(f"  - 过滤后数据: {data.shape[0]} 行 × {data.shape[1]} 列")

        # 排除列（使用模糊匹配删除包含关键词的列）
        if exclude_diagnosis_cols and columns_to_exclude:
            # 模糊匹配：删除列名中包含任意排除关键词的列
            cols_to_drop = []
            for col in data.columns:
                col_str = str(col)
                if any(keyword in col_str for keyword in columns_to_exclude):
                    cols_to_drop.append(col)

            if cols_to_drop:
                print(
                    f"从数据中排除相关列 ({len(cols_to_drop)}个): {cols_to_drop[:5]}..."
                )
                data = data.drop(columns=cols_to_drop)

        # 如果提供了目标列名，进行列名对齐
        if target_columns:
            print("进行列名标准化对齐...")

            if is_disease_group:
                # 早熟组：映射列名并对齐
                data = standardize_disease_group_columns(data, target_columns)
                print(f"  - 早熟组列名已标准化，对齐到 {len(target_columns)} 列")
            else:
                # 正常组：直接对齐
                data = align_normal_group_columns(data, target_columns)
                print(f"  - 正常组列名已对齐，包含 {len(target_columns)} 列")

        # 添加左右对称数据的平均值列
        print("计算左右对称数据的平均值...")
        data = add_bilateral_averages(data)

        # 过滤极端异常值（保留99.82%的数据，删除包含极端值的行）
        print("过滤极端异常值（保留99.82%的数据，已排除0值和NaN）...")
        original_rows = len(data)
        rows_to_drop = set()

        for col in data.columns:
            # 跳过非数值列和ID列
            if col in ["患者编号", "性别", "民族", "籍贯", "出生日期", "就诊日期"]:
                continue

            # 转换为数值型
            numeric_data = pd.to_numeric(data[col], errors="coerce")
            valid_data = numeric_data.dropna()

            if len(valid_data) == 0:
                continue

            # 排除0值进行统计
            non_zero_data = valid_data[valid_data != 0]

            if len(non_zero_data) == 0:
                continue

            # 计算99.82%分位数作为阈值（最优平衡点）
            threshold = non_zero_data.quantile(0.9982)

            # 找出超过阈值的行索引
            extreme_mask = (numeric_data > threshold) & (numeric_data.notna())
            if extreme_mask.sum() > 0:
                extreme_rows = data.index[extreme_mask].tolist()
                rows_to_drop.update(extreme_rows)

        # 删除包含极端值的行
        if rows_to_drop:
            data = data.drop(index=list(rows_to_drop))
            filtered_rows = len(rows_to_drop)
            print(f"  - 删除了 {filtered_rows} 行包含极端异常值的数据")
            print(f"  - 剩余: {len(data)} 行 (原始: {original_rows} 行)")
        else:
            print("  - 未发现需要过滤的极端异常值")

        # 保存为CSV
        print(f"保存为CSV文件: {csv_file}")
        try:
            data.to_csv(csv_file, index=False, encoding="utf-8-sig")
        except:
            data.to_csv(csv_file, index=False, encoding="gbk")

        # 验证保存结果
        try:
            verify_data = pd.read_csv(csv_file, encoding="utf-8-sig")
        except:
            verify_data = pd.read_csv(csv_file, encoding="gbk")

        print(f"  - 验证: {verify_data.shape[0]} 行 × {verify_data.shape[1]} 列")

        # 显示对齐后的列名示例
        print("  - 标准化后列名示例:")
        for i, col in enumerate(verify_data.columns[:10]):
            print(f"    {i+1:2d}. {col}")
        if len(verify_data.columns) > 10:
            print(f"    ... 还有 {len(verify_data.columns) - 10} 个列")

        return True

    except Exception as e:
        print(f"转换失败: {e}")
        return False


def get_target_columns():
    """
    获取标准化的目标列名列表（基于正常组的50个列）
    """

    # 这里列出正常组的50个列名（基于分析结果）
    target_columns = [
        "患者编号",
        "年龄",
        "出生日期",
        "性别",
        "籍贯",
        "民族",
        "身高（cm）",
        "体重（kg）",
        "BMI （体重Kg÷身高m）",
        "就诊日期",
        "LH峰值/FSH峰值≥0.6",
        "LH峰值＞5.0U/I",
        "LH峰值/FSH峰值",
        "Tanner分期",
        "乳晕色素沉着",
        "乳核",
        "右乳腺体宽（cm）",
        "右乳腺体厚（cm）",
        "左乳腺体宽（cm）",
        "左乳腺体厚（cm）",
        "右侧卵泡个数",
        "右卵巢长（cm）",
        "右卵巢宽（cm）",
        "右卵巢厚（cm）",
        "右卵巢体积（长X宽X厚X0.5233）",
        "左侧卵泡个数",
        "左卵巢长（cm）",
        "左卵巢宽（cm）",
        "左卵巢厚（cm）",
        "左卵巢体积（长X宽X厚X0.5233）",
        "子宫长（cm）",
        "子宫宽（cm）",
        "子宫厚（cm）",
        "内膜厚度（备注：没有增厚就写0）",
        "最大卵泡直径直径",
        "基础血清促黄体生成激素（LH）",
        "基础血清卵泡刺激素（FSH）",
        "基础血清卵泡刺激素（FSH）峰值",
        "孕酮",
        "总睾酮",
        "按CHN法测算，左手、腕骨发育成熟度评分",
        "有无腋毛",
        "有无阴毛",
        "检查部位",
        "泌乳素",
        "生物年龄和骨龄之间的差异",
        "血清促黄体生成激素（LH）峰值",
        "雌二醇",
        "骨龄(岁)",
        "月经初潮年龄",
    ]

    return target_columns


def main():
    print("=" * 80)
    print("XLSX转CSV")
    print("数值型数据：计算多次测量平均值 | 分类型数据：使用第一次测量")
    print("=" * 80)

    # 确保input文件夹存在
    input_dir = "./input"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"创建input文件夹: {input_dir}")

    # ===== 配置区：选择要排除的列大类 =====
    column_categories = {
        "基本临床资料": [
            "年龄",
            "出生日期",
            "性别",
            "籍贯",
            "民族",
            "身高（cm）",
            "体重（kg）",
            "BMI （体重Kg÷身高m）",
            "就诊日期",
            "月经初潮年龄",
            "乳核",
            "乳晕色素沉着",
            "有无阴毛",
            "有无腋毛",
            "Tanner分期",
        ],
        "实验室检验": [
            "基础血清促黄体生成激素（LH）",
            "基础血清卵泡刺激素（FSH）",
            "血清促黄体生成激素（LH）峰值",
            "基础血清卵泡刺激素（FSH）峰值",
            "LH峰值/FSH峰值",
            "雌二醇",
            "总睾酮",
            "泌乳素",
            "孕酮",
        ],
        "实验室检验性激素激发试验（确诊性早熟指标）": [
            "LH峰值＞5.0U/I",
            "LH峰值/FSH峰值≥0.6",
        ],
        "超声报告": [
            "子宫长（cm）",
            "子宫宽（cm）",
            "子宫厚（cm）",
            "内膜厚度（备注：没有增厚就写0）",
            "最大卵泡直径直径",
            "左侧卵泡个数",
            "右侧卵泡个数",
            "左卵巢长（cm）",
            "左卵巢宽（cm）",
            "左卵巢厚（cm）",
            "左卵巢体积（长X宽X厚X0.5233）",
            "右卵巢长（cm）",
            "右卵巢宽（cm）",
            "右卵巢厚（cm）",
            "右卵巢体积（长X宽X厚X0.5233）",
        ],
        "超声报告（左乳房腺体）": ["左乳腺体宽（cm）", "左乳腺体厚（cm）"],
        "超声报告（右乳房腺体）": ["右乳腺体宽（cm）", "右乳腺体厚（cm）"],
        "左手X线检查": [
            "骨龄(岁)",
            "生物年龄和骨龄之间的差异",
            "按CHN法测算，左手、腕骨发育成熟度评分",
        ],
        "磁共振报告": ["检查部位", "报告结论"],
    }

    # True = 排除该类别，False = 保留该类别
    exclude_config = {
        "基本临床资料": False,
        "实验室检验": False,
        "实验室检验性激素激发试验（确诊性早熟指标）": True,  # 默认排除确诊指标
        "超声报告": False,
        "超声报告（左乳房腺体）": False,
        "超声报告（右乳房腺体）": False,
        "左手X线检查": False,
        "磁共振报告": True,  # 默认排除磁共振报告
    }
    # ===== 配置区结束 =====

    # 定义要保留的实验室检验字段（只保留这三个）
    lab_fields_to_keep = [
        "基础血清促黄体生成激素（LH）",
        "基础血清卵泡刺激素（FSH）",
        "雌二醇",
    ]

    # 生成要排除的列名列表
    columns_to_exclude = []
    for category, should_exclude in exclude_config.items():
        if should_exclude and category in column_categories:
            columns_to_exclude.extend(column_categories[category])
            print(f"配置排除类别: {category} ({len(column_categories[category])}个列)")

    # 特殊处理：实验室检验字段，只保留指定的三个，其他的加入排除列表
    if "实验室检验" in column_categories:
        lab_fields_to_exclude = [
            field
            for field in column_categories["实验室检验"]
            if field not in lab_fields_to_keep
        ]
        if lab_fields_to_exclude:
            columns_to_exclude.extend(lab_fields_to_exclude)
            print(
                f"实验室检验字段: 保留 {len(lab_fields_to_keep)} 个，排除 {len(lab_fields_to_exclude)} 个"
            )
            print(f"  保留字段: {', '.join(lab_fields_to_keep)}")

    # 获取标准化的目标列名
    target_columns_full = get_target_columns()

    # 根据排除配置过滤目标列
    target_columns = [
        col for col in target_columns_full if col not in columns_to_exclude
    ]

    if columns_to_exclude:
        removed_count = len(target_columns_full) - len(target_columns)
        print(f"从目标列中排除了 {removed_count} 个列")

    print(f"最终目标列名数量: {len(target_columns)}")

    # 定义文件映射关系
    file_mappings = [
        {
            "xlsx": "./input/性早熟数据激发试验正常组.xlsx",
            "csv": "./input/性早熟数据激发试验正常组.csv",
            "skip": 1,
            "is_disease_group": False,
            "description": "正常组数据",
        },
        {
            "xlsx": "./input/激发试验确诊性早熟组数据.xlsx",
            "csv": "./input/激发试验确诊性早熟组数据.csv",
            "skip": 1,
            "is_disease_group": True,
            "description": "性早熟组数据",
        },
    ]

    success_count = 0
    total_count = len(file_mappings)

    for mapping in file_mappings:
        xlsx_file = mapping["xlsx"]
        csv_file = mapping["csv"]
        skip_rows = mapping["skip"]
        is_disease_group = mapping["is_disease_group"]
        description = mapping["description"]

        print(f"\n处理 {description}...")
        print(f"输入文件: {xlsx_file}")
        print(f"输出文件: {csv_file}")

        # 检查输入文件是否存在
        if not os.path.exists(xlsx_file):
            print(f"输入文件不存在: {xlsx_file}")
            continue

        # 执行改进的转换
        if convert_xlsx_to_csv_improved(
            xlsx_file,
            csv_file,
            skip_rows,
            is_disease_group,
            target_columns,
            exclude_diagnosis_cols=True,
            columns_to_exclude=columns_to_exclude,
        ):
            print(f"{description} 转换成功!")
            success_count += 1
        else:
            print(f"{description} 转换失败!")

    # 验证列名对齐结果
    print(f"\n" + "=" * 80)
    print("列名对齐验证:")
    print("=" * 80)

    try:
        normal_df = pd.read_csv(
            "./input/性早熟数据激发试验正常组.csv", encoding="utf-8-sig"
        )
        disease_df = pd.read_csv(
            "./input/激发试验确诊性早熟组数据.csv", encoding="utf-8-sig"
        )

        print(f"正常组: {normal_df.shape[0]} 行 × {normal_df.shape[1]} 列")
        print(f"早熟组: {disease_df.shape[0]} 行 × {disease_df.shape[1]} 列")

        # 检查列名是否完全匹配
        normal_cols = set(normal_df.columns)
        disease_cols = set(disease_df.columns)

        if normal_cols == disease_cols:
            print("列名完全匹配成功！")
        else:
            diff_cols = normal_cols.symmetric_difference(disease_cols)
            print(f"仍有 {len(diff_cols)} 个列名不匹配: {diff_cols}")

    except Exception as e:
        print(f"验证失败: {e}")

    # 总结
    print("\n" + "=" * 80)
    print("转换完成统计:")
    print(f"成功: {success_count}/{total_count}")

    if success_count == total_count:
        print("所有文件转换成功！")
        print("CSV文件已生成并完成列名标准化对齐")
        print("可以直接运行机器学习脚本")
    else:
        print("部分文件转换失败，请检查输入文件")

    print("=" * 80)


if __name__ == "__main__":
    main()
