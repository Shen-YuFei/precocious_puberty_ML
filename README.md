# 性早熟预测机器学习模型

Machine learning models for precocious puberty prediction using clinical data.

## 📋 项目简介

本项目基于临床数据，使用多种机器学习算法对儿童性早熟进行预测和诊断辅助。通过分析激素水平、生理特征等多维度指标，实现早期识别和风险评估。

## ✨ 主要功能

- **数据预处理**：处理纵向数据、多次测量值聚合、缺失值处理
- **多模型集成**：TabPFN、XGBoost、Random Forest 等多种算法
- **模型解释性**：基于 SHAP 的特征重要性分析
- **可视化分析**：特征分布、相关性分析、模型性能评估

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

将原始数据文件放置在 `input/` 目录下（该目录已被 `.gitignore` 排除）：

```
input/
├── 激发试验确诊性早熟组数据.xlsx
├── 性早熟数据激发试验正常组.xlsx
└── 列名映射配置.json
```

### 使用流程

1. **数据转换**：运行 `convert.py` 将原始数据转换为 CSV 格式
   ```bash
   python convert.py
   ```

2. **模型训练**：打开 Jupyter Notebook
   ```bash
   jupyter notebook
   ```
   
   依次运行：
   - `TabPFN_Enhanced.ipynb` - TabPFN 模型训练与评估
   - `机器学习.ipynb` - 其他机器学习模型对比
   - `特征可视化.ipynb` - 特征分析与可视化

## 📁 项目结构

```
precocious_puberty_ML/
├── convert.py                  # 数据预处理脚本
├── TabPFN_Enhanced.ipynb       # TabPFN 模型实现
├── 机器学习.ipynb               # ML 模型对比与评估
├── 特征可视化.ipynb             # 特征分析可视化
├── requirements.txt            # Python 依赖
├── input/                      # 原始数据（本地存储，不上传）
├── output/                     # 处理后的数据和模型（不上传）
└── README.md                   # 项目说明文档
```

## 🛠️ 技术栈

- **机器学习框架**：TabPFN, XGBoost, Scikit-learn
- **数据处理**：Pandas, NumPy
- **可视化**：Matplotlib, Seaborn
- **模型解释**：SHAP
- **开发环境**：Jupyter Notebook

## 📊 数据说明

本项目使用的数据包含：
- 激发试验确诊性早熟组数据
- 正常对照组数据
- 多次随访的纵向数据

**注意**：原始数据包含隐私信息，仅在本地使用，不会上传到远程仓库。

## ⚠️ 注意事项

1. **数据隐私**：`input/` 目录包含敏感医疗数据，已配置为仅本地保存
2. **模型文件**：`.pkl` 模型文件体积较大，已排除在版本控制之外
3. **环境依赖**：建议使用虚拟环境避免依赖冲突

## 📝 License

本项目仅用于学术研究和医学辅助诊断。

## 👤 作者

Shen-YuFei

---

**免责声明**：本项目提供的模型和分析结果仅供医学研究参考，不能替代专业医生的临床诊断。
