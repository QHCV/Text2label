from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 非GUI模式
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
import os
from datetime import datetime
import logging
from config import Config

# 配置日志
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    filename=Config.LOG_FILE
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows系统可用字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

@app.template_filter('enumerate')
def jinja_enumerate(sequence, start=0):
    return enumerate(sequence, start=start)

def get_categories():
    """获取所有分类类别"""
    try:
        categories = [d.replace('_sampled', '') for d in os.listdir(app.config['DATA_DIR'])
                    if d.endswith('_sampled') and not d.endswith('_processed_sampled')]
        logger.info(f"获取到 {len(categories)} 个分类类别")
        return categories
    except Exception as e:
        logger.error(f"获取分类类别时发生错误: {str(e)}")
        return []

def get_file_sequence(category):
    """获取指定类别的文件序列"""
    try:
        folder = os.path.join(app.config['DATA_DIR'], f"{category}_sampled")
        files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
        logger.info(f"类别 {category} 获取到 {len(files)} 个文件")
        return files
    except Exception as e:
        logger.error(f"获取文件序列时发生错误: {str(e)}")
        return []

def get_csv_path(category):
    """获取CSV文件路径"""
    return os.path.join(app.config['DATA_DIR'], f"{category}_annotations.csv")

def init_csv(category):
    """初始化CSV文件"""
    try:
        csv_path = get_csv_path(category)
        if os.path.exists(csv_path):
            return

        # 生成分类器标签
        processed_dir = os.path.join(app.config['DATA_DIR'], f"{category}_processed_sampled")
        classifier_labels = {}
        for f in get_file_sequence(category):
            classifier_labels[f] = 1 if os.path.exists(os.path.join(processed_dir, f)) else 0

        # 使用DictWriter确保列名准确
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['filename', 'classifier_label', 'human_label']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for filename in get_file_sequence(category):
                writer.writerow({
                    'filename': filename,
                    'classifier_label': classifier_labels[filename],
                    'human_label': ''
                })
        logger.info(f"成功初始化 {category} 的CSV文件")
    except Exception as e:
        logger.error(f"初始化CSV文件时发生错误: {str(e)}")
        raise

def load_annotations(category):
    """加载标注数据"""
    try:
        init_csv(category)
        annotations = {}
        csv_path = get_csv_path(category)
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations[row['filename']] = {
                    'classifier': row['classifier_label'],
                    'human': row['human_label'] or ''
                }
        logger.info(f"成功加载 {category} 的标注数据")
        return annotations
    except Exception as e:
        logger.error(f"加载标注数据时发生错误: {str(e)}")
        return {}

def save_annotation(category, filename, label):
    """保存标注数据"""
    try:
        if label not in ['0', '1', '']:
            raise ValueError("无效的标签值")

        csv_path = get_csv_path(category)
        rows = []

        # 读取现有数据
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        # 更新目标行
        for row in rows:
            if row['filename'] == filename:
                row['human_label'] = label
                break

        # 写入更新
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"成功保存 {category}/{filename} 的标注数据")
    except Exception as e:
        logger.error(f"保存标注数据时发生错误: {str(e)}")
        raise

@app.route('/')
def index():
    """首页路由"""
    try:
        # 预初始化所有类别
        for category in get_categories():
            init_csv(category)
        return render_template('index.html', categories=get_categories())
    except Exception as e:
        logger.error(f"首页加载时发生错误: {str(e)}")
        return "服务器错误", 500

@app.route('/category/<category>')
def category_view(category):
    """类别视图路由"""
    try:
        init_csv(category)
        files = get_file_sequence(category)
        annotations = load_annotations(category)
        progress = sum(1 for f in files if annotations[f]['human'].strip())
        return render_template('category.html',
                             category=category,
                             files=files,
                             annotations=annotations,
                             progress=f"{progress}/{len(files)}")
    except Exception as e:
        logger.error(f"类别视图加载时发生错误: {str(e)}")
        return "服务器错误", 500

@app.route('/annotate/<category>/<filename>', methods=['GET', 'POST'])
def annotate(category, filename):
    """标注路由"""
    try:
        files = get_file_sequence(category)
        if filename not in files:
            return "文件不存在", 404

        current_idx = files.index(filename)
        annotations = load_annotations(category)

        if request.method == 'POST':
            label = request.form.get('label', '')
            save_annotation(category, filename, label)
            next_idx = current_idx + 1 if current_idx < len(files) - 1 else current_idx
            return redirect(url_for('annotate', category=category, filename=files[next_idx]))

        # 加载文件内容
        file_path = os.path.join(app.config['DATA_DIR'], f"{category}_sampled", filename)
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        return render_template('annotate.html',
                             category=category,
                             filename=filename,
                             content=content,
                             current_idx=current_idx + 1,
                             total=len(files),
                             files=files,
                             annotations=annotations,
                             annotation=annotations[filename]['human'])
    except FileNotFoundError:
        logger.error(f"文件不存在: {category}/{filename}")
        return "文件不存在", 404
    except Exception as e:
        logger.error(f"标注页面加载时发生错误: {str(e)}")
        return "服务器错误", 500

def calculate_metrics(category):
    """计算分类指标"""
    try:
        csv_path = get_csv_path(category)
        if not os.path.exists(csv_path):
            return {'error': 'CSV文件不存在'}

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            df = pd.read_csv(f, dtype={'human_label': str, 'classifier_label': str})

            # 统一清洗规则
            df['human_label'] = df['human_label'].str.strip().replace({'': np.nan, 'nan': np.nan})
            df['classifier_label'] = df['classifier_label'].str.strip()

            # 修改有效数据筛选条件
            valid_data = df[
                (df['human_label'].isin(['0', '1'])) &
                (df['classifier_label'].isin(['0', '1'])) &
                (df['human_label'].notna()) &
                (df['classifier_label'].notna())
            ]

            if len(valid_data) < 3:
                return {'error': f'有效样本不足（{len(valid_data)}），至少需要3个'}

            y_true = valid_data['human_label'].astype(int)
            y_pred = valid_data['classifier_label'].astype(int)

        return {
            'precision': round(precision_score(y_true, y_pred, zero_division=0), 3),
            'recall': round(recall_score(y_true, y_pred, zero_division=0), 3),
            'f1': round(f1_score(y_true, y_pred, zero_division=0), 3),
            'total': len(valid_data)
        }
    except Exception as e:
        logger.error(f"计算指标时发生错误: {str(e)}")
        return {'error': f'计算异常：{str(e)}'}

def generate_charts():
    """生成可视化图表"""
    try:
        all_metrics = {}
        vis_dir = os.path.join(app.static_folder, 'results')
        os.makedirs(vis_dir, exist_ok=True)

        # 清空旧图表
        for f in os.listdir(vis_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(vis_dir, f))

        for category in get_categories():
            try:
                metrics = calculate_metrics(category)
                if 'error' in metrics:
                    logger.warning(f"跳过 {category}：{metrics['error']}")
                    continue

                # 单类别柱状图生成
                plt.figure(figsize=(8, 4))
                plt.bar(['Precision', 'Recall', 'F1'],
                        [metrics['precision'], metrics['recall'], metrics['f1']],
                        color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                plt.ylim(0, 1.1)
                plt.title(f'{category} 分类指标')
                plt.savefig(os.path.join(vis_dir, f'{category}_metrics.png'),
                            bbox_inches='tight', dpi=100)
                plt.close()

                all_metrics[category] = metrics

            except Exception as e:
                logger.error(f"{category}图表生成失败：{str(e)}")
                continue

        # 全局对比图生成
        if all_metrics:
            plt.figure(figsize=(14, 8))
            categories = list(all_metrics.keys())
            x = np.arange(len(categories))
            width = 0.2

            metrics = ['precision', 'recall', 'f1']
            colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
            for i, metric in enumerate(metrics):
                values = [all_metrics[c][metric] for c in categories]
                plt.bar(x + i * width, values, width, label=metric.upper(), color=colors[i])

            plt.xticks(x + width, [f"{c}\n样本数：{all_metrics[c]['total']}" for c in categories], rotation=25,
                       ha='right')
            plt.ylabel('得分', fontsize=12)
            plt.ylim(0, 1.15)
            plt.title('跨类别分类性能对比（人工标注与分类器双有效样本）', pad=20, fontsize=14)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # 添加数值标签
            for i, category in enumerate(categories):
                for j, metric in enumerate(metrics):
                    value = all_metrics[category][metric]
                    plt.text(x[i] + j * width, value + 0.02, f'{value:.2f}',
                             ha='center', fontsize=9, color=colors[j])

            plt.savefig(os.path.join(vis_dir, 'global_comparison.png'),
                        bbox_inches='tight', dpi=120)
            plt.close()

        logger.info("图表生成完成")
        return vis_dir
    except Exception as e:
        logger.error(f"图表生成过程中发生错误: {str(e)}")
        return None

@app.route('/analysis')
def analysis():
    """分析页面路由"""
    try:
        force_refresh = request.args.get('refresh', '0') == '1'
        vis_dir = generate_charts() if force_refresh else get_existing_charts()
        if not vis_dir:
            return "图表生成失败", 500
        return render_template('analysis.html',
                             categories=get_categories(),
                             chart_dir=vis_dir,
                             now=datetime.now())
    except Exception as e:
        logger.error(f"分析页面加载时发生错误: {str(e)}")
        return "服务器错误", 500

def get_existing_charts():
    """获取已有图表"""
    try:
        vis_dir = os.path.join(app.static_folder, 'results')
        if not os.path.exists(vis_dir) or not os.listdir(vis_dir):
            return generate_charts()
        return vis_dir
    except Exception as e:
        logger.error(f"获取已有图表时发生错误: {str(e)}")
        return None

@app.route('/debug/charts')
def debug_charts():
    """图表调试接口"""
    try:
        vis_dir = generate_charts()
        if not vis_dir:
            return "图表生成失败", 500
        files = os.listdir(vis_dir)
        return {
            'static_folder': app.static_folder,
            'results_dir': vis_dir,
            'existing_files': files
        }
    except Exception as e:
        logger.error(f"调试接口访问时发生错误: {str(e)}")
        return "服务器错误", 500

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, port=Config.PORT, host=Config.HOST)