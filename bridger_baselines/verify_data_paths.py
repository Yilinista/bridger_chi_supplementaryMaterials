#!/usr/bin/env python3
"""
Data Path Verification Script
验证所有数据路径是否存在并可访问
"""

import os
import pandas as pd
import json
from pathlib import Path

def verify_paths():
    """验证所有关键数据路径"""
    
    print("=" * 60)
    print("DATA PATH VERIFICATION")
    print("=" * 60)
    
    # 定义所有数据路径
    paths = {
        "评估数据 (Evaluation Data)": "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv",
        "论文节点 (Paper Nodes)": "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json",
        "作者知识图谱 (Author KG)": "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/authorkg_remove0/author_knowledge_graph_2024.json",
        "DyGIE++模型 (DyGIE++ Model)": "./dygie_specter2_baseline/dygiepp/pretrained/scierc.tar.gz"
    }
    
    all_good = True
    
    for name, path in paths.items():
        print(f"\n{name}:")
        print(f"  路径: {path}")
        
        if os.path.exists(path):
            print(f"  ✅ 文件存在")
            
            # 获取文件大小
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  📏 大小: {size_mb:.1f} MB")
            
            # 特殊检查
            if path.endswith('.csv'):
                try:
                    df = pd.read_csv(path)
                    print(f"  📊 CSV形状: {df.shape}")
                    print(f"  📋 列名: {list(df.columns)}")
                    
                    # 检查关键列
                    if 'author2' in df.columns:
                        print(f"  ✅ 包含 'author2' 列")
                    elif 'author_old_paper' in df.columns:
                        print(f"  ✅ 包含 'author_old_paper' 列 (可映射为team_authors)")
                    else:
                        print(f"  ⚠️  缺少 'author2' 或 'author_old_paper' 列")
                        all_good = False
                        
                except Exception as e:
                    print(f"  ❌ CSV读取失败: {e}")
                    all_good = False
                    
            elif path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    print(f"  📊 JSON条目数: {len(data)}")
                    
                    # 显示几个键示例
                    if isinstance(data, dict):
                        sample_keys = list(data.keys())[:3]
                        print(f"  🔑 示例键: {sample_keys}")
                        
                except Exception as e:
                    print(f"  ❌ JSON读取失败: {e}")
                    all_good = False
                    
        else:
            print(f"  ❌ 文件不存在")
            all_good = False
    
    print(f"\n{'=' * 60}")
    if all_good:
        print("✅ 所有数据路径验证通过！")
        print("📌 系统已准备好运行Bridger基线评估")
    else:
        print("❌ 部分数据路径存在问题")
        print("⚠️  请检查并修复上述问题后再运行")
    print("=" * 60)
    
    return all_good

def check_evaluation_data_format():
    """详细检查评估数据格式"""
    
    eval_path = "/home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv"
    
    if not os.path.exists(eval_path):
        print(f"❌ 评估数据文件不存在: {eval_path}")
        return False
    
    print(f"\n{'=' * 60}")
    print("EVALUATION DATA FORMAT CHECK")
    print("=" * 60)
    
    try:
        df = pd.read_csv(eval_path)
        print(f"📊 数据形状: {df.shape}")
        print(f"📋 列名: {list(df.columns)}")
        
        # 显示前几行数据结构
        print(f"\n前5行数据预览:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.head())
        
        # 检查必需的列
        required_columns = ['author2', 'author_old_paper']  # Either one is acceptable
        optional_columns = ['ground_truth_authors']
        
        print(f"\n列检查:")
        has_required = False
        for col in required_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                print(f"  ✅ {col}: {non_null_count}/{len(df)} 非空")
                has_required = True
        
        if not has_required:
            print(f"  ❌ 缺少必需列: {required_columns} 中的任意一个")
            return False
        
        for col in optional_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                print(f"  ⚠️  {col}: {non_null_count}/{len(df)} 非空 (可选)")
            else:
                print(f"  ⚠️  缺少可选列: {col}")
        
        # 检查作者列的格式
        author_col = 'author2' if 'author2' in df.columns else 'author_old_paper'
        print(f"\n{author_col}列格式检查:")
        sample_authors = df[author_col].dropna().head(3)
        for i, authors in enumerate(sample_authors):
            print(f"  样本 {i+1}: {authors}")
            try:
                import ast
                parsed = ast.literal_eval(authors)
                print(f"    解析结果: {len(parsed)} 个作者")
            except:
                print(f"    ⚠️  无法解析为Python列表")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取评估数据失败: {e}")
        return False

if __name__ == "__main__":
    # 验证所有路径
    paths_ok = verify_paths()
    
    # 详细检查评估数据格式
    format_ok = check_evaluation_data_format()
    
    if paths_ok and format_ok:
        print(f"\n🎉 所有验证通过！系统可以开始运行。")
        print(f"\n📚 接下来可以运行:")
        print(f"   python bridger_baselines.py")
        print(f"   或")
        print(f"   python dygie_specter2_baseline/scripts/embedding_generator.py --evaluation-data /home/jx4237/CM4AI/LLM-scientific-feedback-main/986_paper_matching_pairs.csv")
    else:
        print(f"\n⚠️  请先解决上述问题。")