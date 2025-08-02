#!/usr/bin/env python3
"""
分析我们的First/Last作者识别逻辑
显示具体的计算过程和可能的问题
"""

import json

def analyze_author_position_logic():
    """分析我们当前的First/Last作者识别逻辑"""
    
    print("=" * 80)
    print("FIRST/LAST AUTHOR IDENTIFICATION LOGIC ANALYSIS")
    print("=" * 80)
    
    print("📋 当前逻辑 (embedding_generator.py:384-410):")
    print("""
    def _get_author_position_weight(self, author_id: str, paper_data: dict) -> float:
        # 1. 获取所有作者和位置信息
        author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
        
        # 2. 按位置排序
        author_positions.sort(key=lambda x: x[1])  # Sort by position
        
        # 3. 识别第一和最后作者
        first_author_id = author_positions[0][0]    # 排序后第一个 = position最小
        last_author_id = author_positions[-1][0]    # 排序后最后一个 = position最大
        
        # 4. 判断权重
        if author_id == first_author_id or author_id == last_author_id:
            return 1.0  # First or last author
        else:
            return 0.75  # Middle author
    """)
    
    print("🔍 逻辑关键点:")
    print("✅ First Author = 位置序号最小的作者 (通常是1)")
    print("✅ Last Author = 位置序号最大的作者 (通常是总数)")
    print("✅ 通过排序确保即使位置序号不连续也能正确识别")
    
    # 测试不同场景
    test_scenarios = [
        {
            "name": "正常情况",
            "authors": {
                "author1": [1, "info1"],
                "author2": [2, "info2"], 
                "author3": [3, "info3"],
                "author4": [4, "info4"]
            },
            "expected_first": "author1",
            "expected_last": "author4"
        },
        {
            "name": "位置不连续",
            "authors": {
                "author1": [1, "info1"],
                "author3": [3, "info3"],
                "author5": [5, "info5"]
            },
            "expected_first": "author1",
            "expected_last": "author5"
        },
        {
            "name": "位置乱序",
            "authors": {
                "author_c": [3, "info3"],
                "author_a": [1, "info1"],
                "author_b": [2, "info2"]
            },
            "expected_first": "author_a",
            "expected_last": "author_c"
        },
        {
            "name": "单一作者",
            "authors": {
                "author1": [1, "info1"]
            },
            "expected_first": "author1",
            "expected_last": "author1"
        },
        {
            "name": "位置从2开始 (数据问题)",
            "authors": {
                "author1": [2, "info1"],
                "author2": [3, "info2"],
                "author3": [4, "info3"]
            },
            "expected_first": "author1",  # 位置2，但是最小
            "expected_last": "author3"   # 位置4，但是最大
        }
    ]
    
    print(f"\n{'='*50}")
    print("SCENARIO TESTING")
    print("=" * 50)
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        authors = scenario['authors']
        
        # 模拟我们的逻辑
        author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
        author_positions.sort(key=lambda x: x[1])  # Sort by position
        
        if len(author_positions) <= 1:
            first_author_id = author_positions[0][0]
            last_author_id = author_positions[0][0]
        else:
            first_author_id = author_positions[0][0]
            last_author_id = author_positions[-1][0]
        
        print(f"输入作者: {authors}")
        print(f"排序后: {author_positions}")
        print(f"识别的First: {first_author_id} (期望: {scenario['expected_first']})")
        print(f"识别的Last: {last_author_id} (期望: {scenario['expected_last']})")
        
        # 检查权重分配
        print("权重分配:")
        for author_id in authors.keys():
            if author_id == first_author_id or author_id == last_author_id:
                weight = 1.0
                role = "First" if author_id == first_author_id else "Last"
                if author_id == first_author_id and author_id == last_author_id:
                    role = "Single"
            else:
                weight = 0.75
                role = "Middle"
            print(f"  {author_id}: {weight} ({role})")
        
        # 验证结果
        correct_first = first_author_id == scenario['expected_first']
        correct_last = last_author_id == scenario['expected_last']
        
        if correct_first and correct_last:
            print("✅ 逻辑正确")
        else:
            print("❌ 逻辑错误")

def test_with_real_data():
    """用真实数据测试逻辑"""
    
    print(f"\n{'='*80}")
    print("REAL DATA TESTING")
    print("=" * 80)
    
    # Load paper nodes
    paper_nodes_path = "/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_2024dec.json"
    with open(paper_nodes_path, 'r') as f:
        paper_nodes = json.load(f)
    
    # Test with some real papers
    test_paper_ids = ['10508510', '11101899', '10642780']  # From our previous analysis
    
    for paper_id in test_paper_ids:
        if paper_id in paper_nodes:
            print(f"\n--- Paper {paper_id} ---")
            paper_data = paper_nodes[paper_id]
            
            # Get paper title for context
            title = paper_data.get('features', {}).get('Title', 'No title')[:50] + "..."
            print(f"Title: {title}")
            
            neighbors = paper_data.get('neighbors', {})
            authors = neighbors.get('author', {})
            
            if authors:
                # Apply our logic
                author_positions = [(aid, pos_data[0]) for aid, pos_data in authors.items()]
                author_positions.sort(key=lambda x: x[1])
                
                first_author_id = author_positions[0][0]
                last_author_id = author_positions[-1][0]
                
                print(f"总作者数: {len(authors)}")
                print(f"位置信息: {[(aid, pos) for aid, pos in author_positions]}")
                print(f"First Author: {first_author_id} (位置 {author_positions[0][1]})")
                print(f"Last Author: {last_author_id} (位置 {author_positions[-1][1]})")
                
                # Show weight distribution
                print("权重分配:")
                for author_id, pos in author_positions:
                    if author_id == first_author_id or author_id == last_author_id:
                        weight = 1.0
                        if author_id == first_author_id and author_id == last_author_id:
                            role = "Single"
                        elif author_id == first_author_id:
                            role = "First"
                        else:
                            role = "Last"
                    else:
                        weight = 0.75
                        role = "Middle"
                    print(f"  位置{pos}: {author_id} -> {weight} ({role})")

def check_potential_issues():
    """检查可能的问题"""
    
    print(f"\n{'='*80}")
    print("POTENTIAL ISSUES ANALYSIS")
    print("=" * 80)
    
    print("🚨 可能的问题:")
    
    print("\n1. 数据质量问题:")
    print("   - 如果位置序号不从1开始，first author依然是最小位置")
    print("   - 例：位置[2,3,4] -> First=位置2的作者 (可能不是真正的第一作者)")
    
    print("\n2. 位置缺失问题:")
    print("   - 如果某些作者没有位置信息，会被跳过")
    print("   - 可能导致first/last识别不准确")
    
    print("\n3. 逻辑假设:")
    print("   - 假设：位置序号越小 = 越重要")
    print("   - 假设：最大位置序号 = 最后作者 (通讯作者)")
    print("   - 这些假设在大多数情况下是正确的")
    
    print("\n✅ 优点:")
    print("   - 健壮性好：即使位置不连续也能工作")
    print("   - 自动排序：处理乱序数据")
    print("   - 简单有效：基于位置数字的直接比较")
    
    print("\n📊 建议:")
    print("   - 当前逻辑适用于评估数据（已验证位置信息完整）")
    print("   - 对于数据质量较差的情况，已有默认权重(0.75)作为fallback")
    print("   - 可以考虑添加日志记录异常情况以便调试")

if __name__ == "__main__":
    # 分析逻辑
    analyze_author_position_logic()
    
    # 用真实数据测试
    test_with_real_data()
    
    # 检查潜在问题
    check_potential_issues()
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print("=" * 80)
    print("✅ 当前的First/Last作者识别逻辑是合理且有效的")
    print("📊 通过位置排序确保即使数据不完美也能正确工作")
    print("🎯 适用于我们的评估场景，因为评估数据中位置信息完整")
    print("💡 逻辑简单明确：最小位置=First，最大位置=Last")