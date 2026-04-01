from typing import List, Dict, Any, Optional, Union
import json
import pandas as pd
import random
import yaml
from pathlib import Path

from bfcl_eval.constants.category_mapping import  TEST_COLLECTION_MAPPING
from bfcl_eval.constants.eval_config import PROMPT_PATH, MULTI_TURN_FUNC_DOC_PATH
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.utils import is_multi_turn, parse_test_category_argument, populate_test_cases_with_predefined_functions
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler


TEST_FILE_MAPPING = {'simple': 'BFCL_v4_simple.json', 'irrelevance': 'BFCL_v4_irrelevance.json', 'parallel': 'BFCL_v4_parallel.json', 'multiple': 'BFCL_v4_multiple.json', 'parallel_multiple': 'BFCL_v4_parallel_multiple.json', 'java': 'BFCL_v4_java.json', 'javascript': 'BFCL_v4_javascript.json', 'live_simple': 'BFCL_v4_live_simple.json', 'live_multiple': 'BFCL_v4_live_multiple.json', 'live_parallel': 'BFCL_v4_live_parallel.json', 'live_parallel_multiple': 'BFCL_v4_live_parallel_multiple.json', 'live_irrelevance': 'BFCL_v4_live_irrelevance.json', 'live_relevance': 'BFCL_v4_live_relevance.json', 'multi_turn_base': 'BFCL_v4_multi_turn_base.json', 'multi_turn_miss_func': 'BFCL_v4_multi_turn_miss_func.json', 'multi_turn_miss_param': 'BFCL_v4_multi_turn_miss_param.json', 'multi_turn_long_context': 'BFCL_v4_multi_turn_long_context.json'}

def bfcl_task_preprocess(
    test_categories: Optional[List[str]] = None,
    train_ratio: float = 0.5,
    random_seed: int = 42,
    output_dir: str = "",
    enable_shuffle: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Preprocess training dataset by loading test cases, processing them and splitting into train/test sets.
    """
    
    def load_selected_test_cases(categories: List[str]):
        all_test_entries_by_category = {}
        
        try:
            test_categories_resolved = parse_test_category_argument(categories)
        except Exception as e:
            print(f"Error: Invalid test categories - {e}")
            return {}

        print(f"Selected test categories: {test_categories_resolved}")
        
        for category in test_categories_resolved:
            if category in TEST_FILE_MAPPING:
                test_file_path = TEST_FILE_MAPPING[category]
                test_entries = load_file(PROMPT_PATH / test_file_path)
                print(f"Loaded {len(test_entries)} test cases from {category}")                
                if category not in all_test_entries_by_category:
                    all_test_entries_by_category[category] = []
                all_test_entries_by_category[category].extend(test_entries)
        
        return all_test_entries_by_category

    random.seed(random_seed)
    
    if test_categories is None:
        test_categories = ["all"]
    
    all_test_cases_by_category = load_selected_test_cases(test_categories)
    
    if not all_test_cases_by_category:
        print("Warning: No test cases found")
        return {'train': [], 'test': []}
    
    total_cases = sum(len(cases) for cases in all_test_cases_by_category.values())
    print(f"Loaded {total_cases} test cases in total across {len(all_test_cases_by_category)} categories")

    all_processed_cases = []
    processed_cases_by_category = {}
    
    # 存放最终切分好的数据集
    train_cases = []
    test_cases = []

    # Process and split test cases by category (分层抽样切分)
    for category, test_cases_list in all_test_cases_by_category.items():
        print(f"Processing category: {category}")
        
        # 处理当前类别的数据
        category_processed_cases = populate_test_cases_with_predefined_functions(test_cases_list)
        processed_cases_by_category[category] = category_processed_cases
        all_processed_cases.extend(category_processed_cases)
        
        # --- 关键改动：在这里按类别进行随机打乱和切分 ---
        # 如果启用 shuffle，则单独对该类别打乱，保证随机性
        if enable_shuffle:
            random.shuffle(category_processed_cases)
            
        train_size = int(len(category_processed_cases) * train_ratio)
        
        cat_train = category_processed_cases[:train_size]
        cat_test = category_processed_cases[train_size:]
        
        train_cases.extend(cat_train)
        test_cases.extend(cat_test)
        
        print(f"Successfully processed and split {len(category_processed_cases)} test cases for {category} -> Train: {len(cat_train)}, Test: {len(cat_test)}")
    
    print(f"\nSuccessfully processed {len(all_processed_cases)} test cases in total")
    print(f"Data split complete: {len(train_cases)} training, {len(test_cases)} test cases")

    result = {'train': train_cases, 'test': test_cases}

    # Save combined files
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        test_categories_str = "_".join(test_categories)

        # 保存完整数据集
        full_jsonl_path = output_path / f"{test_categories_str}_processed.jsonl"
        with open(full_jsonl_path, 'w', encoding='utf-8') as f:
            for case in all_processed_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        print(f"Full dataset saved to: {full_jsonl_path}")

        # 保存数据集 ID 信息
        split_ids = {
            'train': [case.get('id', idx) for idx, case in enumerate(train_cases)],
            'val': [case.get('id', idx) for idx, case in enumerate(test_cases)]
        }
        
        split_ids_path = output_path / f"{test_categories_str}_split_ids.json"
        with open(split_ids_path, 'w', encoding='utf-8') as f:
            json.dump(split_ids, f, ensure_ascii=False, indent=2)
        print(f"Split IDs saved to: {split_ids_path}")
    
    return result


if __name__ == "__main__":
    # 为了测试你的需求，我们把 multi_turn 加进去
    test_categories_list=["all","all_scoring","multi_turn","single_turn","live","non_live","non_python","python","multi_turn_base"]

    for test_categories in test_categories_list:
        print(f"\n{'='*20} Running for {test_categories} {'='*20}")
        result = bfcl_task_preprocess(
            test_categories=[test_categories],
            train_ratio=0.5,
            output_dir="./bfcl_data",
            enable_shuffle=True  # 【重要】必须开启这个，才能实现“随机”抽取 50%
        )

        print("-" * 50)
        print("Processing complete!")
        if result["train"]:
            print(f"Training samples: {len(result['train'])}")
        if result["test"]:
            print(f"Test samples: {len(result['test'])}")