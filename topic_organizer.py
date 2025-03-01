import os
import json
from collections import defaultdict

def organize_topics():
    """医師国家試験のトピックを構造化して整理する"""
    categories = {
        "基礎医学": ["解剖学", "生理学", "生化学", "薬理学", "病理学", "微生物学", "免疫学"],
        "臨床医学": ["内科", "外科", "小児科", "産婦人科", "精神科", "皮膚科", "眼科", "耳鼻咽喉科", "整形外科", "泌尿器科", "放射線科"],
        "社会医学": ["公衆衛生学", "医療統計", "医療倫理", "医療法規"],
        "症候": ["発熱", "頭痛", "胸痛", "腹痛", "呼吸困難", "意識障害"],
        "検査": ["血液検査", "画像検査", "生理機能検査", "病理検査"]
    }
    
    # カテゴリごとにディレクトリを作成
    base_dir = "topics"
    os.makedirs(base_dir, exist_ok=True)
    
    for category, subcategories in categories.items():
        category_dir = os.path.join(base_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for subcategory in subcategories:
            subcategory_file = os.path.join(category_dir, f"{subcategory}.txt")
            # サブカテゴリファイルが存在しない場合は作成
            if not os.path.exists(subcategory_file):
                with open(subcategory_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {subcategory}のトピックリスト\n\n")
                    f.write("# 以下にトピックを1行ずつ追加してください\n")
    
    print(f"トピック構造を {base_dir} ディレクトリに作成しました")

if __name__ == "__main__":
    organize_topics() 