import os
import json
from google import genai
import argparse
from tqdm import tqdm
import logging
import time
import random
from dotenv import load_dotenv
import re

# .envファイルから環境変数を読み込む
load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("med_synth.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalSynthesizer:
    def __init__(self, api_key=None, model="gemini-2.0-flash", use_thinking=False, batch_size=500, delay=0.5):
        """医療情報合成クラスの初期化"""
        # APIキーが指定されていない場合は環境変数から取得
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("APIキーが指定されていません。--api_keyオプションまたは.envファイルのGEMINI_API_KEYを設定してください。")
        
        if use_thinking:
            self.client = genai.Client(api_key=api_key, http_options={'api_version':'v1alpha'})
            self.model = "gemini-2.0-flash-thinking-exp"
        else:
            self.client = genai.Client(api_key=api_key)
            self.model = model
            
        self.output_dir = "synthesized_data"
        os.makedirs(self.output_dir, exist_ok=True)
        self.batch_size = batch_size
        self.delay = delay
        self.use_thinking = use_thinking
        
    def generate_content_for_topic(self, topic, category=None):
        """特定の医療トピックに関する詳細情報を生成"""
        prompt = self._create_prompt(topic, category)
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"トピック '{topic}' の生成中にエラーが発生しました: {str(e)}")
            return None
    
    def _determine_category(self, topic):
        """トピックのカテゴリを自動判定"""
        # 概念・倫理関連
        if any(keyword in topic for keyword in ["倫理", "プロフェッショナリズム", "宣言", "権利", "義務", "自律", "尊重", "正義", "善行", "無危害"]):
            return "概念"
        
        # 所見・診察関連
        if any(keyword in topic for keyword in ["呼吸音", "雑音", "聴診", "所見", "徴候", "反射", "触診", "打診"]):
            return "所見"
            
        # 症状関連
        if any(keyword in topic for keyword in ["症状", "痛", "熱", "咳", "吐", "下痢", "麻痺", "しびれ", "浮腫", "発疹"]):
            return "症状"
            
        # 疾患関連（上記に該当しないものは基本的に疾患と判断）
        if any(keyword in topic for keyword in ["症", "病", "炎", "癌", "腫瘍", "障害", "不全"]):
            return "疾患"
            
        # デフォルトは一般カテゴリ
        return "一般"
    
    def _create_prompt(self, topic, category=None):
        """医療トピックに対するプロンプトを作成"""
        # カテゴリが指定されていない場合は自動判定
        if category is None:
            category = self._determine_category(topic)
            
        # カテゴリに応じたプロンプトを返す
        if category == "概念":
            return self._create_concept_prompt(topic)
        elif category == "所見":
            return self._create_finding_prompt(topic)
        elif category == "症状":
            return self._create_symptom_prompt(topic)
        elif category == "疾患":
            return self._create_disease_prompt(topic)
        else:
            return self._create_general_prompt(topic)
    
    def _create_general_prompt(self, topic):
        """一般的な医療トピックに対するプロンプト"""
        return f"""
あなたは日本の医師国家試験対策のための高品質な医療情報を提供する専門家です。
以下のトピックについて、医師国家試験の観点から詳細な解説を日本語で作成してください:

トピック: {topic}

以下の構造で情報を提供してください:
1. 基本的な定義と概念
2. 疫学・病因（日本特有の統計や特徴があれば含める）
3. 病態生理
4. 臨床症状と診断（日本の診断基準や特徴的な所見）
5. 検査所見と画像診断の特徴
6. 日本での標準的治療アプローチ（日本のガイドラインに基づく）
7. 予後と合併症
8. 医師国家試験での出題ポイントと頻出事項
9. 関連する日本の医療制度や保険制度の特記事項（該当する場合）
10. 参考文献や信頼できる日本の情報源

回答は医学的に正確で、最新の日本の医療ガイドラインに準拠し、医師国家試験の受験者が理解しやすい形式で提供してください。
"""

    def _create_concept_prompt(self, topic):
        """概念・倫理関連トピックに対するプロンプト"""
        return f"""
あなたは日本の医師国家試験対策のための高品質な医療情報を提供する専門家です。
以下の医療概念・倫理に関するトピックについて、医師国家試験の観点から詳細な解説を日本語で作成してください:

トピック: {topic}

以下の構造で情報を提供してください:
1. 基本的な定義と概念
2. 日本の医療における歴史的背景と発展
3. 日本医師会の関連する指針や綱領（該当する場合）
4. 具体的な適用場面と事例（日本の医療現場の例を含む）
5. 関連する法律と規制（日本特有の法的枠組み）
6. 国際比較（日本と海外の相違点、該当する場合）
7. 医師国家試験での出題ポイントと頻出事項
8. 臨床現場での実践的アプローチ
9. 最近の動向と将来の展望
10. 参考となる日本の文献や資料

回答は医学的・倫理的に正確で、最新の日本の医療指針に準拠し、医師国家試験の受験者が理解しやすい形式で提供してください。
"""

    def _create_finding_prompt(self, topic):
        """所見・診察関連トピックに対するプロンプト"""
        return f"""
あなたは日本の医師国家試験対策のための高品質な医療情報を提供する専門家です。
以下の診察所見に関するトピックについて、医師国家試験の観点から詳細な解説を日本語で作成してください:

トピック: {topic}

以下の構造で情報を提供してください:
1. 基本的な定義と特徴
2. 正常と異常の鑑別ポイント
3. 発生機序と病態生理学的意義
4. 関連する疾患と臨床的意義
5. 適切な診察・評価方法と注意点
6. 類似所見との鑑別
7. 医師国家試験での出題ポイントと頻出事項
8. 実際の診察での注意点（日本の臨床現場での実践的アドバイス）
9. 参考となる日本の教科書や資料

回答は医学的に正確で、最新の日本の医療教育に準拠し、医師国家試験の受験者が理解しやすい形式で提供してください。
特に、日本の医学教育で重視される診察技術の観点から解説し、国家試験で問われる典型的な所見について詳述してください。
"""

    def _create_symptom_prompt(self, topic):
        """症状関連トピックに対するプロンプト"""
        return f"""
あなたは日本の医師国家試験対策のための高品質な医療情報を提供する専門家です。
以下の症状に関するトピックについて、医師国家試験の観点から詳細な解説を日本語で作成してください:

トピック: {topic}

以下の構造で情報を提供してください:
1. 基本的な定義と症状の特徴
2. 発生機序と病態生理
3. 鑑別診断（重要度順、日本での頻度を考慮）
4. 問診・診察のポイント
5. 適切な検査計画と解釈
6. 初期対応と治療アプローチ
7. 重症度評価と緊急性の判断
8. 医師国家試験での出題ポイントと頻出事項
9. 日本の医療現場での実践的アプローチ
10. 参考となる日本のガイドラインや資料

回答は医学的に正確で、最新の日本の医療ガイドラインに準拠し、医師国家試験の受験者が理解しやすい形式で提供してください。
特に、プライマリケアの観点から、この症状に対する系統的なアプローチを詳述してください。
"""

    def _create_disease_prompt(self, topic):
        """疾患関連トピックに対するプロンプト"""
        return f"""
あなたは日本の医師国家試験対策のための高品質な医療情報を提供する専門家です。
以下の疾患に関するトピックについて、医師国家試験の観点から詳細な解説を日本語で作成してください:

トピック: {topic}

以下の構造で情報を提供してください:
1. 基本的な定義と疾患概念
2. 疫学（日本における特徴的な統計データを含む）
3. 病因・病態生理
4. 臨床症状と経過
5. 診断基準と検査所見（日本の診断基準を含む）
6. 画像所見の特徴
7. 日本のガイドラインに基づく治療アプローチ
8. 予後と合併症
9. 医師国家試験での出題ポイントと頻出事項
10. 参考となる日本の文献やガイドライン

回答は医学的に正確で、最新の日本の医療ガイドラインに準拠し、医師国家試験の受験者が理解しやすい形式で提供してください。
特に、日本の医療事情を反映した内容を含め、国家試験で問われる重要ポイントを強調してください。
"""
    
    def process_topic_list(self, topic_file, start_index=0, end_index=None):
        """トピックリストファイルを処理して各トピックの情報を生成"""
        try:
            with open(topic_file, 'r', encoding='utf-8') as f:
                topics = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # 開始・終了インデックスの処理
            if end_index is None or end_index > len(topics):
                end_index = len(topics)
            
            topics = topics[start_index:end_index]
            logger.info(f"処理するトピック数: {len(topics)}")
            
            # カテゴリごとのトピック数をカウント
            category_counts = {}
            for topic in topics:
                category = self._determine_category(topic)
                category_counts[category] = category_counts.get(category, 0) + 1
            
            logger.info(f"カテゴリ別トピック数: {category_counts}")
            
            results = {}
            # バッチ処理
            for i in range(0, len(topics), self.batch_size):
                batch = topics[i:i+self.batch_size]
                logger.info(f"バッチ {i//self.batch_size + 1}/{(len(topics)-1)//self.batch_size + 1} の処理を開始します")
                
                for topic in tqdm(batch, desc=f"バッチ {i//self.batch_size + 1} 処理中"):
                    category = self._determine_category(topic)
                    logger.info(f"トピック '{topic}' (カテゴリ: {category}) の処理を開始します")
                    content = self.generate_content_for_topic(topic, category)
                    if content:
                        results[topic] = {
                            "content": content,
                            "category": category
                        }
                        # 個別のトピックファイルも保存
                        self._save_topic_to_file(topic, content, category)
                        
                    # APIレート制限を避けるための遅延
                    time.sleep(self.delay + random.uniform(0, 1))
                
                # バッチごとに結果を保存（途中経過のバックアップ）
                self._save_batch_results(results, i//self.batch_size + 1)
                
                # バッチ間の遅延
                if i + self.batch_size < len(topics):
                    logger.info(f"次のバッチまで {self.delay * 2} 秒待機します")
                    time.sleep(self.delay * 2)
                    
            # すべての結果をまとめて保存
            self._save_all_results(results)
            return results
            
        except Exception as e:
            logger.error(f"トピックリスト処理中にエラーが発生しました: {str(e)}")
            return {}
    
    def _save_topic_to_file(self, topic, content, category):
        """個別のトピック情報をファイルに保存"""
        # カテゴリごとのディレクトリを作成
        category_dir = os.path.join(self.output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        safe_topic = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in topic)
        filename = os.path.join(category_dir, f"{safe_topic}.md")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {topic}\n\n")
            f.write(content)
        logger.info(f"トピック '{topic}' を {filename} に保存しました")
    
    def _save_batch_results(self, results, batch_num):
        """バッチ結果をJSONファイルに保存"""
        filename = os.path.join(self.output_dir, f"batch_{batch_num}_results.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"バッチ {batch_num} の結果を {filename} に保存しました")
    
    def _save_all_results(self, results):
        """すべての結果をJSONファイルに保存"""
        model_name = "thinking" if self.use_thinking else self.model.replace("-", "_")
        filename = os.path.join(self.output_dir, f"all_topics_{model_name}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"すべての結果を {filename} に保存しました")

def main():
    parser = argparse.ArgumentParser(description="医師国家試験対策のための医療情報合成ツール")
    parser.add_argument("--api_key", help="Google AI API キー（指定しない場合は.envファイルから読み込み）")
    parser.add_argument("--topic_file", required=True, help="処理するトピックのリストを含むファイルパス")
    parser.add_argument("--model", default="gemini-2.0-flash", help="使用するGeminiモデル")
    parser.add_argument("--use_thinking", action="store_true", help="Thinking モデルを使用する")
    parser.add_argument("--batch_size", type=int, default=100, help="一度に処理するトピック数")
    parser.add_argument("--delay", type=float, default=0.5, help="APIリクエスト間の遅延（秒）")
    parser.add_argument("--start_index", type=int, default=0, help="処理を開始するトピックのインデックス")
    parser.add_argument("--end_index", type=int, default=None, help="処理を終了するトピックのインデックス")
    
    args = parser.parse_args()
    
    synthesizer = MedicalSynthesizer(
        api_key=args.api_key, 
        model=args.model,
        use_thinking=args.use_thinking,
        batch_size=args.batch_size,
        delay=args.delay
    )
    synthesizer.process_topic_list(args.topic_file, args.start_index, args.end_index)

if __name__ == "__main__":
    main() 