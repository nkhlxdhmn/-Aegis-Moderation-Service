import pandas as pd
import re
from config import Config

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^\w\s%₹.-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def analyze_dataset():
    print(f"Loading dataset from {Config.DATA_PATH}...")
    Config.setup_dirs()
    
    try:
        df = pd.read_csv(Config.DATA_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    report = ["# Dataset Analysis Report\n"]
    
    total_rows = len(df)
    report.append(f"- **Total Rows**: {total_rows}")
    
    # Check missing values
    missing = df.isnull().sum()
    report.append(f"- **Missing Values**:\n```text\n{missing.to_string()}\n```")
    
    # Drop missing messages
    df = df.dropna(subset=[Config.TEXT_COLUMN, Config.LABEL_COLUMN])
    report.append(f"- **Rows after dropping missing texts**: {len(df)}")
    
    # Class distribution
    dist = df[Config.LABEL_COLUMN].value_counts()
    report.append(f"## Class Distribution\n```text\n{dist.to_string()}\n```")
    
    # Text length stats
    df['text_len'] = df[Config.TEXT_COLUMN].astype(str).apply(len)
    df['word_count'] = df[Config.TEXT_COLUMN].astype(str).apply(lambda x: len(x.split()))
    
    report.append("## Text Statistics")
    report.append(f"- **Average character length**: {df['text_len'].mean():.2f}")
    report.append(f"- **Max character length**: {df['text_len'].max()}")
    report.append(f"- **Average word count**: {df['word_count'].mean():.2f}")
    report.append(f"- **Max word count**: {df['word_count'].max()}")
    
    # Keyword detection baseline
    promo_kws = Config.PROMO_KEYWORDS
    def has_keyword(text):
        text_lower = str(text).lower()
        return any(kw in text_lower for kw in promo_kws)
    
    df['has_keyword'] = df[Config.TEXT_COLUMN].apply(has_keyword)
    kw_promo = df[(df[Config.LABEL_COLUMN] == 'promotional') & df['has_keyword']].shape[0]
    total_promo = df[df[Config.LABEL_COLUMN] == 'promotional'].shape[0]
    kw_service = df[(df[Config.LABEL_COLUMN] == 'service') & df['has_keyword']].shape[0]
    total_service = df[df[Config.LABEL_COLUMN] == 'service'].shape[0]
    
    if total_promo > 0:
        report.append(f"\n## Keyword Heuristics")
        report.append(f"- **Promotional messages containing keywords**: {kw_promo}/{total_promo} ({kw_promo/total_promo*100:.2f}%)")
        report.append(f"- **Service messages containing keywords**: {kw_service}/{total_service} ({kw_service/total_service*100:.2f}%)")

    # Sample texts
    report.append("\n## Sample Data")
    report.append("### Promotional")
    promo_samples = df[df[Config.LABEL_COLUMN] == 'promotional'][Config.TEXT_COLUMN].head(3)
    for s in promo_samples:
        report.append(f"- {s}")
        
    report.append("\n### Service")
    service_samples = df[df[Config.LABEL_COLUMN] == 'service'][Config.TEXT_COLUMN].head(3)
    for s in service_samples:
        report.append(f"- {s}")

    report_content = "\n".join(report)
    
    report_path = Config.REPORTS_DIR / "dataset_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Dataset analysis saved to {report_path}")

if __name__ == "__main__":
    analyze_dataset()
