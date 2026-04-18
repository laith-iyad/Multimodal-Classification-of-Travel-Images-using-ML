import re
import unicodedata
import pandas as pd

INPUT_CSV = "Data_Cleaned_v2.csv"
OUTPUT_CSV = "Data_Cleaned_v3.csv"

BOILERPLATE_PATTERNS = [
    r"^\s*a\s+clear\s+image\s+of\b",
    r"^\s*a\s+vibrant\s+image\s+of\b",
    r"^\s*an?\s+image\s+of\b",
    r"^\s*this\s+image\s+shows\b",
    r"^\s*in\s+this\s+image\b",
    r"^\s*this\s+photo\s+shows\b",
    r"^\s*the\s+photo\s+captures\b",
    r"^\s*the\s+photo\s+shows\b",
    r"^\s*this\s+is\s+a\s+photo\s+of\b",
    r"^\s*this\s+photo\s+of\b",
    r"^\s*this\s+is\s+a\s+photo\s+for\b",
    r"^\s*this\s+photo\s+is\s+a\s+photo\s+for\b",
    r"^\s*a\s+night\s+view\s+of\b",
    r"^\s*a\s+bright\s+and\s+clear\s+view\s+of\b",
    r"^\s*a\s+stunning\s+.*view\s+of\b",
    r"^\s*a\s+distant\s+view\s+of\b",
    r"^\s*a\s+classic\s+view\s+of\b",
    r"^\s*a\s+beautiful\s+.*view\s+of\b",
    r"^\s*a\s+scenic\s+.*view\s+of\b",
    r"^\s*a\s+majestic\s+.*view\s+of\b",
    r"^\s*a\s+peaceful\s+.*view\s+of\b",
    r"^\s*a\s+breathtaking\s+.*view\s+of\b",
    r"^\s*a\s+view\s+of\b", 
    r"^\s*an?\s+(?:[a-zA-Z-]+\s+){0,4}(?:view|image|photo|picture|scene|shot)\s+of\b",
]

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def find_description_column(df: pd.DataFrame) -> str:
    if "Description" in df.columns:
        return "Description"
    for c in df.columns:
        if c.strip().lower() == "description":
            return c
    raise KeyError("Could not find a 'Description' column in the CSV.")

def strip_wrapping_quotes(s: str) -> str:
    s2 = s.strip()
    if len(s2) >= 2 and ((s2[0] == '"' and s2[-1] == '"') or (s2[0] == "'" and s2[-1] == "'")):
        return s2[1:-1].strip()
    return s2

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = CONTROL_CHARS_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def remove_boilerplate_prefix(s: str) -> str:
    out = s
    for pat in BOILERPLATE_PATTERNS:
        if re.search(pat, out, flags=re.IGNORECASE):
            out = re.sub(pat, "", out, flags=re.IGNORECASE)
            out = re.sub(r"^\s*[:,-]\s*", "", out)  
            return out.strip()
    return out.strip()

def remove_personal_reflection(s: str) -> str:
    reflection_patterns = [
        r"(?i)(?:[\.\,\-]\s*|\s+and\s+)?I\s+(?:chose|choose|picked|selected|included|would\s+love|want|dream)\s+.*",
        r"(?i)(?:[\.\,\-]\s*|\s+and\s+)?I\s+have\s+been\s+there\s+before.*",
        r"(?i)(?:[\.\,\-]\s*|\s+and\s+)?It\s+represents\s+.*", 
        r"(?i)(?:[\.\,\-]\s*|\s+and\s+)?i\s+choose\s+.*",
    ]
    
    out = s
    for pat in reflection_patterns:
        if "represents" in pat and not re.search(r"I\s+(?:chose|picked)", out, re.IGNORECASE):
             continue
             
        out = re.sub(pat, "", out)
    
    out = re.sub(r"(\s+and)+\s*$", "", out, flags=re.IGNORECASE)
    out = re.sub(r"[\.,\-]+\s*$", "", out)
    
    return out.strip()

def remove_leading_punctuation(s: str) -> str:
    return re.sub(r"^[\s\.\,\-\:\;]+", "", s).strip()

MOJIBAKE_MAPPING = {
    "Ã§": "ç",
    "Ã¶": "ö",
    "Ã©": "é",
    "Ã¼": "ü",
    "Ã¡": "á",
    "Ã\xad": "í",
    "Ã³": "ó",
    "Ã±": "ñ",
    "â€“": "–",
    "â€”": "—",
    "â€˜": "‘",
    "â€™": "’",
    "â€œ": "“",
    "â€\x9d": "”",
    "â€": "”", 
    "â": "",   
}

def fix_mojibake(s: str) -> str:
    out = s
    for bad, good in MOJIBAKE_MAPPING.items():
        out = out.replace(bad, good)
    return out

def impute_description(row) -> str:
    desc = str(row.get("Description", "")).strip()
    if desc.lower() == "nan": 
        desc = ""
        
    if not desc or len(desc) < 10:
        parts = []
        
        mood = str(row.get("Mood/Emotion", "")).strip()
        activity = str(row.get("Activity", "")).strip()
        country = str(row.get("Country", "")).strip()
        
        if mood and mood.lower() != "nan":
            parts.append(mood)
        
        if activity and activity.lower() != "nan":
            parts.append(activity)
            
        scene_desc = " ".join(parts)
        if scene_desc:
            scene_desc += " scene"
        else:
            scene_desc = "A scene"
            
        if country and country.lower() != "nan":
            scene_desc += f" in {country}"
            
        return f"{scene_desc}."
        
    return desc

def clean_description(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    
    if s.lower() == "ar_2":
        return ""
    
    s = fix_mojibake(s)
    
    s = strip_wrapping_quotes(s)
    s = normalize_text(s)
    s = remove_boilerplate_prefix(s)
    s = remove_personal_reflection(s)
    s = remove_leading_punctuation(s) 
    
    if s:
        s = s[0].upper() + s[1:]
        
    return s

def main():
    df = pd.read_csv(INPUT_CSV)
    desc_col = find_description_column(df)

    df[desc_col] = df[desc_col].apply(clean_description)
    
    df[desc_col] = df.apply(impute_description, axis=1)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")
    print(f"Rows kept: {len(df)} (no rows dropped)")
    print(f"Modified column: {desc_col}")

if __name__ == "__main__":
    main()
