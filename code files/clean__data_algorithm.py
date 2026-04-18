import re
import unicodedata
import argparse
from difflib import get_close_matches

import pandas as pd



ACTIVITY_CANON = ["Sightseeing", "Relaxing", "Hiking", "Religious", "Adventure/Sports", "Water Activities"]
MOOD_CANON = ["Excitement", "Happiness", "Curiosity", "Nostalgia", "Adventure", "Romance", "Melancholy"]
SEASON_CANON = ["Spring", "Summer", "Fall", "Winter", "Not Clear"]
TIME_CANON = ["Morning", "Afternoon", "Evening"]
WEATHER_CANON = ["Sunny", "Cloudy", "Rainy", "Snowy", "Not Clear"]

BEACH_COUNTS_AS_WATER = False


def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\t", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip().strip(" .,-;:|/")
    return s.lower()


def pick_close_match(raw: str, choices: list[str], cutoff: float = 0.86) -> str | None:
    if not raw:
        return None
    matches = get_close_matches(raw, choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def choose_best(scores: dict[str, int], tie_order: list[str]) -> str:
    maxv = max(scores.values())
    best = [k for k, v in scores.items() if v == maxv]
    if len(best) == 1:
        return best[0]
    for t in tie_order:
        if t in best:
            return t
    return best[0]


def assert_no_new_missing(original: pd.Series, cleaned: pd.Series, colname: str) -> None:
    created_missing = original.notna() & cleaned.isna()
    n = int(created_missing.sum())
    if n > 0:
        raise AssertionError(f"[{colname}] Created {n} new missing values (not allowed).")


def clean_country(val: object) -> object:
    if pd.isna(val):
        return pd.NA

    raw = str(val)
    s = unicodedata.normalize("NFKC", raw)
    s = s.replace("\t", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" .,-;:|")

    if not s:
        return "Unknown"

    low = s.lower()

    fixes = {
        "eygpt": "Egypt",
        "eygypt": "Egypt",
        "preu": "Peru",
        "suadi arabia": "Saudi Arabia",
        "tã¼rkiye": "Türkiye",
        "turkiye": "Türkiye",
        "türkiye": "Türkiye",
        "united states of america": "United States",
        "united states of america.": "United States",
        "united states": "United States",
        "united states.": "United States",
        "usa": "United States",
        "uk": "United Kingdom",
        "united kingdom": "United Kingdom",
        "the netherlands": "Netherlands",
        "holland": "Netherlands",
        "maldive": "Maldives",
        "bali – indonesia.": "Indonesia",
        "bali": "Indonesia",
        "tokyo/japan": "Japan",
        "italy (lake como)": "Italy",
    }
    if low in fixes:
        return fixes[low]

    city_to_country = {
        "paris": "France",
        "london": "United Kingdom",
        "new york": "United States",
        "new york.": "United States",
        "miami": "United States",
        "california": "United States",
        "california, usa": "United States",
        "jerusalem": "Palestine",
        "mecca": "Saudi Arabia",
        "interlaken": "Switzerland",
        "prague": "Czech Republic",
        "vienna": "Austria",
        "santorini": "Greece",
        "north pole, alaska": "United States",
        "antarctica (continent, not a country)": "Antarctica",
    }
    if low in city_to_country:
        return city_to_country[low]

    if s.upper() == "UAE":
        return "United Arab Emirates"

    return s


def map_weather_row(row: pd.Series) -> object:
    if pd.isna(row["Weather"]):
        return pd.NA

    w_raw = normalize_text(row["Weather"])
    desc = normalize_text(row.get("Description", ""))

    if re.search(r"\b(not clear|no clear|unclear|night lighting|dark|fog\w*|haze\w*|mist\w*)\b", w_raw):
        return "Not Clear"

    if re.search(r"\b(clear)\b", w_raw) and re.search(r"\b(night|nighttime|night sky|dark)\b", w_raw):
        return "Not Clear"

    text = (w_raw + " " + desc).strip()
    scores = {c: 0 for c in WEATHER_CANON}

    def add_score(label: str, pattern: str, w_field: int, w_desc: int):
        if re.search(pattern, w_raw):
            scores[label] += w_field
        if re.search(pattern, desc):
            scores[label] += w_desc

    add_score("Rainy", r"\b(rain\w*|drizzle|storm\w*|shower\w*)\b", 6, 2)
    add_score("Snowy", r"\b(snow\w*|blizzard)\b", 6, 2)
    add_score("Cloudy", r"\b(cloud\w*|overcast|partly cloudy)\b", 5, 2)

    if re.search(r"\b(sunny)\b", w_raw):
        scores["Sunny"] += 6
    if re.search(r"\b(clear)\b", w_raw) and not re.search(r"\b(night|dark)\b", text):
        scores["Sunny"] += 4

    add_score("Not Clear", r"\b(cold|windy|any|unclear)\b", 6, 1)
    if re.search(r"\b(night|nighttime|dark)\b", text):
        scores["Not Clear"] += 2

    if max(scores.values()) == 0:
        if re.search(r"\b(snow)\b", desc):
            return "Snowy"
        if re.search(r"\b(rain|storm)\b", desc):
            return "Rainy"
        if re.search(r"\b(cloud)\b", desc):
            return "Cloudy"
        if re.search(r"\b(sunny|clear)\b", desc) and not re.search(r"\b(night|dark)\b", desc):
            return "Sunny"
        return "Not Clear"

    return choose_best(scores, ["Snowy", "Rainy", "Cloudy", "Sunny", "Not Clear"])


def map_time_row(row: pd.Series) -> object:
    if pd.isna(row["Time of Day"]):
        return pd.NA

    t = normalize_text(row["Time of Day"])
    desc = normalize_text(row.get("Description", ""))
    text = (t + " " + desc).strip()

    if re.search(r"\b(sunrise|morning|dawn)\b", text):
        return "Morning"
    if re.search(r"\b(noon|midday|afternoon|daytime)\b", text):
        return "Afternoon"
    if re.search(r"\b(sunset|evening|dusk|night)\b", text):
        return "Evening"

    m = pick_close_match(t, [c.lower() for c in TIME_CANON], cutoff=0.75)
    if m:
        idx = [c.lower() for c in TIME_CANON].index(m)
        return TIME_CANON[idx]

    return "Afternoon"



def map_season_row(row: pd.Series) -> object:
    if pd.isna(row["Season"]):
        return pd.NA

    s_raw = normalize_text(row["Season"]).replace("autumn", "fall").replace("springl", "spring")
    desc = normalize_text(row.get("Description", ""))
    weather = normalize_text(row.get("Weather_mapped", ""))
    text = (s_raw + " " + desc).strip()

    if re.search(r"\b(winter)\b", text):
        return "Winter"
    if re.search(r"\b(summer)\b", text):
        return "Summer"
    if re.search(r"\b(spring)\b", text):
        return "Spring"
    if re.search(r"\b(fall|autumn)\b", text):
        return "Fall"

    if re.search(r"\b(not clear|unclear|clear)\b", s_raw):
        if weather == "snowy" or re.search(r"\b(snow)\b", desc):
            return "Winter"
        if re.search(r"\b(blossom|bloom|flowers)\b", desc):
            return "Spring"
        if re.search(r"\b(foliage|autumn leaves|fall colors)\b", desc):
            return "Fall"
        if re.search(r"\b(heat|hot|tropical|summer vibes)\b", desc):
            return "Summer"
        return "Not Clear"

    m = pick_close_match(s_raw, [c.lower() for c in SEASON_CANON], cutoff=0.78)
    if m:
        idx = [c.lower() for c in SEASON_CANON].index(m)
        return SEASON_CANON[idx]

    if weather == "snowy":
        return "Winter"

    return "Not Clear"



def map_mood_row(row: pd.Series) -> object:
    if pd.isna(row["Mood/Emotion"]):
        return pd.NA

    mood = normalize_text(row["Mood/Emotion"])
    desc = normalize_text(row.get("Description", ""))

    typo = {
        "advanture": "adventure",
        "curosity": "curiosity",
        "curiousity": "curiosity",
        "curiosty": "curiosity",
        "exitment": "excitement",
        "melanchol": "melancholy",
    }
    if mood in typo:
        mood = typo[mood]

    scores = {c: 0 for c in MOOD_CANON}

    def add(label: str, pattern: str, w_mood: int, w_desc: int):
        if re.search(pattern, mood):
            scores[label] += w_mood
        if re.search(pattern, desc):
            scores[label] += w_desc

    add("Melancholy", r"\b(melanch\w*|sad\w*|gloom\w*)\b", 7, 2)
    add("Nostalgia", r"\b(nostalg\w*)\b", 7, 2)
    add("Romance", r"\b(romance|romantic|love|honeymoon)\b", 7, 2)
    add("Curiosity", r"\b(curios\w*|wonder\w*|awe|amazed|mystery)\b", 6, 2)
    add("Adventure", r"\b(adventure|adventurous|thrill\w*|daring)\b", 6, 2)
    add("Happiness", r"\b(happy|happiness|joy|cheer\w*|smile\w*|peace\w*|calm\w*|seren\w*|relax\w*|tranquil\w*)\b", 5, 2)
    add("Excitement", r"\b(excit\w*|energetic|energy|vibrant|lively|amazing|wow)\b", 6, 2)

    if max(scores.values()) == 0:
        if re.search(r"\b(honeymoon|romantic|love)\b", desc):
            return "Romance"
        if re.search(r"\b(adventure|hike|trek|skydiv|bungee|thrill)\b", desc):
            return "Adventure"
        if re.search(r"\b(ancient|historic|ruins|heritage)\b", desc):
            return "Nostalgia"
        if re.search(r"\b(awe|amazing|majestic|magical)\b", desc):
            return "Excitement"
        return "Happiness"

    return choose_best(scores, ["Melancholy", "Romance", "Adventure", "Excitement", "Curiosity", "Nostalgia", "Happiness"])



GENERIC_ACT_PAT = re.compile(
    r"^(sightseeing|exploring|walking|tourism|visiting|traveling|vacation|vacationing|city exploring|city exploration|urban exploration)$"
)

SPEC_PATS = {
    "Religious": re.compile(r"\b(pray\w*|worship\w*|pilgrimage|mosque|church|temple|cathedral|shrine|spiritual)\b"),
    "Hiking": re.compile(r"\b(hike\w*|trek\w*|trail|trailhead|mountaineer\w*|climb\w*|summit|peak|ridge)\b"),
    "Water": re.compile(r"\b(swim\w*|snorkel\w*|scuba|dive\w*|surf\w*|sail\w*|boat\w*|cruise\w*|canoe\w*|kayak\w*|jet ski\w*|rafting)\b"),
    "Adv": re.compile(r"\b(ski\w*|snowboard\w*|skate\w*|skydiv\w*|bungee|zipline|paraglid\w*|atv|quad|cliff jump\w*|cycling|biking|football|stadium|sports|match|game|safari)\b"),
    "Relax": re.compile(r"\b(relax\w*|meditat\w*|spa|resort|vacation\w*|chill\w*|loung\w*|sunbath\w*)\b"),
}


def map_activity_row(row: pd.Series) -> object:
    if pd.isna(row["Activity"]):
        return pd.NA

    act = normalize_text(row["Activity"])
    desc = normalize_text(row.get("Description", ""))

    has_specific = any(p.search(act) for p in SPEC_PATS.values())
    is_generic = bool(GENERIC_ACT_PAT.match(act)) or (("sightsee" in act or "explor" in act or "walk" in act or "tour" in act) and not has_specific)

    scores = {c: 0 for c in ACTIVITY_CANON}

    def add(label: str, pat_act: str, pat_desc: str, w_act: int, w_desc: int):
        if re.search(pat_act, act):
            scores[label] += w_act
        if re.search(pat_desc, desc):
            scores[label] += w_desc

    add("Religious",
        r"\b(pray\w*|worship\w*|pilgrimage|mosque|church|temple|cathedral|shrine|spiritual)\b",
        r"\b(pray\w*|worship\w*|pilgrimage|mosque|church|temple|cathedral|shrine)\b",
        9, 4)

    add("Hiking",
        r"\b(hike\w*|trek\w*|trail|trailhead|mountaineer\w*|climb\w*|summit|peak|ridge)\b",
        r"\b(hike\w*|trek\w*|trail|mountain|summit|peak|ridge)\b",
        8, 4)

    add("Adventure/Sports",
        r"\b(ski\w*|snowboard\w*|skate\w*|skydiv\w*|bungee|zipline|paraglid\w*|atv|quad|cliff jump\w*|cycling|biking|football|stadium|sports|match|game|safari)\b",
        r"\b(ski\w*|snowboard\w*|skydiv\w*|bungee|zipline|paraglid\w*|stadium|sports|match|game|safari|atv|quad)\b",
        8, 4)

    add("Water Activities",
        r"\b(swim\w*|snorkel\w*|scuba|dive\w*|surf\w*|sail\w*|boat\w*|cruise\w*|canoe\w*|kayak\w*|jet ski\w*|rafting)\b",
        r"\b(swim\w*|snorkel\w*|scuba|dive\w*|surf\w*|sail\w*|boat\w*|cruise\w*|canoe\w*|kayak\w*|jet ski\w*|rafting)\b",
        8, 4)

    if re.search(r"\b(beach|ocean|sea|lake|river|lagoon|waterfall|coast)\b", desc):
        action = bool(re.search(r"\b(swim\w*|snorkel\w*|scuba|dive\w*|surf\w*|sail\w*|boat\w*|kayak\w*|canoe\w*)\b", desc))
        scores["Water Activities"] += 2 if action else 1

    add("Relaxing",
        r"\b(relax\w*|meditat\w*|spa|resort|vacation\w*|chill\w*|loung\w*|sunbath\w*)\b",
        r"\b(relax\w*|meditat\w*|spa|resort|vacation\w*|chill\w*|loung\w*|sunbath\w*|calm|peaceful|tranquil)\b",
        7, 3)

    if re.search(r"\b(beach|seaside|coast)\b", desc) and not re.search(r"\b(swim\w*|snorkel\w*|scuba|dive\w*|surf\w*|sail\w*|boat\w*|kayak\w*)\b", desc):
        if BEACH_COUNTS_AS_WATER:
            scores["Water Activities"] += 2
        else:
            scores["Relaxing"] += 2

    sightseeing_w_act = 1 if is_generic else 5
    add("Sightseeing",
        r"\b(sightsee\w*|tour\w*|explor\w*|visit\w*|walk\w*|stroll\w*|wander\w*|photograph\w*|shop\w*)\b",
        r"\b(sightsee\w*|tour\w*|museum|landmark\w*|heritage|historic|ruins|architecture|market|viewpoint|scenic|photograph\w*|shop\w*|neighborhood)\b",
        sightseeing_w_act, 1)

    if re.search(r"\b(no activity|none)\b", act):
        if scores["Relaxing"] >= scores["Water Activities"] and scores["Relaxing"] > 0:
            return "Relaxing"
        if scores["Water Activities"] > 0:
            return "Water Activities"
        return "Sightseeing"

    if max(scores.values()) == 0:
        if re.search(r"\b(mosque|church|temple|pray\w*|pilgrimage)\b", desc):
            return "Religious"
        if re.search(r"\b(swim\w*|snorkel\w*|scuba|dive\w*|surf\w*|boat\w*|kayak\w*|canoe\w*|ocean|sea|lake|river|beach)\b", desc):
            return "Water Activities"
        if re.search(r"\b(ski\w*|stadium|sports|match|safari|zipline|paraglid\w*|atv|quad)\b", desc):
            return "Adventure/Sports"
        if re.search(r"\b(trail|mountain|hike\w*|trek\w*|summit|peak|ridge)\b", desc):
            return "Hiking"
        if re.search(r"\b(resort|spa|relax\w*|calm|peaceful|tranquil|sunbath)\b", desc):
            return "Relaxing"
        return "Sightseeing"

    return choose_best(scores, ["Religious", "Water Activities", "Adventure/Sports", "Hiking", "Relaxing", "Sightseeing"])


def main(in_path: str, out_path: str) -> None:
    df = pd.read_csv(in_path)

    required_cols = ["Image Path", "Description", "Country", "Weather", "Time of Day", "Season", "Activity", "Mood/Emotion"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}\nFound: {list(df.columns)}")

    orig = {c: df[c].copy() for c in required_cols}

    df["Country"] = df["Country"].apply(clean_country)

    df["Weather"] = df.apply(map_weather_row, axis=1)
    df["Time of Day"] = df.apply(map_time_row, axis=1)

    df["Weather_mapped"] = df["Weather"]
    df["Season"] = df.apply(map_season_row, axis=1)
    df.drop(columns=["Weather_mapped"], inplace=True)

    df["Activity"] = df.apply(map_activity_row, axis=1)
    df["Mood/Emotion"] = df.apply(map_mood_row, axis=1)

    for col in ["Country", "Weather", "Time of Day", "Season", "Activity", "Mood/Emotion"]:
        assert_no_new_missing(orig[col], df[col], col)

    base_cols = ["Image Path", "Description", "Country", "Weather", "Time of Day", "Season", "Activity", "Mood/Emotion"]
    df[base_cols].to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("\n--- Value counts ---")
    for col in ["Weather", "Time of Day", "Season", "Activity", "Mood/Emotion"]:
        print(f"\n{col}:\n{df[col].value_counts(dropna=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="Data.csv", help="Input CSV path (default: Data.csv)")
    parser.add_argument("--out", dest="out_path", default="Data_Cleaned_v2.csv", help="Output CSV path (default: Data_Cleaned_v2.csv)")
    args = parser.parse_args()
    main(args.in_path, args.out_path)
