import pandas as pd

known_cues = {}


def ask(token):
    print(token)
    print("1 - Sentence ending")
    print("2 - Addressee term")
    print("3 - Honorific marker")
    print("4 - Other")
    n = input()
    print()

    if n == "1":
        return "Sentence ending"
    elif n == "2":
        return "Addressee term"
    elif n == "3":
        return "Honorific marker"
    elif n == "4":
        return "Other"
    
    # Mistyped, ask again
    return ask(token) 


def find_cue_type(token):
    token = token.strip()

    if type(token) != str:
        return "Other"
    elif "냐" in token:
        return "Sentence ending"
    elif "해야" == token:
        return "Other"
    elif "너" == token:
        return "Addressee term"
    elif "어야" == token:
        return "Other"
    elif "지요" in token:
        return "Sentence ending"
    elif "하신" in token:
        return "Sentence ending"
    elif "하시는" in token:
        return "Sentence ending"
    elif "죠" in token:
        return "Sentence ending"
    elif "요" == token:
        return "Sentence ending"
    elif token.endswith("요"):
        return "Sentence ending"
    elif token.endswith("야"):
        return "Sentence ending"
    elif "제가" in token:
        return "Addressee term"
    elif "내가" in token:
        return "Addressee term"
    elif "저는" in token:
        return "Addressee term"
    elif "저희" in token:
        return "Addressee term"
    elif "니가" in token:
        return "Addressee term"
    elif "네가" in token:
        return "Addressee term"
    elif "나는" in token:
        return "Addressee term"
    elif "나를" in token:
        return "Addressee term"
    elif "너는" in token:
        return "Addressee term"
    elif "너를" in token:
        return "Addressee term"
    elif "넌" in token:
        return "Addressee term"
    elif "너희" in token:
        return "Addressee term"
    elif "니들" in token:
        return "Addressee term"
    elif "당신" in token:
        return "Addressee term"
    elif "님" in token:
        return "Honorific marker"
    elif "씨는" in token:
        return "Honorific marker"
    elif "씨가" in token:
        return "Honorific marker"
    elif "셨" in token:
        return "Honorific marker"
    elif "씨" == token:
        return "Other"
    elif "시" == token:
        return "Other"
    elif "니다" == token:
        return "Other"
    elif token.endswith("니다"):
        return "Sentence ending"
    elif "습니까" in token:
        return "Sentence ending"
    elif "시오" in token:
        return "Sentence ending"
    elif "시다" in token:
        return "Sentence ending"
    elif token.endswith("이다"):
        return "Sentence ending"
    elif "어" == token:
        return "Other"
    elif token.endswith("어"):
        return "Sentence ending"
    elif token.endswith("하게"):
        return "Sentence ending"
    elif token.endswith("해"):
        return "Sentence ending"
    elif token.endswith("다"):
        return "Sentence ending"
    elif "얘" == token:
        return "Addressee term"
    elif "쟤" == token:
        return "Addressee term"
    elif "걔" == token:
        return "Addressee term"
    elif "엄마"in token:
        return "Addressee term"
    elif "아빠" in token:
        return "Addressee term"
    elif "언니"in token:
        return "Addressee term"
    elif "오빠" in token:
        return "Addressee term"
    elif "누나" in token:
        return "Addressee term"
    elif "형" == token:
        return "Addressee term"
    elif "우리나라" == token:
        return "Addressee term"
    elif "우리" in token:
        return "Addressee term"
    elif "대리" in token:
        return "Addressee term"
    elif "의사" in token:
        return "Addressee term"
    elif "기사" in token:
        return "Addressee term"
    else:
        if token in known_cues:
            return known_cues[token]
        else:
            cue = ask(token)
            known_cues[token] = cue
            return cue


df50 = pd.read_csv("shap_cue_summary_top50.csv")
for _, row in df50.iterrows():
    token = row["Token"].strip()
    cue = row["Cue Type"]
    if cue == "Other" or cue == "Honorific marker" or cue == "Addressee term" or cue == "Sentence ending":
        known_cues[token] = cue


df = pd.read_csv("shap_cue_summary_top100.csv")

def modify_row(row):
    token = row["Token"]
    row["Cue Type"] = find_cue_type(token)
    return row

df = df.apply(modify_row, axis=1)
print(f"Total: {len(df)}")
print(f"Count missing cue type: {df["Cue Type"].isna().sum()}")

df.to_csv("shap_cue_summary_top100.csv", mode="w+")