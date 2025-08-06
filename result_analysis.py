import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; SwimBot/1.0; +https://example.com/bot)"
}

def scrape_tournament_results(tournament_id):
    base_url = f"https://result.swim.or.jp/tournament/{tournament_id}"
    res = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')

    data = []
    for tr in soup.select("table tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        record = {
            "name": tds[1].text.strip(),
            "school_year": tds[2].text.strip(),  # 学年
            "team": tds[3].text.strip(),         # 所属
            "event": tds[4].text.strip(),        # 種目
            "time": tds[5].text.strip(),         # タイム
            "rank": tds[6].text.strip(),         # 順位
        }
        data.append(record)

    df = pd.DataFrame(data)
    df.to_csv(f"results_{tournament_id}.csv", index=False)
    print(f"Saved results_{tournament_id}.csv")

# 例：大会IDを使って複数大会取得
tournament_ids = ["4025721", "4025709"]  # 実際の大会IDに置き換えてください
for tid in tournament_ids:
    scrape_tournament_results(tid)
    time.sleep(2)  # サーバー負荷対策の待機


 import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("results_4025721.csv")

# カテゴリ変換
df = df.dropna()
le_event = LabelEncoder()
df['event_id'] = le_event.fit_transform(df['event'])
le_team = LabelEncoder()
df['team_id'] = le_team.fit_transform(df['team'])

# タイム変換（"1:03.12" → 秒数）
def time_to_sec(t):
    try:
        if ":" in t:
            m, s = t.split(":")
            return int(m) * 60 + float(s)
        return float(t)
    except:
        return None

df["time_sec"] = df["time"].apply(time_to_sec)
df = df.dropna(subset=["time_sec"])

# 特徴量と予測対象
X = df[["event_id", "school_year", "team_id"]]
y = df["time_sec"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))   

import matplotlib.pyplot as plt

# 例：複数大会を比較し、「ベスト更新かどうか」を判定
df1 = pd.read_csv("results_4025709.csv")  # 旧大会
df2 = pd.read_csv("results_4025721.csv")  # 新大会

merged = pd.merge(df1, df2, on=["name", "event"], suffixes=("_old", "_new"))
merged["time_old_sec"] = merged["time_old"].apply(time_to_sec)
merged["time_new_sec"] = merged["time_new"].apply(time_to_sec)

merged["improved"] = merged["time_new_sec"] < merged["time_old_sec"]

update_rate = merged.groupby("team_new")["improved"].mean().sort_values(ascending=False).head(10)
update_rate.plot(kind="barh", title="ベスト更新率の高い所属", figsize=(8, 6))
plt.xlabel("更新率")
plt.tight_layout()
plt.show()


# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("データ分析ダッシュボード")

df = pd.read_csv("results_4025721.csv")

selected_event = st.selectbox("種目を選択", df["event"].unique())
filtered = df[df["event"] == selected_event]

fig, ax = plt.subplots()
filtered["time_sec"] = filtered["time"].apply(time_to_sec)
filtered.boxplot(column="time_sec", by="school_year", ax=ax)
st.pyplot(fig)