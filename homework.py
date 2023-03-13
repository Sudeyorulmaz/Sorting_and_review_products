import numpy as np
import pandas as pd
import math
import scipy.stats as st
pd.set_option("display.max_columns",None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
df = pd.read_csv("amazon_review.csv")
df.head(10)

# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

# Adım 1: Ürünün Ortalama Puanını Hesaplayınız.
df["overall"].mean()

# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.

df.loc[df["day_diff"]<= df["day_diff"].quantile(0.25),"overall"].mean()
df.loc[(df["day_diff"]> df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean()
df.loc[(df["day_diff"]> df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean()
df.loc[df["day_diff"]> df["day_diff"].quantile(0.75),"overall"].mean()

def time_based_weighted_average(dataframe,w1=28,w2=26,w3=24,w4=22):
    return  df.loc[df["day_diff"]<= df["day_diff"].quantile(0.25),"overall"].mean() * w1/100 + \
            df.loc[(df["day_diff"]> df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean() * w2/100 + \
            df.loc[(df["day_diff"]> df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() * w3/100 +\
            df.loc[df["day_diff"]> df["day_diff"].quantile(0.75),"overall"].mean() * w4/100

time_based_weighted_average(df)


#Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
# Adım 1. helpful_no Değişkenini Üretiniz.
# Not: total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]
df.head()

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def score_up_down_diff(up,down):
    return  up - down

def score_average_rating(up,down):
    if up+down == 0 :
        return 0
    return up/(up+down)

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x : score_up_down_diff(x["helpful_yes"],x["helpful_no"]),axis=1)
df.sort_values("score_pos_neg_diff",ascending=False).head()

# score_avg_rating
df["score_avg_rating"] = df.apply(lambda x : score_average_rating(x["helpful_yes"],x["helpful_no"]),axis=1)
df.sort_values("score_avg_rating",ascending=False).head()


# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("wilson_lower_bound", ascending=False).head(20)