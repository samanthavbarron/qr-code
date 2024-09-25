import numpy as np
from itertools import product
import qrcode
from random import choice
import pandas as pd
from scipy.ndimage import label
from multiprocessing import Pool
from tqdm import tqdm
import random, string
import seaborn as sns
import matplotlib.pyplot as plt

# Args
parallel = False
num_procs = 2
num_extremal = 2

random_case = False
include_slash = True
include_numbers = True

num_random_urls = 10000
num_chars = 5
tld = None
random_tlds = ["io", "ai", "me"]

# Weights
s_contiguous_blocks_w = -1.0
s_neighbors_w = 0.02
s_boundaries_w = -0.0
s_corners_w = -0.01
s_crosses_w = -0.01

shift_adjc = [(0,1),(1,0),(1,1)]
def score_diffs(x: np.ndarray) -> np.float64:
    res = np.zeros(x.shape)

    for shift in shift_adjc:
        r = np.roll(x, shift, axis=(0,1))
        res += np.logical_not(np.logical_xor(x, r))

    return np.sum(res)

def count_corners(x: np.ndarray) -> int:
    res = x
    res += np.roll(x, (0,-1), axis=(0,1))
    res += np.roll(x, (-1,0), axis=(0,1))
    res += np.roll(x, (1,1), axis=(0,1))

    return (res%2 == 1).sum()

def count_crosses(x: np.ndarray) -> int:
    x_ul = x
    x_bl = np.roll(x, (-1,0), axis=(0,1))
    x_ur = np.roll(x, (0,-1), axis=(0,1))
    x_br = np.roll(x, (-1,-1), axis=(0,1))

    res = (x_ul == x_br) & (x_bl == x_ur) & (x_ul != x_bl)

    return res.sum()


neighbors = [
    (+1,+0),
    (+1,+1),
    (+0,+1),
    (-1,+1),
    (-1,+0),
    (-1,-1),
    (+0,-1),
    (+1,-1),
]
def count_same_neighbors(x: np.ndarray) -> int:
    res = np.zeros(x.shape)
    for shift in neighbors:
        res += (x == np.roll(x, shift, axis=(0,1)))
    res = np.exp(res)
    return res.sum()

def score_url(url:str, img_file = None):
    qr = qrcode.QRCode( # type: ignore
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_Q, # type: ignore
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=False)
    mat = np.array(qr.modules, dtype=int)

    if img_file is not None:
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(img_file)

    tld = url.replace("/", "").split(".")[1]

    return {
        "url": url,
        "tld": tld,
        "s_contiguous_blocks": label(mat)[1], # type: ignore
        "s_contiguous_blocks_w": s_contiguous_blocks_w,
        "s_neighbors": count_same_neighbors(mat),
        "s_neighbors_w": s_neighbors_w,
        "s_boundaries": score_diffs(mat),
        "s_boundaries_w": s_boundaries_w,
        "s_corners": count_corners(mat),
        "s_corners_w": s_corners_w,
        "s_crosses": count_crosses(mat),
        "s_crosses_w": s_crosses_w
    }

def random_url(seed:int, num_chars:int=4, tld=None, include_slash:bool=True, random_case: bool = True, include_numbers: bool = True) -> str:
    rng = np.random.default_rng(seed=seed)
    random.seed(seed)
    if tld is None:
        tld = rng.choice(random_tlds)

    valid_chars = string.ascii_letters
    if include_numbers:
        valid_chars = valid_chars + string.digits

    domain = ''.join(random.choices(valid_chars, k=num_chars))
    domain = domain.lower()

    url = f"{domain}.{tld}"

    if random_case:
        url = ''.join(choice((str.upper, str.lower))(char) for char in url)

    if include_slash:
        url = url + "/"

    return url

def _main(seed:int):
    return score_url(random_url(seed, tld=tld, num_chars=num_chars, include_slash=include_slash, random_case=random_case, include_numbers=include_numbers))

if __name__ == "__main__":

    if parallel:
        with Pool(processes=num_procs) as P:
            _df_data = P.map(_main, tqdm(range(num_random_urls)))
    else:
        _df_data = list(map(_main, tqdm(range(num_random_urls))))

    df = pd.DataFrame(_df_data)

    metrics = []
    for metric in df.columns:
        if metric[:2] != "s_":
            continue
        if metric[-2:] == "_w":
            continue
        metrics.append(metric)
        df[metric] -= df[metric].mean()
        df[metric] /= df[metric].std()
        df[metric] *= df[metric + "_w"]
        df.drop(columns=[metric + "_w"], inplace=True)

    df["score"] = 0.0
    for metric in metrics:
        df["score"] += df[metric]

    print(df.describe())

    ax = sns.histplot(
        data=df,
        x="score",
        hue="tld",
    )
    ax.set_xlabel("Score")
    ax.set_ylabel("Counts")

    df.sort_values(by="score", inplace=True, ascending=False)

    def print_score(i, name):
        url = df.iloc[i]["url"]
        url_stripped = url.replace("/","")
        s = df.iloc[i]["score"]
        score_url(url, img_file=f"images/{name}_{abs(i)}_{url_stripped}.png")
        print(f"\tURL: {url}\t Score: {s}")

    print("Best Scores")
    for i in range(num_extremal):
        print_score(i, "max")
    print("Worst Scores")
    for i in range(-num_extremal,0):
        print_score(i, "min")
    
    plt.savefig(f"sample_dist.png")