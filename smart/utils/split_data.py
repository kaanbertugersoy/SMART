
SPLIT_RATIO = 0.666


def split_data(df, split_ratio=SPLIT_RATIO):
    split = int(len(df) * split_ratio)
    s1 = df.iloc[:split].copy()
    s2 = df.iloc[split:].copy()
    return s1, s2
