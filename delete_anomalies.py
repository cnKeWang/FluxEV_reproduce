def change(df, a, l):
    i = 0
    while i < a:
        point = i
        # print(df['value'][i])
        while df['label'][i] == 1:
            i += 1
            if 0 < i - point < 5:
                y1 = df['value'][point - 1]
                y2 = df['value'][i]
                for j in range(point, i):
                    df['value'][j] = (j - i) / (point - 1 - i) * y1 + (j - point + 1) / (i - point + 1) * y2
            if i - point >= 5:
                for j in range(point, i):
                    df['value'][j] = (df['value'][j - l] + df['value'][j + l]) / 2
        i += 1

    return df
