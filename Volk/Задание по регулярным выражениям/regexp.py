def calculate(data, findall):
    matches = findall(r"([abc])([+-]?=)([abc])?([+-]?\d+)?")
    for v1, s, v2, n in matches:
        n = 0 if n == "" else n
        value = get_active_value(n, v2, data)

        if s == "+=":
            data[v1] += value
        elif s == "-=":
            data[v1] -= value
        else:
            data[v1] = value

    return data


def get_active_value(n, v2, data):
    return int(n) if v2 == "" else data[v2] + int(n)
