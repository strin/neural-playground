def create_idict(dic):
    res = {}
    for pair in dic.items():
        res[pair[1]] = pair[0]
    return res

