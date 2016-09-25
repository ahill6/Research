def items(x):
    if isinstance(x, (list, tuple)):
        for y in x:
            for z in items(y):
                yield z
    else:
        yield x