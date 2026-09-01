def max_nesting_depth(values):
    depth = 1
    for value in values:
        if isinstance(value, list):
            current_depth = max_nesting_depth(value) + 1
            if current_depth > depth:
                depth = current_depth
    return depth


if __name__ == "__main__":
    print(max_nesting_depth([1, 2, 3]))
    print(max_nesting_depth([1, [2, [3]]]))
