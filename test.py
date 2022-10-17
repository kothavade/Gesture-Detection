# test function with optional arguments
def test_func(x, y, z=1, *args, **kwargs):
    print(x, y, z)
    print(args)
    print(kwargs)


test_func(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, a=1, b=2, c=3)

# function with signature: (arg, [arg, [arg, ...]])
def test_func2(x, y, z=1, *args):
    # if args more than 5, then raise error
    if len(args) > 5:
        raise ValueError("Too many arguments")
    print(x, y, z)
    print(args)


# test list
test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x = test_list.index(1, 3, 4, 5, 6)
print(x)
