
def obj_function(x, none_variable):
    capacity = 80 * x[0]
    demand = 54 * x[0]

    # State limit function
    constraint = capacity - demand

    return [capacity], [demand], [constraint]
