from numpy import array

from tlp import TLP


inp = array([
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1]
])

inp2 = 1 - array([
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1]
]) 

out = array([0, 1, 1, 0])

tlp = TLP(4, 2, 1, True)

for i in range(5000):
    tlp.back_prop_epoch(inp, out, True)

print(tlp.feed_forward(inp))
print(tlp.feed_forward(inp2))
