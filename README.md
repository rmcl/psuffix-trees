# psuffix-trees

A Python port of the Matlab probabilistic suffix trees and automata library by Markowitz (https://github.com/jmarkow/pst)

This port is an implementation probabilistic suffix trees by Ron, Singer and Tishby 1996.

## How to use

### Simple Example

You'll need a dataset of sequences. There is a nice example in `pypst/fixtures/output_symbols.json` if you want something to play around with.

```
dataset = [
    "VHDEFAZDEFABGNVbEFKJaSAHDHD",
    "BN",
    "CTCQMTJcO",
    ...
]

from pypst import PST

pst = PST(
    L = 2,
    p_min = 0.0073,
    g_min = .01,
    r = 1.6,
    alpha = 17.5,
)

pst.fit(dataset)

pst.tree
```

### Any sequence that is iterable is acceptable

Setup your dataset of sequences
```
dataset = [
    "VHDEFAZDEFABGNVbEFKJaSAHDHD",
    "BN",
    "CTCQMTJcO",
    ...
]
```

Your dataset entries can be any sequence whether its string, list or tuple.

```
dataset = [
    ['1','2','3'],
    ['2','4','7','8']
    ['5','1','2']
]
```
