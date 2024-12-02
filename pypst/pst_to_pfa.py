class Node:
    def __init__(self, label, trans=None, order=None, recurrent=1):
        self.label = label
        self.trans = trans if trans is not None else []
        self.order = order
        self.recurrent = recurrent
        self.arcs = []
        self.arcs_p = []
        self.arcs_states = []

    def __repr__(self):
        return ':'.join(self.label)


def pst_convert_to_pfa(TREE, ALPHABET):
    PFA = []
    counter = 0

    # Traverse TREE
    for i in range(len(TREE)):
        for label_idx, label in enumerate(TREE[i]['label']):
            if TREE[i]['internal'][label_idx]:
                continue

            if label == 'epsilon':
                label = ['epsilon']

            # Create a node for PFA with properties from TREE
            pfa_node = Node(
                label=label,
                trans=list(TREE[i]['g_sigma_s'][label_idx]),  # converting numpy array slice to list
                order=i
            )
            PFA.append(pfa_node)
            counter += 1

    # Node order
    ORDER = [len(pfa_node.label) for pfa_node in PFA]

    # Add additional nodes for suffixes
    i = 0
    while i < len(PFA):
        conns = [idx for idx, val in enumerate(PFA[i].trans) if val > 0]
        currlabel = PFA[i].label or ''

        for j in conns:
            suffix_node_label = currlabel + [ALPHABET[j]]
            suffix_node = get_suffix(suffix_node_label, PFA)

            if suffix_node is None:
                order_edge = max(idx for idx, val in enumerate(ORDER) if val == 1)
                insertion_point = order_edge + 1

                new_pfa_node = Node(
                    label=[ALPHABET[j]],
                    trans=PFA[0].trans.copy(),
                    order=len(suffix_node_label)
                )

                PFA.insert(insertion_point, new_pfa_node)
                ORDER.insert(insertion_point, 1)
                suffix_node = insertion_point

        i += 1

    # Add arcs to PFA nodes
    for i in range(len(PFA)):
        conns = [idx for idx, val in enumerate(PFA[i].trans) if val > 0]
        currlabel = PFA[i].label or ''

        for j in conns:
            suffix_node_label = currlabel + [ALPHABET[j]]

            suffix_node = get_suffix(suffix_node_label, PFA)

            if suffix_node is None:
                raise ValueError('Suffix node not found')

            PFA[i].arcs.append(suffix_node)
            PFA[i].arcs_p.append(PFA[i].trans[j])
            PFA[i].arcs_states.append(ALPHABET[j])

    # Add prefixes that don't exist as states
    exitflag = False
    while not exitflag:
        exitflag = True
        for i in range(len(PFA)):
            if PFA[i].order > 2:
                for j in range(len(PFA[i].label) - 1, 0, -1):
                    currprefix = PFA[i].label[:j]
                    match = next((k for k, node in enumerate(PFA) if node.label == currprefix), None)

                    if match is None:
                        new_node_idx = len(PFA)
                        suffix_node = get_suffix(currprefix, PFA) or 0

                        new_pfa_node = Node(
                            label=currprefix,
                            trans=PFA[suffix_node].trans.copy(),
                            order=len(currprefix)
                        )

                        conns = [idx for idx, val in enumerate(new_pfa_node.trans) if val > 0]
                        for k in conns:
                            suffix_node = get_suffix(currprefix + [ALPHABET[k]], PFA)
                            new_pfa_node.arcs.append(suffix_node)
                            new_pfa_node.arcs_p.append(new_pfa_node.trans[k])
                            new_pfa_node.arcs_states.append(ALPHABET[k])

                        PFA.append(new_pfa_node)
                        exitflag = False

    return PFA

def get_suffix(sequence, PFA):
    # Find the longest suffix of a given sequence
    while sequence:
        for j in range(len(PFA) - 1, -1, -1):
            if PFA[j].label == sequence:
                return j
        sequence = sequence[1:]
    return None
