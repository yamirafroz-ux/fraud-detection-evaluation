import pandas as pd

from fluxfraud.playback import temporal_loops


def tx(edges):
    return pd.DataFrame(
        [(i, *row) for i, row in enumerate(edges)],
        columns=["transaction_id", "time", "sender", "receiver", "accepted"],
    )


def test_temporal_loop_order_and_settlement():
    frame = tx([(1, 0, 1, 1), (2, 1, 2, 1), (3, 2, 0, 1), (4, 0, 2, 0), (30, 1, 0, 1)])
    assert temporal_loops(frame) == {2: [0, 1, 2, 0]}
    # A static triangle with events in the wrong traversal order is not a temporal loop.
    assert temporal_loops(tx([(1, 1, 2, 1), (2, 0, 1, 1), (3, 2, 0, 1)])) == {}


def test_loops_are_prefix_invariant():
    frame = tx([(1, 0, 1, 1), (2, 1, 0, 1), (3, 0, 2, 1), (4, 2, 1, 1), (5, 1, 0, 1)])
    full = temporal_loops(frame)
    assert temporal_loops(frame.iloc[:3]) == {k: v for k, v in full.items() if k < 3}
