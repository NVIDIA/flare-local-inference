"""Scalar oracle for the Rust immediate-frontier prefix allocation contract."""

import math
import struct


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def total_order(value):
    bits = struct.unpack("Q", struct.pack("d", value))[0]
    return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits ^ (1 << 63)


def allocate(confidence, slots, submitted, source_valid, limits, budget, capacity):
    batch = len(slots)
    active = [row for row in range(batch) if submitted[row]]
    depth = len(confidence[0])
    valid = (len(active) <= budget and len({slots[row] for row in active}) == len(active)
             and all(source_valid[row] and 0 <= slots[row] < capacity and 1 <= limits[row] <= depth + 1
                     and all(math.isfinite(value) and 0 <= value <= 1 for value in confidence[row])
                     for row in active))
    if not valid:
        return [0] * batch, [False] * batch
    lengths = [int(value) for value in submitted]
    survival = [1.] * batch
    for _ in range(budget - len(active)):
        choices = [(survival[row] * float32(confidence[row][lengths[row] - 1]), row)
                   for row in active if lengths[row] < limits[row]]
        if not choices:
            break
        probability, winner = max(choices, key=lambda item: (total_order(item[0]),
                                       -lengths[item[1]], -slots[item[1]]))
        survival[winner] = probability
        lengths[winner] += 1
    return lengths, list(submitted)
