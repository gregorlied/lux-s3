from .constants import *

def warp(x: int) -> int:
    if x >= SPACE_SIZE:
        x -= SPACE_SIZE
    elif x < 0:
        x += SPACE_SIZE
    return x

def warp_point(x: int, y: int) -> tuple:
    return warp(x), warp(y)

def get_opposite(x: int, y: int) -> tuple:
    # Returns the mirrored point across the diagonal
    return SPACE_SIZE - y - 1, SPACE_SIZE - x - 1

def is_upper_sector(x: int, y: int) -> bool:
    return SPACE_SIZE - x - 1 >= y

def is_lower_sector(x: int, y: int) -> bool:
    return SPACE_SIZE - x - 1 <= y

def is_team_sector(team_id: int, x: int, y: int) -> bool:
    return is_upper_sector(x, y) if team_id == 0 else is_lower_sector(x, y)

def get_match(step: int) -> int:
    return step // (MAX_STEPS_IN_MATCH + 1)

def get_match_step(step: int) -> int:
    return step % (MAX_STEPS_IN_MATCH + 1)

def get_step(step: int) -> int:
    match = get_match(step)
    match_step = get_match_step(step)
    return match_step + match * MAX_STEPS_IN_MATCH

def manhattan_distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_closest_target(start, targets):
    target, min_distance = None, float("inf")
    for t in targets:
        d = manhattan_distance(start, t)
        if d < min_distance:
            target, min_distance = t, d

    return target, min_distance
