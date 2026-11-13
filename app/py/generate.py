"""Puzzle generator built on the sudoku_cv solver.

Fill an empty grid with random backtracking, then remove digits one at a time while the
puzzle still has exactly one solution. Difficulty is the number of givens.
"""

import random

from sudoku_cv.solver import copy_grid, fewest_candidates_cell

GIVENS = {"easy": 40, "medium": 32, "hard": 26}


def fill(grid, rng):
    best = fewest_candidates_cell(grid)
    if best is None:
        return True
    r, c, cand = best
    rng.shuffle(cand)
    for d in cand:
        grid[r][c] = d
        if fill(grid, rng):
            return True
    grid[r][c] = 0
    return False


def count_solutions(grid, limit=2):
    """Number of solutions, counting no further than `limit`. Changes `grid` only temporarily."""
    best = fewest_candidates_cell(grid)
    if best is None:
        return 1
    r, c, cand = best
    total = 0
    for d in cand:
        grid[r][c] = d
        total += count_solutions(grid, limit - total)
        if total >= limit:
            break
    grid[r][c] = 0
    return total


def generate(difficulty="medium", seed=None):
    rng = random.Random(seed)
    solution = [[0] * 9 for _ in range(9)]
    fill(solution, rng)

    puzzle = copy_grid(solution)
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    givens = 81
    for r, c in cells:
        if givens <= GIVENS[difficulty]:
            break
        digit, puzzle[r][c] = puzzle[r][c], 0
        if count_solutions(puzzle) == 1:
            givens -= 1
        else:
            puzzle[r][c] = digit
    return {"givens": puzzle, "solution": solution}
