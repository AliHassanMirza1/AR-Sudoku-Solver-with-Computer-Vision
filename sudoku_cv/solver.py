def copy_grid(grid):
    return [[int(d) for d in row] for row in grid]


def box_cells(r, c):
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return [(br + i, bc + j) for i in range(3) for j in range(3)]


GROUPS = ([[(r, c) for c in range(9)] for r in range(9)]
          + [[(r, c) for r in range(9)] for c in range(9)]
          + [box_cells(3 * (b // 3), 3 * (b % 3)) for b in range(9)])


def is_valid(grid):
    for i in range(9):
        groups = [
            [grid[i][c] for c in range(9)],
            [grid[r][i] for r in range(9)],
            [grid[r][c] for r, c in box_cells(3 * (i // 3), 3 * (i % 3))],
        ]
        for group in groups:
            digits = [d for d in group if d]
            if len(digits) != len(set(digits)):
                return False
    return True


def candidates(grid, r, c):
    used = set(grid[r]) | {grid[i][c] for i in range(9)}
    used |= {grid[i][j] for i, j in box_cells(r, c)}
    return [d for d in range(1, 10) if d not in used]


def fewest_candidates_cell(grid):
    best = None
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                cand = candidates(grid, r, c)
                if best is None or len(cand) < len(best[2]):
                    best = (r, c, cand)
    return best


def backtrack(grid):
    best = fewest_candidates_cell(grid)
    if best is None:
        return True
    r, c, cand = best
    for d in cand:
        grid[r][c] = d
        if backtrack(grid):
            return True
    grid[r][c] = 0
    return False


def next_cell(grid):
    cands = {(r, c): candidates(grid, r, c) for r in range(9) for c in range(9) if grid[r][c] == 0}
    if not cands:
        return None
    for cell, cand in cands.items():
        if len(cand) == 1:
            return cell
    for group in GROUPS:
        for d in range(1, 10):
            spots = [cell for cell in group if d in cands.get(cell, [])]
            if len(spots) == 1:
                return spots[0]
    return min(cands, key=lambda cell: len(cands[cell]))


def reveal_order(grid, solution):
    grid = copy_grid(grid)
    order = []
    cell = next_cell(grid)
    while cell is not None:
        r, c = cell
        grid[r][c] = solution[r][c]
        order.append([r, c])
        cell = next_cell(grid)
    return order


def solve(givens, user_digits=None):
    givens = copy_grid(givens)
    user_digits = copy_grid(user_digits) if user_digits else [[0] * 9 for _ in range(9)]
    solution = copy_grid(givens)
    if not is_valid(givens) or not backtrack(solution):
        return {"status": "no_solution", "solution": None, "order": [], "wrong": []}

    wrong = []
    grid = copy_grid(givens)
    for r in range(9):
        for c in range(9):
            d = user_digits[r][c]
            if d and not givens[r][c]:
                if d == solution[r][c]:
                    grid[r][c] = d
                else:
                    wrong.append([r, c])
                    grid[r][c] = solution[r][c]

    order = reveal_order(grid, solution)
    return {"status": "solved", "solution": solution, "order": order, "wrong": wrong}
