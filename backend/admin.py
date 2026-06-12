"""Operator admin tasks for the leaderboard.

    modal run -e workshop backend/admin.py::reset   # clear all submissions
    modal run -e workshop backend/admin.py::dump    # print current board state
"""

import modal

app = modal.App("cell-arena-admin")
board_state = modal.Dict.from_name("arena-board", create_if_missing=True)


@app.function()
def reset() -> None:
    n = 0
    for key in list(board_state.keys()):
        del board_state[key]
        n += 1
    print(f"cleared {n} team(s) from the board")


@app.function()
def dump() -> None:
    rows = sorted(board_state.values(), key=lambda r: r.get("public", 0), reverse=True)
    for r in rows:
        print(f"{r['team']:16s} public={r.get('public'):.4f} private={r.get('private'):.4f} subs={r.get('n_subs')}")
    print(f"({len(rows)} teams)")
