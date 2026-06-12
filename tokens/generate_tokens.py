"""Generate team tokens + admin token and (optionally) push the Modal secret.

    python tokens/generate_tokens.py            # write tokens/tokens.txt
    python tokens/generate_tokens.py --push     # also update the arena-tokens secret

Tokens are the bouncer + identity for the event: each team pastes theirs into the
notebook. tokens.txt is gitignored — it's the handout sheet, keep it private.
"""

import argparse
import secrets
import subprocess
from pathlib import Path

N_TEAMS = 30
ENV = "workshop"


def generate(n: int = N_TEAMS) -> dict[str, str]:
    return {f"team{i:02d}": f"wksp_team{i:02d}_{secrets.token_hex(4)}" for i in range(1, n + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="update the arena-tokens Modal secret")
    ap.add_argument("-n", type=int, default=N_TEAMS)
    args = ap.parse_args()

    teams = generate(args.n)
    admin = f"admin_{secrets.token_hex(8)}"

    out = Path(__file__).parent / "tokens.txt"
    lines = ["# Dense Cell Segmentation Arena — team tokens (KEEP PRIVATE)", ""]
    lines += [f"{team}\t{token}" for team, token in teams.items()]
    lines += ["", f"ADMIN (private board reveal)\t{admin}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(teams)} teams + admin)")

    if args.push:
        all_tokens = ",".join(teams.values())
        subprocess.run(
            [
                "modal", "secret", "create", "arena-tokens",
                f"ARENA_TOKENS={all_tokens}",
                f"ARENA_ADMIN_TOKEN={admin}",
                "-e", ENV, "--force",
            ],
            check=True,
        )
        print(f"pushed arena-tokens secret to env '{ENV}' ({len(teams)} tokens)")
        print("redeploy both apps so the web containers pick up the new tokens.")


if __name__ == "__main__":
    main()
