"""Affiche le journal d'audit MCP et vérifie son intégrité (chaîne de hachage).

    python scripts/audit_mcp.py                 # 20 dernières lignes + vérification
    python scripts/audit_mcp.py --user tech_s01 --limit 50
    python scripts/audit_mcp.py --request <thread_id>
"""

import argparse
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.server import AUDIT_SCHEMA, _db, verify_audit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user")
    ap.add_argument("--request")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    with closing(_db()) as con:
        con.execute(AUDIT_SCHEMA)
        where, params = [], []
        if args.user:
            where.append("user_id = ?"), params.append(args.user)
        if args.request:
            where.append("request_id = ?"), params.append(args.request)
        sql = "SELECT * FROM audit_mcp" + (" WHERE " + " AND ".join(where) if where else "")
        rows = con.execute(sql + " ORDER BY id DESC LIMIT ?", params + [args.limit]).fetchall()
        for r in reversed(rows):
            extra = r["ticket_id"] or r["motif"] or ""
            print(f"{r['id']:>5}  {r['horodatage']}  {r['outil']:<22} {r['user_id'] or '?':<10} "
                  f"{r['equipment_id'] or '':<16} {r['resultat']:<8} {extra}")
        broken = verify_audit(con)
    print(f"\n{len(rows)} ligne(s) affichée(s). Intégrité : " +
          ("OK" if not broken else f"ROMPUE aux lignes {broken}"))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
