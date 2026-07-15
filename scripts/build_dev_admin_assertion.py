from __future__ import annotations

import argparse
import json
import os
import secrets

from flood_system.identity import TrustedIdentityVerifier
from flood_system.response_workflow.models import OperatorRole


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create one short-lived signed development admin assertion for a specific request."
    )
    parser.add_argument("--method", default="GET")
    parser.add_argument("--path", default="/response/configuration/feature-flags")
    parser.add_argument("--operator-id", default="development-admin")
    args = parser.parse_args()
    environment = os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower()
    if environment in {"production", "prod"}:
        raise SystemExit("development assertion generator is disabled in production")
    secret = os.getenv(
        "FLOOD_TRUSTED_IDENTITY_SECRET",
        "local-development-identity-secret-change-before-production-v1",
    )
    headers = TrustedIdentityVerifier.build_headers(
        secret=secret,
        operator_id=args.operator_id,
        operator_role=OperatorRole.ADMIN,
        assurance_level="aal2",
        terminal_id="development-admin-cli",
        nonce=secrets.token_urlsafe(24),
        method=args.method,
        path=args.path,
    )
    print(json.dumps(headers, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
