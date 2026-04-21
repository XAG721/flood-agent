from __future__ import annotations

from .models import EntityType


HIGH_RISK_TYPES = {
    EntityType.SCHOOL,
    EntityType.FACTORY,
    EntityType.HOSPITAL,
    EntityType.NURSING_HOME,
}

ROUTE_KEYWORDS = (
    "route",
    "road",
    "traffic",
    "shelter",
    "evacu",
    "path",
    "go to",
    "\u8def\u7ebf",
    "\u9053\u8def",
    "\u4ea4\u901a",
    "\u907f\u96be",
    "\u907f\u9669",
    "\u64a4\u79bb",
    "\u53bb\u54ea\u91cc",
)

ACTION_KEYWORDS = (
    "what should",
    "advice",
    "action",
    "recommend",
    "impact",
    "mean for",
    "\u600e\u4e48\u529e",
    "\u5efa\u8bae",
    "\u884c\u52a8",
    "\u5f71\u54cd",
    "\u610f\u5473\u7740\u4ec0\u4e48",
)

PROPOSAL_KEYWORDS = (
    "proposal",
    "approve",
    "queue",
    "advisory",
    "plan",
    "\u5ba1\u6279",
    "\u6279\u590d",
    "\u65b9\u6848",
    "\u9884\u6848",
)

EVIDENCE_KEYWORDS = (
    "why",
    "reason",
    "evidence",
    "grounding",
    "why this",
    "\u4e3a\u4ec0\u4e48",
    "\u4f9d\u636e",
    "\u8bc1\u636e",
)

ENTITY_TYPE_KEYWORDS = {
    EntityType.RESIDENT: ("elder", "elderly", "resident", "\u8001\u4eba", "\u5c45\u6c11", "\u4f4e\u6d3c\u533a\u8001\u4eba"),
    EntityType.SCHOOL: ("school", "primary school", "campus", "\u5b66\u6821", "\u5c0f\u5b66", "\u4e2d\u5b66"),
    EntityType.FACTORY: ("factory", "plant", "inventory", "\u5de5\u5382", "\u5382\u533a", "\u5e93\u5b58"),
    EntityType.HOSPITAL: ("hospital", "clinic", "\u533b\u9662", "\u8bca\u6240"),
    EntityType.NURSING_HOME: ("nursing", "care home", "\u517b\u8001\u9662", "\u62a4\u7406\u9662"),
    EntityType.METRO_STATION: ("metro", "subway", "station", "\u5730\u94c1", "\u5730\u94c1\u53e3"),
    EntityType.UNDERGROUND_SPACE: ("underground", "basement", "mall", "\u5730\u4e0b", "\u5730\u4e0b\u7a7a\u95f4", "\u5730\u4e0b\u5546\u573a"),
    EntityType.COMMUNITY: ("community", "grid", "\u793e\u533a", "\u7f51\u683c"),
}
