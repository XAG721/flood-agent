import type { AgentDialogResponse, EntityType, TwinFocusObjectSummary, TwinObjectMapLayer } from "../types/api";

export type ObjectMentionCandidate = {
  objectId: string;
  name: string;
  entityType?: EntityType | string;
};

const ENTITY_TYPE_KEYWORDS: Partial<Record<EntityType, string[]>> = {
  resident: ["李奶奶", "李阿姨", "老人", "独居老人", "居民", "弱势人群", "转移对象"],
  school: ["学校", "小学", "中学", "学生", "家长", "校门"],
  hospital: ["医院", "急诊", "病人", "医护", "后勤入口"],
  nursing_home: ["养老院", "护理院", "老人院"],
  community: ["社区", "网格", "小区", "片区"],
  metro_station: ["地铁", "地铁口", "换乘", "枢纽"],
  underground_space: ["地下空间", "地下室", "下穿", "地下通道"],
  factory: ["工厂", "厂区", "仓库"],
};

export function buildMentionCandidates(
  focusObjects: TwinFocusObjectSummary[],
  mapLayers: TwinObjectMapLayer[],
): ObjectMentionCandidate[] {
  const candidates = [
    ...focusObjects.map((item) => ({
      objectId: item.object_id,
      name: item.name,
      entityType: item.entity_type,
    })),
    ...mapLayers.map((item) => ({
      objectId: item.object_id,
      name: item.name,
      entityType: item.entity_type,
    })),
  ];

  return Array.from(new Map(candidates.map((item) => [item.objectId, item])).values());
}

export function isRouteGuidanceText(text: string) {
  return /避险|避难|疏散|撤离|转移|路线|路径|绕行|通行|怎么走|去哪里|安全出口|安置点/.test(text);
}

export function resolveMentionedObjectId(
  question: string,
  candidates: ObjectMentionCandidate[],
  response?: AgentDialogResponse,
) {
  const trimmedQuestion = question.trim();
  if (!trimmedQuestion) {
    return null;
  }

  const directMatch = candidates.find(
    (item) => trimmedQuestion.includes(item.name) || trimmedQuestion.includes(item.objectId),
  );
  if (directMatch) {
    return directMatch.objectId;
  }

  if (response && (trimmedQuestion.includes(response.object_name) || trimmedQuestion.includes(response.object_id))) {
    return response.object_id;
  }

  const keywordMatch = candidates.find((item) =>
    (ENTITY_TYPE_KEYWORDS[item.entityType as EntityType] ?? []).some((keyword) => trimmedQuestion.includes(keyword)),
  );
  return keywordMatch?.objectId ?? null;
}
