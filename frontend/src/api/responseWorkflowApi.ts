import { responseDispatchApi } from "../features/response/api/dispatch";
import { responseDocumentApi } from "../features/response/api/documents";
import { responseEventApi } from "../features/response/api/events";
import { responseEvidenceApi } from "../features/response/api/evidence";
import { responseRegistryApi } from "../features/response/api/registry";
import { responseTaskApi } from "../features/response/api/tasks";

export const responseWorkflowApi = {
  ...responseEventApi,
  ...responseRegistryApi,
  ...responseDocumentApi,
  ...responseEvidenceApi,
  ...responseTaskApi,
  ...responseDispatchApi,
};
