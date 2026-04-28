import { agentRuntimeApi } from "../api/agentRuntimeApi";
import { adminApi } from "../api/adminApi";
import { datasetApi } from "../api/datasetApi";
import { reliabilityApi } from "../api/reliabilityApi";
import { v2BridgeApi } from "../api/v2BridgeApi";
import { setApiOperatorContext } from "./httpClient";

export { setApiOperatorContext };

// Compatibility facade: existing hooks can keep using api.xxx while new code
// migrates toward domain-specific API modules under src/api.
export const api = {
  ...v2BridgeApi,
  ...agentRuntimeApi,
  ...adminApi,
  ...datasetApi,
  ...reliabilityApi,
};
