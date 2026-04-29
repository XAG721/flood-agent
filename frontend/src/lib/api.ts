import { agentRuntimeApi } from "../api/agentRuntimeApi";
import { adminApi } from "../api/adminApi";
import { datasetApi } from "../api/datasetApi";
import { reliabilityApi } from "../api/reliabilityApi";
import { platformBridgeApi } from "../api/platformBridgeApi";
import { setApiOperatorContext } from "./httpClient";

export { setApiOperatorContext };

// Compatibility facade: existing hooks can keep using api.xxx while new code
// migrates toward domain-specific API modules under src/api.
export const api = {
  ...platformBridgeApi,
  ...agentRuntimeApi,
  ...adminApi,
  ...datasetApi,
  ...reliabilityApi,
};
