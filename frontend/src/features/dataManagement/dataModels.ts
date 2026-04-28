import type { EntityProfile, ResourceStatus } from "../../types/api";

export function createBlankProfile(areaId: string): EntityProfile {
  return {
    entity_id: "",
    area_id: areaId,
    entity_type: "resident",
    name: "",
    village: "",
    location_hint: "",
    resident_count: 0,
    current_occupancy: 0,
    vulnerability_tags: [],
    mobility_constraints: [],
    key_assets: [],
    inventory_summary: "",
    continuity_requirement: "",
    preferred_transport_mode: "walk",
    notification_preferences: [],
    emergency_contacts: [],
    custom_attributes: {},
  };
}

export function createBlankResourceStatus(areaId: string): ResourceStatus {
  return {
    area_id: areaId,
    vehicle_count: 0,
    staff_count: 0,
    supply_kits: 0,
    rescue_boats: 0,
    ambulance_count: 0,
    drone_count: 0,
    portable_pumps: 0,
    power_generators: 0,
    medical_staff_count: 0,
    volunteer_count: 0,
    satellite_phones: 0,
    notes: "",
  };
}
