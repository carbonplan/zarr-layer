/** Map projections the demo registers with MapLibre's `addProjection`. */
export const CUSTOM_PROJECTION_IDS = [
  'antarctic',
  'arctic',
  'plate carrée',
  'equal earth',
  'conus albers',
] as const
export type CustomProjectionId = (typeof CUSTOM_PROJECTION_IDS)[number]

export function isCustomProjection(
  projection: string
): projection is CustomProjectionId {
  return (CUSTOM_PROJECTION_IDS as readonly string[]).includes(projection)
}
