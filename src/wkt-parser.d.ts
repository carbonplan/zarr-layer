declare module 'wkt-parser' {
  function parseWkt(wkt: string | Record<string, unknown>): unknown
  export default parseWkt
}
