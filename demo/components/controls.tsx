import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Filter,
  Slider,
  Row,
  Column,
  Colorbar,
  Badge,
  Button,
  Input,
  Select,
} from '@carbonplan/components'
import { useThemedColormap } from '@carbonplan/colormaps'
import { Info, RotatingArrow } from '@carbonplan/icons'
import { Box, Flex, IconButton } from 'theme-ui'
import { SidebarDivider } from '@carbonplan/layouts'
import { useAppStore } from '../lib/store'
import { subheadingSx } from './shared-controls'
import DatasetBrowser from './dataset-browser'
import type {
  QueryGeometry,
  QueryResult,
  QueryDataValues,
} from '@carbonplan/zarr-layer'

const colormaps = [
  'reds',
  'oranges',
  'yellows',
  'greens',
  'teals',
  'blues',
  'purples',
  'pinks',
  'greys',
  'fire',
  'earth',
  'water',
  'heart',
  'wind',
  'warm',
  'cool',
  'pinkgreen',
  'redteal',
  'orangeblue',
  'yellowpurple',
  'redgrey',
  'orangegrey',
  'yellowgrey',
  'greengrey',
  'tealgrey',
  'bluegrey',
  'purplegrey',
  'pinkgrey',
  'rainbow',
  'sinebow',
]

const VIEWPORT_QUERY_MIN_ZOOM = 6

const headingSx = {
  fontFamily: 'heading',
  letterSpacing: 'smallcaps',
  textTransform: 'uppercase',
  fontSize: [2, 2, 3, 3],
}

const clampLat = (lat: number) => Math.max(-90, Math.min(90, lat))

const normalizeLng = (lng: number) => {
  // Wrap longitude to [-180, 180]
  const wrapped = ((((lng + 180) % 360) + 360) % 360) - 180
  return wrapped === -180 ? 180 : wrapped
}

type BoundsLike =
  | {
      toArray: () => [number, number][]
      getWest: () => number
      getEast: () => number
      getSouth?: () => number
      getNorth?: () => number
    }
  | [number, number, number, number]
export const boundsToGeometry = (bounds: BoundsLike): QueryGeometry => {
  let west: number
  let east: number
  let south: number
  let north: number

  if (Array.isArray(bounds)) {
    ;[west, south, east, north] = bounds
  } else {
    const arr = bounds.toArray() as [[number, number], [number, number]]
    const [[swLng, swLat], [neLng, neLat]] = arr
    south = clampLat(Math.min(swLat, neLat))
    north = clampLat(Math.max(swLat, neLat))
    west = normalizeLng(bounds.getWest())
    east = normalizeLng(bounds.getEast())

    if (bounds.getSouth) south = clampLat(bounds.getSouth())
    if (bounds.getNorth) north = clampLat(bounds.getNorth())
  }

  south = clampLat(south)
  north = clampLat(north)
  west = normalizeLng(west)
  east = normalizeLng(east)

  if (east >= west) {
    return {
      type: 'Polygon',
      coordinates: [
        [
          [west, south],
          [west, north],
          [east, north],
          [east, south],
          [west, south],
        ],
      ],
    }
  }

  // Handle antimeridian crossing by splitting into two polygons
  return {
    type: 'MultiPolygon',
    coordinates: [
      [
        [
          [west, south],
          [west, north],
          [180, north],
          [180, south],
          [west, south],
        ],
      ],
      [
        [
          [-180, south],
          [-180, north],
          [east, north],
          [east, south],
          [-180, south],
        ],
      ],
    ],
  }
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max)

const collectNumbers = (
  values: QueryDataValues | undefined,
  fillValue: number,
  depth: number = 0
): number[] => {
  if (!values) return []

  // Prevent infinite recursion
  if (depth > 10) {
    console.warn('collectNumbers: max depth reached')
    return []
  }

  if (Array.isArray(values)) {
    return values.filter(
      (value): value is number =>
        value !== fillValue &&
        typeof value === 'number' &&
        Number.isFinite(value)
    )
  }

  // Only process plain objects
  if (typeof values !== 'object' || values === null) return []

  let results: number[] = []
  for (const entry of Object.values(values)) {
    if (entry === values) continue // Skip circular references
    // Use concat instead of spread to avoid stack overflow with large arrays
    const collected = collectNumbers(
      entry as QueryDataValues,
      fillValue,
      depth + 1
    )
    results = results.concat(collected)
  }
  return results
}

const getRegionMean = (
  result: QueryResult | null,
  fillValue: number
): number | null => {
  if (!result) return null

  let numbers: number[] = []

  for (const [key, value] of Object.entries(result)) {
    // Skip metadata fields
    if (key === 'dimensions' || key === 'coordinates') continue

    // Skip if value is not object or array
    if (!value || typeof value !== 'object') continue

    // This is the variable data - collect all numbers from it
    try {
      // Use concat instead of spread to avoid stack overflow with large arrays
      numbers = numbers.concat(
        collectNumbers(value as QueryDataValues, fillValue, 0)
      )
    } catch (error) {
      console.error('Error collecting numbers from', key, error)
    }
  }

  if (numbers.length === 0) return null
  const sum = numbers.reduce((acc, value) => acc + value, 0)
  return sum / numbers.length
}

interface LineProfile {
  values: number[]
  distance: number[]
  line: number[][]
}

const TRANSECT_LAYER_ID = 'transect-line'

const leafArrays = (values: QueryDataValues): number[][] =>
  Array.isArray(values)
    ? [values]
    : Object.values(values).flatMap((entry) =>
        leafArrays(entry as QueryDataValues)
      )

// A multi-value selector returns one aligned series per label, with NaN where
// a series has no data. The profile shows their mean at each sample.
const getLineProfile = (
  result: QueryResult,
  variable: string,
  line: number[][]
): LineProfile | null => {
  const distance = result.coordinates?.distance as number[] | undefined
  const data = result[variable] as QueryDataValues | undefined
  if (!distance || !data) return null

  const series = leafArrays(data)
  const values = distance.map((_, i) => {
    const samples = series.map((s) => s[i]).filter(Number.isFinite)
    return samples.length > 0
      ? samples.reduce((acc, v) => acc + v, 0) / samples.length
      : NaN
  })
  return values.some(Number.isFinite) ? { values, distance, line } : null
}

const SPARKLINE_WIDTH = 200
const SPARKLINE_HEIGHT = 40

const ProfileSparkline = ({ profile }: { profile: LineProfile }) => {
  const finite = profile.values.filter(Number.isFinite)
  const min = Math.min(...finite)
  const max = Math.max(...finite)
  const length = profile.distance[profile.distance.length - 1] || 1
  const points = profile.values
    .map((value, i) =>
      Number.isFinite(value)
        ? `${(profile.distance[i] / length) * SPARKLINE_WIDTH},${
            SPARKLINE_HEIGHT -
            ((value - min) / (max - min || 1)) * SPARKLINE_HEIGHT
          }`
        : null
    )
    .filter(Boolean)
    .join(' ')

  return (
    <Box sx={{ mt: 2, color: 'secondary', fontSize: 1 }}>
      <Box
        as='svg'
        // @ts-expect-error theme-ui Box does not type svg attributes
        viewBox={`0 0 ${SPARKLINE_WIDTH} ${SPARKLINE_HEIGHT}`}
        preserveAspectRatio='none'
        sx={{ width: '100%', height: SPARKLINE_HEIGHT, overflow: 'visible' }}
      >
        <polyline
          points={points}
          fill='none'
          stroke='currentColor'
          strokeWidth={1.5}
          vectorEffect='non-scaling-stroke'
        />
      </Box>
      <Flex sx={{ justifyContent: 'space-between', mt: 1 }}>
        <Box>
          {min.toFixed(2)} to {max.toFixed(2)}
        </Box>
        <Box>
          {profile.values.length} samples over {(length / 1000).toFixed(0)} km
        </Box>
      </Flex>
    </Box>
  )
}

const Controls = () => {
  const datasetId = useAppStore((state) => state.datasetId)
  const datasetModule = useAppStore((state) => state.getDatasetModule())
  const datasetState = useAppStore((state) => state.datasetState)
  const opacity = useAppStore((state) => state.opacity)
  const clim = useAppStore((state) => state.clim)
  const colormap = useAppStore((state) => state.colormap)
  const globeProjection = useAppStore((state) => state.globeProjection)
  const terrainEnabled = useAppStore((state) => state.terrainEnabled)
  const renderPoles = useAppStore((state) => state.renderPoles)
  const setRenderPoles = useAppStore((state) => state.setRenderPoles)
  const mapProvider = useAppStore((state) => state.mapProvider)
  const pointResult = useAppStore((state) => state.pointResult)
  const regionResult = useAppStore((state) => state.regionResult)
  const mapInstance = useAppStore((state) => state.mapInstance)
  const zarrLayer = useAppStore((state) => state.zarrLayer)
  const fillValue =
    zarrLayer?.fillValue ?? datasetModule.fillValue ?? Number.NaN
  const [zoomLevel, setZoomLevel] = useState<number | null>(() =>
    mapInstance ? mapInstance.getZoom() : null
  )

  useEffect(() => {
    if (!mapInstance) {
      setZoomLevel(null)
      return
    }

    const updateZoom = () => {
      try {
        setZoomLevel(mapInstance.getZoom())
      } catch (error) {
        console.error('Failed to read zoom', error)
      }
    }

    updateZoom()
    mapInstance.on?.('zoom', updateZoom)
    mapInstance.on?.('move', updateZoom)

    return () => {
      mapInstance?.off?.('zoom', updateZoom)
      mapInstance?.off?.('move', updateZoom)
    }
  }, [mapInstance])

  const viewportQueryDisabled =
    !mapInstance ||
    !zarrLayer ||
    zoomLevel === null ||
    zoomLevel <= VIEWPORT_QUERY_MIN_ZOOM

  const setOpacity = useAppStore((state) => state.setOpacity)
  const setClim = useAppStore((state) => state.setClim)
  const setColormap = useAppStore((state) => state.setColormap)
  const setGlobeProjection = useAppStore((state) => state.setGlobeProjection)
  const setTerrainEnabled = useAppStore((state) => state.setTerrainEnabled)
  const setMapProvider = useAppStore((state) => state.setMapProvider)
  const setRegionResult = useAppStore((state) => state.setRegionResult)
  const setPointResult = useAppStore((state) => state.setPointResult)
  const hoverQueryEnabled = useAppStore((state) => state.hoverQueryEnabled)
  const setHoverQueryEnabled = useAppStore(
    (state) => state.setHoverQueryEnabled
  )
  const themedColormap = useThemedColormap(colormap)
  const [queryInFlight, setQueryInFlight] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const [lineProfile, setLineProfile] = useState<LineProfile | null>(null)
  const [lineInFlight, setLineInFlight] = useState(false)
  const lineAbortRef = useRef<AbortController | null>(null)

  const layerConfig = useMemo(
    () => datasetModule.buildLayerProps(datasetState),
    [datasetModule, datasetState]
  )

  const isCarbonplan4d = datasetModule.id === 'carbonplan_4d'
  const currentBand = datasetState['band']
  const monthStart = Number(datasetState['monthStart']) || null
  const monthEnd = Number(datasetState['monthEnd']) || null
  const isRangeBand =
    isCarbonplan4d &&
    (currentBand === 'tavg_range' || currentBand === 'prec_range')

  useEffect(() => {
    // Abort in-flight query and clear results when switching dataset or selector
    abortRef.current?.abort()
    lineAbortRef.current?.abort()
    setPointResult(null)
    setRegionResult(null)
    setLineProfile(null)
  }, [datasetId, datasetState, setPointResult, setRegionResult])

  const currentVariable = useMemo(() => {
    const layerConfig = datasetModule.buildLayerProps(datasetState)
    return layerConfig.variable ?? datasetModule.variable
  }, [datasetModule, datasetState])

  const pointDisplayValue = useMemo(() => {
    if (!pointResult) return null
    const values = collectNumbers(
      pointResult[currentVariable] as QueryDataValues,
      fillValue
    )
    if (values.length === 0) return null

    // Range bands render with month=[1..12] so queryData caches-hit, but the
    // displayed average must only cover the user-selected month range.
    const coordMonths = pointResult.coordinates?.month as number[] | undefined
    const filtered =
      isRangeBand &&
      coordMonths &&
      coordMonths.length === values.length &&
      monthStart !== null &&
      monthEnd !== null
        ? values.filter(
            (_, i) => coordMonths[i] >= monthStart && coordMonths[i] <= monthEnd
          )
        : values

    if (filtered.length === 0) return null
    const mean = filtered.reduce((acc, v) => acc + v, 0) / filtered.length
    return Number.isFinite(mean) ? mean : null
  }, [
    currentVariable,
    fillValue,
    pointResult,
    isRangeBand,
    monthStart,
    monthEnd,
  ])

  const regionMean = useMemo(
    () => getRegionMean(regionResult, fillValue),
    [regionResult, fillValue]
  )

  const [terrainInfo, setTerrainInfo] = useState(false)
  const [renderPolesInfo, setRenderPolesInfo] = useState(false)
  const [climInputs, setClimInputs] = useState<[string, string]>([
    String(clim[0]),
    String(clim[1]),
  ])

  useEffect(() => {
    setClimInputs([String(clim[0]), String(clim[1])])
  }, [clim])

  const commitClimInput = (index: 0 | 1, value?: string) => {
    const val = parseFloat(value ?? climInputs[index])
    if (Number.isFinite(val)) {
      handleClimChange((prev) =>
        index === 0 ? [val, prev[1]] : [prev[0], val]
      )
    } else {
      // Reset to current clim if invalid
      setClimInputs([String(clim[0]), String(clim[1])])
    }
  }

  const handleClimInputChange = (index: 0 | 1, newValue: string) => {
    const newNum = parseFloat(newValue)
    const oldNum = clim[index]
    // Detect arrow click: valid number that differs by ~1 (step)
    const isArrowClick =
      Number.isFinite(newNum) && Math.abs(newNum - oldNum) <= 1.01

    if (isArrowClick) {
      commitClimInput(index, newValue)
    } else {
      setClimInputs(
        index === 0 ? [newValue, climInputs[1]] : [climInputs[0], newValue]
      )
    }
  }

  const handleClimChange = (
    next: (prev: [number, number]) => [number, number]
  ) => {
    const resolved = next(clim)

    if (!Array.isArray(resolved) || resolved.length < 2) return
    const [lo, hi] = resolved
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return

    setClim([lo, hi])
  }

  // In range mode, query only the selected month range
  const viewportSelector = () => {
    if (!isRangeBand || monthStart === null || monthEnd === null) {
      return layerConfig.selector
    }
    const monthRange: number[] = []
    for (let m = monthStart; m <= monthEnd; m++) {
      monthRange.push(m)
    }
    // Get the base band (tavg or prec) from current selection
    const baseBand = currentBand === 'tavg_range' ? 'tavg' : 'prec'
    return { band: baseBand, month: monthRange }
  }

  // Draw the sampled transect on the map for as long as its profile is shown.
  useEffect(() => {
    if (!mapInstance || !lineProfile) return
    const remove = () => {
      try {
        if (mapInstance.getLayer(TRANSECT_LAYER_ID)) {
          mapInstance.removeLayer(TRANSECT_LAYER_ID)
        }
        if (mapInstance.getSource(TRANSECT_LAYER_ID)) {
          mapInstance.removeSource(TRANSECT_LAYER_ID)
        }
      } catch (e) {}
    }
    remove()
    try {
      mapInstance.addSource(TRANSECT_LAYER_ID, {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: { type: 'LineString', coordinates: lineProfile.line },
        },
      })
      mapInstance.addLayer({
        id: TRANSECT_LAYER_ID,
        type: 'line',
        source: TRANSECT_LAYER_ID,
        layout: { 'line-cap': 'round' },
        paint: {
          'line-color': '#ffffff',
          'line-width': 2,
          'line-dasharray': [2, 2],
        },
      })
    } catch (error) {
      console.warn('Could not draw the transect line', error)
    }
    return remove
  }, [mapInstance, lineProfile])

  // A profile belongs to the map it was drawn on.
  useEffect(() => {
    setLineProfile(null)
  }, [mapInstance])

  const handleTransectQuery = async () => {
    if (lineInFlight) return
    if (!mapInstance || !zarrLayer || !mapInstance.getBounds) return

    lineAbortRef.current?.abort()
    const controller = new AbortController()
    lineAbortRef.current = controller

    setLineInFlight(true)
    try {
      const bounds = mapInstance.getBounds()
      const center = mapInstance.getCenter()
      if (!bounds || !center) {
        throw new Error('Transect query is not available')
      }
      // Bounds come back unwrapped, so a view across the antimeridian gives
      // an east past 180 and the query reads it as a crossing.
      let west = bounds.getWest()
      let east = bounds.getEast()
      if (east - west >= 360) {
        west = -180
        east = 180
      }
      const lat = clampLat(center.lat)
      const geometry: QueryGeometry = {
        type: 'LineString',
        coordinates: [
          [west, lat],
          [east, lat],
        ],
      }
      const result = (await zarrLayer.queryData(geometry, viewportSelector(), {
        signal: controller.signal,
        includeSpatialCoordinates: true,
      })) as QueryResult
      setLineProfile(
        getLineProfile(result, currentVariable, geometry.coordinates)
      )
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      console.error('Transect query failed', error)
      setLineProfile(null)
    } finally {
      setLineInFlight(false)
    }
  }

  const handleViewportQuery = async () => {
    if (viewportQueryDisabled || queryInFlight) return
    if (!mapInstance || !zarrLayer || !mapInstance.getBounds) return

    // Abort any previous query
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setQueryInFlight(true)
    try {
      const bounds = mapInstance.getBounds()
      if (!bounds) {
        throw new Error('Viewport query is not available')
      }
      const geometry = boundsToGeometry(bounds)
      const result = (await zarrLayer.queryData(geometry, viewportSelector(), {
        signal: controller.signal,
        includeSpatialCoordinates: false,
      })) as QueryResult
      setRegionResult(result)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      console.error('Viewport query failed', error)
      setRegionResult(null)
    } finally {
      setQueryInFlight(false)
    }
  }

  return (
    <Box>
      <Box sx={headingSx}>Dataset</Box>

      <DatasetBrowser />

      <SidebarDivider sx={{ my: 3 }} />

      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={4}>
          <Box sx={headingSx}>Query</Box>
        </Column>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Point</Box>
        </Column>
        <Column start={2} width={3}>
          <Box sx={{ color: 'secondary' }}>
            <Flex
              sx={{ justifyContent: 'space-between', alignItems: 'baseline' }}
            >
              <Flex sx={{ gap: 2, alignItems: 'center' }}>
                <Badge>
                  {pointDisplayValue !== null
                    ? pointDisplayValue.toFixed(2)
                    : '---'}
                </Badge>
                {pointDisplayValue !== null && (
                  <Box
                    as='span'
                    onClick={() => setPointResult(null)}
                    sx={{
                      cursor: 'pointer',
                      fontSize: 0,
                      color: 'secondary',
                      '&:hover': { color: 'primary' },
                    }}
                  >
                    ✕
                  </Box>
                )}
              </Flex>
              <Flex sx={{ gap: 2 }}>
                <Box sx={{ fontSize: 2, color: 'secondary' }}>Hover</Box>
                <Filter
                  values={{
                    on: hoverQueryEnabled,
                    off: !hoverQueryEnabled,
                  }}
                  setValues={(obj: Record<string, boolean>) => {
                    if (obj.off) setHoverQueryEnabled(false)
                    if (obj.on) setHoverQueryEnabled(true)
                  }}
                />
              </Flex>
            </Flex>
          </Box>
        </Column>
      </Row>
      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Region</Box>
        </Column>
        <Column start={2} width={3}>
          <Flex sx={{ justifyContent: 'space-between', alignItems: 'center' }}>
            <Flex sx={{ alignItems: 'center', gap: 2, color: 'secondary' }}>
              <Badge>
                {regionMean !== null ? regionMean.toFixed(2) : '---'}
              </Badge>
              {regionMean !== null && (
                <Box
                  as='span'
                  onClick={() => setRegionResult(null)}
                  sx={{
                    cursor: 'pointer',
                    fontSize: 0,
                    color: 'secondary',
                    '&:hover': { color: 'primary' },
                  }}
                >
                  ✕
                </Box>
              )}
            </Flex>
            {viewportQueryDisabled ? (
              <Box sx={{ color: 'secondary', fontSize: 2 }}>
                Zoom in to query
              </Box>
            ) : (
              <Button
                onClick={handleViewportQuery}
                suffix={<RotatingArrow />}
                size='xs'
                title='Query viewport'
                disabled={queryInFlight}
                sx={{ fontSize: 2 }}
              >
                {queryInFlight ? 'Querying...' : 'Query viewport average'}
              </Button>
            )}
          </Flex>
        </Column>
      </Row>

      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Line</Box>
        </Column>
        <Column start={2} width={3}>
          <Flex sx={{ justifyContent: 'space-between', alignItems: 'center' }}>
            <Flex sx={{ alignItems: 'center', gap: 2, color: 'secondary' }}>
              <Badge>{lineProfile ? lineProfile.values.length : '---'}</Badge>
              {lineProfile && (
                <Box
                  as='span'
                  onClick={() => setLineProfile(null)}
                  sx={{
                    cursor: 'pointer',
                    fontSize: 0,
                    color: 'secondary',
                    '&:hover': { color: 'primary' },
                  }}
                >
                  ✕
                </Box>
              )}
            </Flex>
            <Button
              onClick={handleTransectQuery}
              suffix={<RotatingArrow />}
              size='xs'
              title='Profile west to east through the map center'
              disabled={lineInFlight || !mapInstance || !zarrLayer}
              sx={{ fontSize: 2 }}
            >
              {lineInFlight ? 'Querying...' : 'Query viewport transect'}
            </Button>
          </Flex>
          {lineProfile && <ProfileSparkline profile={lineProfile} />}
        </Column>
      </Row>

      <SidebarDivider sx={{ my: 3 }} />

      <Row columns={[4, 4, 4, 4]}>
        <Column start={1} width={4}>
          <Box sx={headingSx}>Display</Box>
        </Column>
      </Row>

      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Colormap</Box>
        </Column>
        <Column start={2} width={3}>
          <Select
            value={colormap}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
              setColormap(e.target.value)
            }
            size='xs'
          >
            {colormaps.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </Select>
        </Column>
      </Row>

      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Range</Box>
        </Column>
        <Column start={2} width={3}>
          <Flex sx={{ gap: 2, alignItems: 'center' }}>
            <Input
              size='xs'
              type='number'
              value={climInputs[0]}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                handleClimInputChange(0, e.target.value)
              }
              onBlur={() => commitClimInput(0)}
              onKeyDown={(e: React.KeyboardEvent) => {
                if (e.key === 'Enter') commitClimInput(0)
              }}
              sx={{
                width: `${Math.max(2, climInputs[0].length + 2)}ch`,
              }}
            />
            <Box sx={{ flex: 1 }}>
              <Colorbar width='100%' colormap={themedColormap} horizontal />
            </Box>
            <Input
              size='xs'
              type='number'
              value={climInputs[1]}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                handleClimInputChange(1, e.target.value)
              }
              onBlur={() => commitClimInput(1)}
              onKeyDown={(e: React.KeyboardEvent) => {
                if (e.key === 'Enter') commitClimInput(1)
              }}
              sx={{
                width: `${Math.max(2, climInputs[1].length + 2)}ch`,
              }}
            />
          </Flex>
        </Column>
      </Row>

      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Opacity</Box>
        </Column>

        <Column start={2} width={3}>
          <Flex>
            <Slider
              min={0}
              max={1}
              step={0.01}
              value={opacity}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setOpacity(parseFloat(e.target.value))
              }
            />
          </Flex>
        </Column>
      </Row>

      <SidebarDivider sx={{ my: 3 }} />

      <Box sx={headingSx}>Map</Box>

      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1} sx={subheadingSx}>
          provider
        </Column>
        <Column start={2} width={3}>
          <Filter
            values={{
              maplibre: mapProvider === 'maplibre',
              mapbox: mapProvider === 'mapbox',
            }}
            setValues={(obj: Record<string, boolean>) => {
              if (obj.maplibre) setMapProvider('maplibre')
              if (obj.mapbox) setMapProvider('mapbox')
            }}
          />
        </Column>
      </Row>

      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1} sx={subheadingSx}>
          Projection
        </Column>
        <Column start={2} width={3}>
          <Filter
            values={{ globe: globeProjection, mercator: !globeProjection }}
            setValues={(obj: Record<string, boolean>) => {
              if (obj.mercator) setGlobeProjection(false)
              if (obj.globe) setGlobeProjection(true)
            }}
          />
        </Column>
      </Row>

      {mapProvider === 'mapbox' && (
        <>
          <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
            <Column start={1} width={1} sx={subheadingSx}>
              Terrain
            </Column>
            <Column start={2} width={3}>
              <Flex sx={{ alignItems: 'center', gap: 2 }}>
                <Filter
                  values={{ on: terrainEnabled, off: !terrainEnabled }}
                  setValues={(obj: Record<string, boolean>) => {
                    if (obj.off) setTerrainEnabled(false)
                    if (obj.on) {
                      setTerrainEnabled(true)
                      setRenderPoles(false)
                    }
                  }}
                />
                <IconButton
                  onClick={() => setTerrainInfo(!terrainInfo)}
                  aria-label='More information'
                  aria-expanded={terrainInfo}
                  sx={{
                    cursor: 'pointer',
                    width: '16px',
                    height: '16px',
                    p: 0,
                    flexShrink: 0,
                    '&:hover > #terrain-info': { stroke: 'primary' },
                  }}
                >
                  <Info
                    id='terrain-info'
                    height='16px'
                    width='16px'
                    sx={{
                      stroke: terrainInfo ? 'primary' : 'secondary',
                      transition: '0.1s',
                    }}
                  />
                </IconButton>
              </Flex>
            </Column>
          </Row>
          {terrainInfo && (
            <Box
              sx={{
                fontSize: 2,
                color: 'secondary',
                mt: 1,
                mb: 2,
                fontFamily: 'body',
              }}
            >
              Drapes the zarr layer over Mapbox 3D terrain. Incompatible with
              render poles. Mapbox only.
            </Box>
          )}

          <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
            <Column start={1} width={1} sx={subheadingSx}>
              Render poles
            </Column>
            <Column start={2} width={3}>
              <Flex sx={{ alignItems: 'center', gap: 2 }}>
                <Filter
                  values={{ on: renderPoles, off: !renderPoles }}
                  setValues={(obj: Record<string, boolean>) => {
                    if (obj.off) setRenderPoles(false)
                    if (obj.on) {
                      setRenderPoles(true)
                      setTerrainEnabled(false)
                    }
                  }}
                />
                <IconButton
                  onClick={() => setRenderPolesInfo(!renderPolesInfo)}
                  aria-label='More information'
                  aria-expanded={renderPolesInfo}
                  sx={{
                    cursor: 'pointer',
                    width: '16px',
                    height: '16px',
                    p: 0,
                    flexShrink: 0,
                    '&:hover > #render-poles-info': { stroke: 'primary' },
                  }}
                >
                  <Info
                    id='render-poles-info'
                    height='16px'
                    width='16px'
                    sx={{
                      stroke: renderPolesInfo ? 'primary' : 'secondary',
                      transition: '0.1s',
                    }}
                  />
                </IconButton>
              </Flex>
            </Column>
          </Row>
          {renderPolesInfo && (
            <Box
              sx={{
                fontSize: 2,
                color: 'secondary',
                mt: 1,
                mb: 2,
                fontFamily: 'body',
              }}
            >
              Experimental direct ECEF rendering that avoids the visible polar
              gap from Web Mercator clipping. Incompatible with terrain. Toggle
              impacts Mapbox only: Maplibre renders to the poles by default
            </Box>
          )}
        </>
      )}
    </Box>
  )
}

export default Controls
