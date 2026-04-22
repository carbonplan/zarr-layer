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
    setPointResult(null)
    setRegionResult(null)
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
    // For range bands or multi-values, show mean of collected values
    const mean = values.reduce((acc, value) => acc + value, 0) / values.length
    return Number.isFinite(mean) ? mean : null
  }, [currentVariable, fillValue, pointResult])

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
      // If in range mode, query only the selected month range
      let querySelector = layerConfig.selector
      if (isRangeBand && monthStart !== null && monthEnd !== null) {
        const monthRange: number[] = []
        for (let m = monthStart; m <= monthEnd; m++) {
          monthRange.push(m)
        }
        // Get the base band (tavg or prec) from current selection
        const baseBand = currentBand === 'tavg_range' ? 'tavg' : 'prec'
        querySelector = { band: baseBand, month: monthRange }
      }

      const result = (await zarrLayer.queryData(geometry, querySelector, {
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
