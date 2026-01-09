import React, { useEffect, useMemo, useState } from 'react'
import {
  Filter,
  Select,
  Slider,
  Row,
  Column,
  Colorbar,
  Badge,
  Button,
  // @ts-expect-error - carbonplan components types not available
} from '@carbonplan/components'
// @ts-expect-error - carbonplan colormaps types not available
import { useThemedColormap } from '@carbonplan/colormaps'
// @ts-expect-error - carbonplan icons types not available
import { RotatingArrow } from '@carbonplan/icons'
import { Box, Divider, Flex } from 'theme-ui'
import { DATASET_MAP } from '../datasets'
import { useAppStore } from '../lib/store'
import type { ControlsProps } from '../datasets/types'
import { SELECTOR_SECTIONS } from '../datasets/sections'
import { subheadingSx } from './shared-controls'
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

const VIEWPORT_QUERY_MIN_ZOOM = 4

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
  const datasetState = useAppStore((state) => state.getDatasetState())
  const opacity = useAppStore((state) => state.opacity)
  const clim = useAppStore((state) => state.clim)
  const colormap = useAppStore((state) => state.colormap)
  const globeProjection = useAppStore((state) => state.globeProjection)
  const terrainEnabled = useAppStore((state) => state.terrainEnabled)
  const mapProvider = useAppStore((state) => state.mapProvider)
  const pointResult = useAppStore((state) => state.pointResult)
  const regionResult = useAppStore((state) => state.regionResult)
  const mapInstance = useAppStore((state) => state.mapInstance)
  const zarrLayer = useAppStore((state) => state.zarrLayer)
  const fillValue =
    zarrLayer?.fillValue ?? datasetModule.fillValue ?? Number.NaN
  const [zoomLevel, setZoomLevel] = useState<number | null>(() =>
    typeof (mapInstance as any)?.getZoom === 'function'
      ? (mapInstance as any).getZoom()
      : null
  )

  useEffect(() => {
    if (!mapInstance || typeof (mapInstance as any)?.getZoom !== 'function') {
      setZoomLevel(null)
      return
    }

    const updateZoom = () => {
      try {
        setZoomLevel((mapInstance as any).getZoom())
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

  const setDatasetId = useAppStore((state) => state.setDatasetId)
  const setOpacity = useAppStore((state) => state.setOpacity)
  const setClim = useAppStore((state) => state.setClim)
  const setColormap = useAppStore((state) => state.setColormap)
  const setGlobeProjection = useAppStore((state) => state.setGlobeProjection)
  const setTerrainEnabled = useAppStore((state) => state.setTerrainEnabled)
  const setMapProvider = useAppStore((state) => state.setMapProvider)
  const setActiveDatasetState = useAppStore(
    (state) => state.setActiveDatasetState
  )
  const setRegionResult = useAppStore((state) => state.setRegionResult)
  const setPointResult = useAppStore((state) => state.setPointResult)
  const themedColormap = useThemedColormap(colormap)

  const layerConfig = useMemo(
    () => datasetModule.buildLayerProps(datasetState as any),
    [datasetModule, datasetState]
  )

  const isCarbonplan4d = datasetModule.id === 'carbonplan_4d'
  const currentBand = (datasetState as any)?.band
  const monthStart = (datasetState as any)?.monthStart ?? null
  const monthEnd = (datasetState as any)?.monthEnd ?? null
  const isRangeBand =
    isCarbonplan4d &&
    (currentBand === 'tavg_range' || currentBand === 'prec_range')

  useEffect(() => {
    // Clear query results when switching dataset or band to avoid stale display
    setPointResult(null)
    setRegionResult(null)
  }, [datasetId, currentBand, setPointResult, setRegionResult])

  const currentVariable = useMemo(() => {
    const layerConfig = datasetModule.buildLayerProps(datasetState as any)
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

  const handleClimChange = (
    next: (prev: [number, number]) => [number, number]
  ) => {
    const base = datasetModule.clim
    const resolved = next(clim)
    if (!Array.isArray(resolved) || resolved.length < 2) return setClim(base)

    const [rawLo, rawHi] = resolved
    if (!Number.isFinite(rawLo) || !Number.isFinite(rawHi)) return setClim(base)

    const span = Math.max(base[1] - base[0], 1)
    const lower = base[0] - span * 0.5
    const upper = base[1] + span * 0.5
    const [lo, hiRaw] = [Math.min(rawLo, rawHi), Math.max(rawLo, rawHi)].map(
      (value) => clamp(value, lower, upper)
    )
    const hi = hiRaw === lo ? lo + span * 0.001 : hiRaw

    setClim([lo, hi])
  }

  const handleDatasetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setDatasetId(e.target.value)
  }

  const ActiveDatasetControls = datasetModule.Controls as React.FC<
    ControlsProps<any>
  >

  const handleViewportQuery = async () => {
    if (viewportQueryDisabled) return
    if (!mapInstance || !zarrLayer || !mapInstance.getBounds) return
    try {
      const bounds = mapInstance.getBounds()
      if (!bounds) {
        throw new Error('Viewport query is not available')
      }
      const geometry = boundsToGeometry(bounds)
      console.log('geometry', geometry)
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
        console.log(
          `Querying range mode: band=${baseBand}, months=${monthRange.join(
            ','
          )}`
        )
      }

      const result = (await zarrLayer.queryData(
        geometry,
        querySelector
      )) as QueryResult
      console.log('Query result:', result)
      setRegionResult(result)
    } catch (error) {
      console.error('Viewport query failed', error)
      setRegionResult(null)
    }
  }

  return (
    <Box>
      <Box sx={headingSx}>Dataset</Box>

      <Box sx={{ width: '100%', my: 2 }}>
        <Select
          value={datasetId}
          onChange={handleDatasetChange}
          size='xs'
          sxSelect={{ width: '100%' }}
        >
          {SELECTOR_SECTIONS.map((section) => (
            <optgroup key={section.label} label={section.label}>
              {section.datasetIds.map((id) => {
                const config = DATASET_MAP[id]
                if (!config) return null
                return (
                  <option key={id} value={id}>
                    {config.info}
                  </option>
                )
              })}
            </optgroup>
          ))}
        </Select>
        <Box sx={{ color: 'secondary', mt: 1 }}>{datasetModule.sourceInfo}</Box>
      </Box>

      <ActiveDatasetControls
        state={datasetState as any}
        setState={setActiveDatasetState as any}
      />

      <Divider sx={{ mt: 4, mb: 3 }} />

      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={4}>
          <Box sx={headingSx}>Query</Box>
        </Column>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Point</Box>
        </Column>
        <Column start={2} width={3}>
          <Box sx={{ color: 'secondary' }}>
            <Flex sx={{ justifyContent: 'space-between' }}>
              <Badge>
                {pointDisplayValue !== null
                  ? pointDisplayValue.toFixed(2)
                  : '---'}
              </Badge>
              <Box>Click map to query</Box>
            </Flex>
          </Box>
        </Column>
      </Row>
      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Region</Box>
        </Column>
        <Column start={2} width={3}>
          <Flex sx={{ justifyContent: 'space-between' }}>
            <Box sx={{ color: 'secondary' }}>
              <Badge>
                {regionMean !== null ? regionMean.toFixed(2) : '---'}
              </Badge>
            </Box>
            {viewportQueryDisabled ? (
              <Box sx={{ color: 'secondary' }}> Zoom in to query</Box>
            ) : (
              <Button
                onClick={handleViewportQuery}
                suffix={<RotatingArrow />}
                size='xs'
                title='Query viewport'
              >
                Query viewport average
              </Button>
            )}
          </Flex>
        </Column>
      </Row>

      <Divider sx={{ mt: 4, mb: 3 }} />

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
          <Colorbar
            width='100%'
            colormap={themedColormap}
            units=''
            clim={clim}
            setClim={handleClimChange}
            horizontal
          />
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

      <Divider sx={{ mt: 4, mb: 3 }} />

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

      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1} sx={subheadingSx}>
          Terrain
        </Column>
        <Column start={2} width={3}>
          {mapProvider === 'mapbox' ? (
            <Filter
              values={{ on: terrainEnabled, off: !terrainEnabled }}
              setValues={(obj: Record<string, boolean>) => {
                if (obj.off) setTerrainEnabled(false)
                if (obj.on) setTerrainEnabled(true)
              }}
            />
          ) : (
            <Box sx={{ color: 'secondary', fontSize: 1 }}>
              Not yet supported in MapLibre
            </Box>
          )}
        </Column>
      </Row>
    </Box>
  )
}

export default Controls
