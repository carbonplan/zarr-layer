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
import { DATASET_MODULES } from '../lib/constants'
import { useAppStore } from '../lib/store'
import { DatasetControlsProps } from '../datasets/types'
import type {
  QueryGeometry,
  RegionQueryResult,
  RegionValues,
} from '../../src/query/types'

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

const subheadingSx = {
  mt: 3,
  mb: 1,
  fontFamily: 'mono',
  letterSpacing: 'smallcaps',
  textTransform: 'uppercase',
  fontSize: [1, 1, 1, 2],
  color: 'secondary',
}

const clampLat = (lat: number) => Math.max(-90, Math.min(90, lat))

const normalizeLng = (lng: number) => {
  // Wrap longitude to [-180, 180]
  const wrapped = ((((lng + 180) % 360) + 360) % 360) - 180
  return wrapped === -180 ? 180 : wrapped
}

type BoundsLike =
  | {
      toArray: () => [[number, number], [number, number]]
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
    const [[swLng, swLat], [neLng, neLat]] = bounds.toArray()
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
  values: RegionValues | undefined,
  fillValue: number,
): number[] => {
  if (!values) return []
  if (Array.isArray(values)) {
    return values.filter(
      (value): value is number =>
        value !== fillValue &&
        typeof value === 'number' &&
        Number.isFinite(value),
    )
  }

  const results: number[] = []
  for (const entry of Object.values(values)) {
    results.push(...collectNumbers(entry as RegionValues, fillValue))
  }
  return results
}

const collectRangeBandValues = (
  bandValues:
    | Record<string, RegionValues | number | null | undefined>
    | undefined,
  monthStart: number | null,
  monthEnd: number | null,
  fillValue: number,
): number[] => {
  if (!bandValues || monthStart === null || monthEnd === null) return []

  const inRange = (month: number) => month >= monthStart && month <= monthEnd
  const collected: number[] = []

  for (const [bandName, values] of Object.entries(bandValues)) {
    const match = /^month_(\d+)/.exec(bandName)
    if (!match) continue
    const month = Number(match[1])
    if (!Number.isFinite(month) || !inRange(month)) continue

    if (typeof values === 'number') {
      if (values !== fillValue && Number.isFinite(values))
        collected.push(values)
      continue
    }
    if (values === null || values === undefined) continue
    collected.push(...collectNumbers(values as RegionValues, fillValue))
  }

  return collected
}

const getRegionMean = (
  result: RegionQueryResult | null,
  fillValue: number,
  monthStart: number | null,
  monthEnd: number | null,
): number | null => {
  if (!result) return null

  const bandRangeValues = collectRangeBandValues(
    result.bandValues as
      | Record<string, RegionValues | number | null | undefined>
      | undefined,
    monthStart,
    monthEnd,
    fillValue,
  )
  if (bandRangeValues.length > 0) {
    const sum = bandRangeValues.reduce((acc, value) => acc + value, 0)
    return sum / bandRangeValues.length
  }

  const numbers: number[] = []

  if (result.bandValues) {
    for (const values of Object.values(result.bandValues)) {
      numbers.push(...collectNumbers(values as RegionValues, fillValue))
    }
  } else if (result.values) {
    numbers.push(...collectNumbers(result.values, fillValue))
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
      : null,
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
    ;(mapInstance as any).on?.('zoom', updateZoom)
    ;(mapInstance as any).on?.('move', updateZoom)

    return () => {
      ;(mapInstance as any)?.off?.('zoom', updateZoom)
      ;(mapInstance as any)?.off?.('move', updateZoom)
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
  const setMapProvider = useAppStore((state) => state.setMapProvider)
  const setActiveDatasetState = useAppStore(
    (state) => state.setActiveDatasetState,
  )
  const setRegionResult = useAppStore((state) => state.setRegionResult)
  const themedColormap = useThemedColormap(colormap)

  const layerConfig = useMemo(
    () => datasetModule.buildLayerProps({ state: datasetState as any }),
    [datasetModule, datasetState],
  )

  const isCarbonplan4d = datasetModule.id === 'carbonplan_4d'
  const currentBand = (datasetState as any)?.band
  const monthStart = (datasetState as any)?.monthStart ?? null
  const monthEnd = (datasetState as any)?.monthEnd ?? null
  const isRangeBand =
    isCarbonplan4d &&
    (currentBand === 'tavg_range' || currentBand === 'prec_range')

  const pointDisplayValue = useMemo(() => {
    if (isRangeBand) {
      const values = collectRangeBandValues(
        pointResult?.bandValues as
          | Record<string, RegionValues | number | null | undefined>
          | undefined,
        monthStart,
        monthEnd,
        fillValue,
      )
      if (values.length > 0) {
        const sum = values.reduce((acc, value) => acc + value, 0)
        return sum / values.length
      }
    }

    const value = pointResult?.value
    if (typeof value === 'number' && Number.isFinite(value)) return value
    return null
  }, [fillValue, isRangeBand, monthEnd, monthStart, pointResult])

  const regionMean = useMemo(
    () => getRegionMean(regionResult, fillValue, monthStart, monthEnd),
    [regionResult, fillValue, monthStart, monthEnd],
  )

  const handleClimChange = (
    next: (prev: [number, number]) => [number, number],
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
      (value) => clamp(value, lower, upper),
    )
    const hi = hiRaw === lo ? lo + span * 0.001 : hiRaw

    setClim([lo, hi])
  }

  const handleDatasetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setDatasetId(e.target.value)
  }

  const ActiveDatasetControls = datasetModule.Controls as React.FC<
    DatasetControlsProps<any>
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
      const result = await zarrLayer.queryRegion(geometry, layerConfig.selector)
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
        <Select value={datasetId} onChange={handleDatasetChange}>
          {Object.entries(DATASET_MODULES).map(([key, config]) => (
            <option key={key} value={key}>
              {config.info}
            </option>
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
          <Box sx={subheadingSx}>Click</Box>
        </Column>
        <Column start={2} width={3}>
          <Box sx={{ color: 'secondary' }}>
            <Badge>
              {pointDisplayValue !== null
                ? pointDisplayValue.toFixed(2)
                : '---'}
            </Badge>
          </Box>
        </Column>
      </Row>
      <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Viewport</Box>
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
    </Box>
  )
}

export default Controls
