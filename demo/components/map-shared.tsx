import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Box, Spinner } from 'theme-ui'
// @ts-expect-error - carbonplan colormaps types not available
import { useThemedColormap } from '@carbonplan/colormaps'
import {
  ZarrLayer,
  ZarrLayerOptions,
  QueryGeometry,
  QueryResult,
} from '@carbonplan/zarr-layer'
import maplibregl from 'maplibre-gl'
import mapboxgl from 'mapbox-gl'
import { layers, namedFlavor } from '@protomaps/basemaps'
import { Protocol } from 'pmtiles'
import { useAppStore } from '../lib/store'
import type { LayerProps } from '../datasets/types'

export type MapProvider = 'maplibre' | 'mapbox'

// Minimal interface for map methods we use. Using a union of maplibregl.Map | mapboxgl.Map
// doesn't work because their overloaded on()/off() signatures are incompatible in a union.
export interface MapInstance {
  on(event: string, handler: (e: any) => void): unknown
  off(event: string, handler: (e: any) => void): unknown
  remove(): void
  getLayer(id: string): unknown
  removeLayer(id: string): void
  addLayer(layer: ZarrLayer, beforeId?: string): unknown
  setProjection(projection: any): unknown
  resize(): void
  getBounds(): { toArray(): [number, number][]; getWest(): number; getEast(): number }
  getZoom(): number
  flyTo(options: { center: [number, number]; zoom: number }): void
  getStyle(): { layers?: Array<{ id: string; type: string }> }
  addSource(id: string, source: any): void
  setTerrain(terrain: any): void
}

const backgroundColor = '#1b1e23'
const mapLibreTheme = {
  ...namedFlavor('black'),
  background: backgroundColor,
  earth: backgroundColor,
  park_a: backgroundColor,
  park_b: backgroundColor,
  golf_course: backgroundColor,
  aerodrome: backgroundColor,
  industrial: backgroundColor,
  university: backgroundColor,
  school: backgroundColor,
  zoo: backgroundColor,
  farmland: backgroundColor,
  wood_a: backgroundColor,
  wood_b: backgroundColor,
  residential: backgroundColor,
  protected_area: backgroundColor,
  scrub_a: backgroundColor,
  scrub_b: backgroundColor,
  landcover: {
    barren: backgroundColor,
    farmland: backgroundColor,
    forest: backgroundColor,
    glacier: backgroundColor,
    grassland: backgroundColor,
    scrub: backgroundColor,
    urban_area: backgroundColor,
  },
  regular: 'Relative Pro Book',
  bold: 'Relative Pro Book',
  italic: 'Relative Pro Book',
}

export interface MapConfig {
  createMap: (
    container: HTMLDivElement,
    globeProjection: boolean,
  ) => MapInstance
  setProjection: (map: MapInstance, globeProjection: boolean) => void
  getLayerBeforeId: (map: MapInstance) => string | undefined
}

const mapLibreConfig: MapConfig = {
  createMap: (container: HTMLDivElement, globeProjection: boolean) => {
    const protocol = new Protocol()
    maplibregl.addProtocol('pmtiles', protocol.tile)

    return new maplibregl.Map({
      container,
      style: {
        projection: globeProjection ? { type: 'globe' } : { type: 'mercator' },
        version: 8,
        glyphs:
          'https://carbonplan-maps.s3.us-west-2.amazonaws.com/basemaps/fonts/{fontstack}/{range}.pbf',
        sources: {
          protomaps: {
            type: 'vector',
            url: 'pmtiles://https://carbonplan-maps.s3.us-west-2.amazonaws.com/basemaps/pmtiles/global.pmtiles',
            attribution:
              '<a href="https://overturemaps.org/">Overture Maps</a>, <a href="https://protomaps.com">Protomaps</a>, © <a href="https://openstreetmap.org">OpenStreetMap</a>',
          },
        },
        layers: layers('protomaps', mapLibreTheme, { lang: 'en' }),
      },
      center: [0, 20],
      zoom: 2.4,
    }) as MapInstance
  },
  setProjection: (map: MapInstance, globeProjection: boolean) => {
    ;(map as maplibregl.Map).setProjection(
      globeProjection ? { type: 'globe' } : { type: 'mercator' },
    )
  },
  getLayerBeforeId: () => 'landuse_pedestrian',
}

const mapboxConfig: MapConfig = {
  createMap: (container: HTMLDivElement, globeProjection: boolean) => {
    if (process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN) {
      mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN
    }

    const map = new mapboxgl.Map({
      container,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [0, 20],
      zoom: 2,
      projection: globeProjection ? 'globe' : 'mercator',
    })

    map.on('load', () => {
      map.addSource('mapbox-dem', {
        type: 'raster-dem',
        url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
        tileSize: 512,
        maxzoom: 14,
      })
    })

    return map as MapInstance
  },
  setProjection: (map: MapInstance, globeProjection: boolean) => {
    ;(map as mapboxgl.Map).setProjection({
      name: globeProjection ? 'globe' : 'mercator',
    } as mapboxgl.ProjectionSpecification)
  },
  getLayerBeforeId: (map: MapInstance) => {
    const styleLayers = (map as mapboxgl.Map).getStyle().layers
    return styleLayers?.find((layer) => layer.type === 'symbol')?.id
  },
}

export const getMapConfig = (provider: MapProvider): MapConfig => {
  return provider === 'mapbox' ? mapboxConfig : mapLibreConfig
}

export const useMapLayer = (map: MapInstance | null, isMapLoaded: boolean) => {
  const zarrLayerRef = useRef<InstanceType<typeof ZarrLayer> | null>(null)
  const prevDatasetIdRef = useRef<string | null>(null)
  const datasetId = useAppStore((state) => state.datasetId)
  const datasetModule = useAppStore((state) => state.getDatasetModule())
  const datasetState = useAppStore((state) => state.getDatasetState())
  const opacity = useAppStore((state) => state.opacity)
  const clim = useAppStore((state) => state.clim)
  const colormap = useAppStore((state) => state.colormap)
  const mapProvider = useAppStore((state) => state.mapProvider)
  const setLoadingState = useAppStore((state) => state.setLoadingState)
  const colormapArray = useThemedColormap(colormap, { format: 'hex' })
  const setPointResult = useAppStore((state) => state.setPointResult)
  const setZarrLayer = useAppStore((state) => state.setZarrLayer)

  const layerConfig: LayerProps = useMemo(
    () => datasetModule.buildLayerProps(datasetState as any),
    [datasetModule, datasetState],
  )

  const isCarbonplan4d = datasetModule.id === 'carbonplan_4d'
  const currentBand = (datasetState as any)?.band
  const monthStart = (datasetState as any)?.monthStart ?? null
  const monthEnd = (datasetState as any)?.monthEnd ?? null
  const isRangeBand =
    isCarbonplan4d &&
    (currentBand === 'tavg_range' || currentBand === 'prec_range')

  const latestLayerConfigRef = useRef(layerConfig)
  const latestRangeStateRef = useRef({
    isRangeBand,
    monthStart,
    monthEnd,
    currentBand,
  })

  useEffect(() => {
    latestLayerConfigRef.current = layerConfig
  }, [layerConfig])

  useEffect(() => {
    latestRangeStateRef.current = {
      isRangeBand,
      monthStart,
      monthEnd,
      currentBand,
    }
  }, [isRangeBand, monthStart, monthEnd, currentBand])

  useEffect(() => {
    if (!map || !isMapLoaded) return

    const mapConfig = getMapConfig(mapProvider)
    let clickHandler: ((event: any) => void) | null = null

    if (zarrLayerRef.current) {
      try {
        if (map.getLayer('zarr-layer')) {
          map.removeLayer('zarr-layer')
        }
      } catch (e) {}
      zarrLayerRef.current = null
    }

    const currentLayerConfig = latestLayerConfigRef.current

    const options: ZarrLayerOptions = {
      id: 'zarr-layer',
      source: datasetModule.source,
      variable: currentLayerConfig.variable ?? datasetModule.variable,
      clim: clim,
      colormap: colormapArray,
      opacity: opacity,
      selector: currentLayerConfig.selector,
      zarrVersion: datasetModule.zarrVersion,
      fillValue: datasetModule.fillValue,
      spatialDimensions: datasetModule.spatialDimensions,
      bounds: datasetModule.bounds,
      latIsAscending: datasetModule.latIsAscending,
      proj4: datasetModule.proj4,
      onLoadingStateChange: setLoadingState,
    }

    if (currentLayerConfig.customFrag) {
      options.customFrag = currentLayerConfig.customFrag
    }
    if (currentLayerConfig.uniforms) {
      options.uniforms = currentLayerConfig.uniforms
    }

    try {
      const layer = new ZarrLayer(options)
      let beforeId: string | undefined
      try {
        beforeId = mapConfig.getLayerBeforeId(map)
      } catch (e) {}
      map.addLayer(layer, beforeId)
      console.log('zarr-layer', layer)
      clickHandler = (event: any) => {
        const lng = event.lngLat.lng
        const lat = event.lngLat.lat
        const geometry: QueryGeometry = {
          type: 'Point',
          coordinates: [lng, lat],
        }
        const {
          isRangeBand: rangeMode,
          monthStart: latestMonthStart,
          monthEnd: latestMonthEnd,
          currentBand: latestBand,
        } = latestRangeStateRef.current
        const latestSelector = latestLayerConfigRef.current.selector

        let querySelector = latestSelector
        if (rangeMode && latestMonthStart !== null && latestMonthEnd !== null) {
          const monthRange: number[] = []
          for (let m = latestMonthStart; m <= latestMonthEnd; m++) {
            monthRange.push(m)
          }
          const baseBand = latestBand === 'tavg_range' ? 'tavg' : 'prec'
          querySelector = { band: baseBand, month: monthRange }
        }

        layer.queryData(geometry, querySelector).then((result) => {
          console.log('queryData result', result)
          setPointResult(result as QueryResult)
        })
      }
      map.on('click', clickHandler)
      zarrLayerRef.current = layer
      setZarrLayer(layer)

      // Only fly to dataset center when dataset changes (not on variable/band change)
      if (datasetModule.center && prevDatasetIdRef.current !== datasetId) {
        map.flyTo({
          center: datasetModule.center,
          zoom: datasetModule.zoom || 4,
        })
      }
      prevDatasetIdRef.current = datasetId
    } catch (error) {
      console.error('Error creating ZarrLayer:', error)
    }

    return () => {
      if (zarrLayerRef.current) {
        try {
          if (map.getLayer('zarr-layer')) {
            map.removeLayer('zarr-layer')
          }
        } catch (e) {}
        setZarrLayer(null)
        zarrLayerRef.current = null
      }
      if (clickHandler && map.off) {
        try {
          map.off('click', clickHandler)
        } catch (e) {}
      }
    }
    // colormap changes are handled via the update effect to avoid full layer
    // recreation, so we intentionally omit it from deps here.
  }, [
    map,
    isMapLoaded,
    datasetId,
    datasetModule,
    layerConfig.customFrag,
    layerConfig.variable,
    mapProvider,
    setLoadingState,
  ])

  useEffect(() => {
    const layer = zarrLayerRef.current
    if (!layer || !map || !isMapLoaded) return

    layer.setOpacity(opacity)
    layer.setColormap(colormapArray)
    layer.setClim(clim)

    layer.setSelector(layerConfig.selector)

    if (layerConfig.uniforms && Object.keys(layerConfig.uniforms).length > 0) {
      layer.setUniforms(layerConfig.uniforms)
    }
  }, [opacity, clim, colormapArray, layerConfig, map, isMapLoaded])

  return zarrLayerRef
}

export const Map = () => {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<MapInstance | null>(null)
  const [map, setMap] = useState<MapInstance | null>(null)
  const [isMapLoaded, setIsMapLoaded] = useState(false)

  const sidebarWidth = useAppStore((state) => state.sidebarWidth)
  const mapProvider = useAppStore((state) => state.mapProvider)
  const globeProjection = useAppStore((state) => state.globeProjection)
  const terrainEnabled = useAppStore((state) => state.terrainEnabled)
  const loadingState = useAppStore((state) => state.loadingState)
  const setMapInstance = useAppStore((state) => state.setMapInstance)

  const mapConfig = getMapConfig(mapProvider)

  useMapLayer(map, isMapLoaded)

  useEffect(() => {
    if (!mapContainer.current) return

    const newMap = mapConfig.createMap(mapContainer.current, globeProjection)
    mapInstanceRef.current = newMap

    newMap.on('load', () => {
      setMap(newMap)
      setIsMapLoaded(true)
      setMapInstance(newMap)
    })

    return () => {
      if (mapInstanceRef.current) {
        try {
          mapInstanceRef.current.remove()
        } catch (error) {
          console.warn('Error removing map:', error)
        }
        setMapInstance(null)
        mapInstanceRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!map || !isMapLoaded) return
    mapConfig.setProjection(map, globeProjection)
  }, [map, isMapLoaded, globeProjection])

  // Toggle terrain (Mapbox only - MapLibre doesn't support terrain draping for custom layers)
  useEffect(() => {
    if (!map || !isMapLoaded || mapProvider !== 'mapbox') return
    const mapboxMap = map as mapboxgl.Map
    try {
      if (terrainEnabled) {
        mapboxMap.setTerrain({ source: 'mapbox-dem', exaggeration: 1.5 })
      } else {
        mapboxMap.setTerrain(null)
      }
    } catch (e) {
      console.warn('Error toggling terrain:', e)
    }
  }, [map, isMapLoaded, terrainEnabled, mapProvider])

  useEffect(() => {
    if (!map || !isMapLoaded) return
    if (map.resize) {
      map.resize()
    }
  }, [map, isMapLoaded, sidebarWidth])

  return (
    <>
      <Box
        ref={mapContainer}
        sx={{
          position: 'absolute',
          top: 0,
          right: 0,
          bottom: 0,
          left: sidebarWidth,
        }}
      />
      <Box sx={{ position: 'absolute', top: '8px', left: sidebarWidth + 10 }}>
        {loadingState.loading && <Spinner size={40} />}
      </Box>
    </>
  )
}
