import React, { useEffect, useRef, useState } from 'react'
import { Box } from 'theme-ui'
import { useThemedColormap } from '@carbonplan/colormaps'
import { ZarrLayer } from 'zarr-maplibre'
import maplibregl from 'maplibre-gl'
import mapboxgl from 'mapbox-gl'
import { layers, namedFlavor } from '@protomaps/basemaps'
import { Protocol } from 'pmtiles'
import { DatasetConfig } from '../lib/constants'

export type MapProvider = 'maplibre' | 'mapbox'

const backgroundColor = '#1b1e23'
const mapLibreTheme = {
  ...namedFlavor('black'),
  buildings: 'rgba(0, 0, 0, 0)',
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

export const combinedBandsCustomFrag = `
  uniform float u_precipWeight;
  float combined = tavg + prec * u_precipWeight;
  float norm = (combined - clim.x) / (clim.y - clim.x);
  float cla = clamp(norm, 0.0, 1.0);
  vec4 c = texture(colormap, vec2(cla, 0.5));
  fragColor = vec4(c.r, c.g, c.b, opacity);
`

export interface MapProps {
  sidebarWidth: number
  datasetId: string
  dataset: DatasetConfig
  opacity: number
  clim: [number, number]
  colormap: string
  time: number
  band: string
  month: number
  precipWeight: number
  globeProjection: boolean
  mapProvider: MapProvider
}

export interface MapInstance {
  on(event: string, callback: () => void): void
  remove(): void
  getLayer(id: string): any
  removeLayer(id: string): void
  addLayer(layer: any, beforeId?: string): void
  setProjection(projection: any): void
  resize?(): void
  flyTo(options: { center: [number, number]; zoom: number }): void
}

export interface MapConfig {
  createMap: (
    container: HTMLDivElement,
    globeProjection: boolean,
  ) => MapInstance
  setProjection: (map: MapInstance, globeProjection: boolean) => void
  getLayerBeforeId: (map: MapInstance) => string | undefined
  needsResize?: boolean
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
      zoom: 2,
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

    return new mapboxgl.Map({
      container,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [0, 20],
      zoom: 2,
      projection: globeProjection ? 'globe' : 'mercator',
    }) as unknown as MapInstance
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
  needsResize: true,
}

export const getMapConfig = (provider: MapProvider): MapConfig => {
  return provider === 'mapbox' ? mapboxConfig : mapLibreConfig
}

export const useMapLayer = (
  map: MapInstance | null,
  isMapLoaded: boolean,
  props: MapProps,
) => {
  const zarrLayerRef = useRef<ZarrLayer | null>(null)
  const colormapArray = useThemedColormap(props.colormap, { format: 'hex' })

  const buildSelector = () => {
    const isCombined = props.band === 'combined'
    if (props.dataset.has4D) {
      if (isCombined) {
        return { band: ['tavg', 'prec'], month: props.month }
      } else {
        return { band: props.band, month: props.month }
      }
    } else {
      return { time: props.time }
    }
  }

  useEffect(() => {
    if (!map || !isMapLoaded) return

    const mapConfig = getMapConfig(props.mapProvider)
    const isCombined = props.band === 'combined'

    if (zarrLayerRef.current) {
      try {
        if (map.getLayer('zarr-layer')) {
          map.removeLayer('zarr-layer')
        }
      } catch (e) {}
      zarrLayerRef.current = null
    }

    const selector = buildSelector()

    const options: any = {
      id: 'zarr-layer',
      source: props.dataset.source,
      variable: props.dataset.variable,
      clim: props.clim,
      colormap: colormapArray,
      opacity: props.opacity,
      selector: selector,
      zarrVersion: props.dataset.zarrVersion,
      minRenderZoom: props.dataset.minRenderZoom ?? 0,
      fillValue: props.dataset.fillValue,
      dimensionNames: props.dataset.dimensionNames,
    }

    if (isCombined) {
      options.customFrag = combinedBandsCustomFrag
      options.uniforms = { u_precipWeight: props.precipWeight }
    }

    try {
      const layer = new ZarrLayer(options)
      let beforeId: string | undefined
      try {
        beforeId = mapConfig.getLayerBeforeId(map)
      } catch (e) {}
      map.addLayer(layer, beforeId)
      zarrLayerRef.current = layer

      if (props.dataset.center) {
        map.flyTo({
          center: props.dataset.center,
          zoom: props.dataset.zoom || 4,
        })
      }
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
        zarrLayerRef.current = null
      }
    }
  }, [map, isMapLoaded, props.datasetId, props.band, colormapArray])

  useEffect(() => {
    const layer = zarrLayerRef.current
    if (!layer || !map || !isMapLoaded) return

    layer.setOpacity(props.opacity)
    layer.setColormap(colormapArray)
    layer.setClim(props.clim)

    const selector = buildSelector()
    layer.setSelector(selector)

    if (props.band === 'combined') {
      layer.setUniforms({ u_precipWeight: props.precipWeight })
    }
  }, [
    props.opacity,
    props.clim,
    colormapArray,
    props.time,
    props.band,
    props.month,
    props.precipWeight,
    props.dataset,
    map,
    isMapLoaded,
  ])

  return zarrLayerRef
}

export const MapComponentBase = ({
  sidebarWidth,
  mapProvider,
  ...props
}: MapProps) => {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<MapInstance | null>(null)
  const [map, setMap] = useState<MapInstance | null>(null)
  const [isMapLoaded, setIsMapLoaded] = useState(false)

  const mapConfig = getMapConfig(mapProvider)

  useMapLayer(map, isMapLoaded, { ...props, mapProvider })

  useEffect(() => {
    if (!mapContainer.current) return

    const newMap = mapConfig.createMap(
      mapContainer.current,
      props.globeProjection,
    )
    mapInstanceRef.current = newMap

    newMap.on('load', () => {
      setMap(newMap)
      setIsMapLoaded(true)
    })

    return () => {
      if (mapInstanceRef.current) {
        try {
          mapInstanceRef.current.remove()
        } catch (error) {
          console.warn('Error removing map:', error)
        }
        mapInstanceRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!map || !isMapLoaded) return
    mapConfig.setProjection(map, props.globeProjection)
  }, [map, isMapLoaded, props.globeProjection])

  useEffect(() => {
    if (!map || !isMapLoaded || !mapConfig.needsResize) return
    if (map.resize) {
      map.resize()
    }
  }, [map, isMapLoaded, sidebarWidth])

  return (
    <Box
      ref={mapContainer}
      sx={{
        position: 'absolute',
        top: 0,
        right: 0,
        bottom: 0,
        left: sidebarWidth,
        transition: 'left 0.2s',
      }}
    />
  )
}
