// @ts-expect-error - carbonplan components types not available
import { Filter, Select, Slider } from '@carbonplan/components'
import { Box } from 'theme-ui'
import { DATASETS, DatasetConfig } from '../lib/constants'
import { combinedBandsCustomFrag, MapProvider } from './map-shared'

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

interface ControlsProps {
  datasetId: string
  setDatasetId: (id: string) => void
  opacity: number
  setOpacity: (opacity: number) => void
  clim: [number, number]
  setClim: (clim: [number, number]) => void
  colormap: string
  setColormap: (colormap: string) => void
  time: number
  setTime: (time: number) => void
  band: string
  setBand: (band: string) => void
  month: number
  setMonth: (month: number) => void
  precipWeight: number
  setPrecipWeight: (weight: number) => void
  globeProjection: boolean
  setGlobeProjection: (globeProjection: boolean) => void
  mapProvider: MapProvider
  setMapProvider: (provider: MapProvider) => void
  dataset: DatasetConfig
}

const Controls = ({
  datasetId,
  setDatasetId,
  opacity,
  setOpacity,
  clim,
  setClim,
  colormap,
  setColormap,
  time,
  setTime,
  band,
  setBand,
  month,
  setMonth,
  precipWeight,
  setPrecipWeight,
  globeProjection,
  setGlobeProjection,
  mapProvider,
  setMapProvider,
  dataset,
}: ControlsProps) => {
  const handleDatasetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newId = e.target.value
    setDatasetId(newId)
    const config = DATASETS[newId]

    setClim(config.clim)
    setColormap(config.colormap)

    if (config.has4D) {
      setBand('tavg')
      setMonth(1)
    } else {
      setTime(0)
    }
  }

  const handleBandChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newBand = e.target.value
    setBand(newBand)
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Box>
        Dataset
        <Box sx={{ display: 'block', mt: 1, width: '100%' }}>
          <Select value={datasetId} onChange={handleDatasetChange}>
            {Object.entries(DATASETS).map(([key, config]) => (
              <option key={key} value={key}>
                {config.info}
              </option>
            ))}
          </Select>
        </Box>
      </Box>

      {dataset.has4D ? (
        <>
          <Box>
            <Box>Band</Box>
            <Select value={band} onChange={handleBandChange}>
              <option value='tavg'>tavg</option>
              <option value='prec'>prec</option>
              <option value='combined'>
                combined (custom frag w/ uniform)
              </option>
            </Select>
            {band === 'combined' && (
              <Box
                as='code'
                sx={{ fontSize: 0, color: 'secondary', whiteSpace: 'pre-wrap' }}
              >
                {combinedBandsCustomFrag}
              </Box>
            )}
          </Box>

          <Box>
            Month: {month}
            <Slider
              min={1}
              max={12}
              step={1}
              value={month}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setMonth(parseInt(e.target.value))
              }
            />
          </Box>

          {band === 'combined' && (
            <Box>
              Precip Weight: {precipWeight}
              <Slider
                min={0}
                max={5}
                step={0.1}
                value={precipWeight}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setPrecipWeight(parseFloat(e.target.value))
                }
              />
            </Box>
          )}
        </>
      ) : (
        <Box>
          Time Index: {time}
          <Slider
            min={0}
            max={10}
            step={1}
            value={time}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setTime(parseInt(e.target.value))
            }
          />
        </Box>
      )}

      <Box>
        Opacity: {opacity}
        <Slider
          min={0}
          max={1}
          step={0.01}
          value={opacity}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setOpacity(parseFloat(e.target.value))
          }
        />
      </Box>

      <Box>
        Min value: {clim[0]}
        <Slider
          min={dataset.clim[0] - (dataset.clim[1] - dataset.clim[0]) * 0.5}
          max={dataset.clim[1]}
          step={(dataset.clim[1] - dataset.clim[0]) / 100}
          value={clim[0]}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setClim([parseFloat(e.target.value), clim[1]])
          }
        />
      </Box>

      <Box>
        Max value: {clim[1]}
        <Slider
          min={dataset.clim[0]}
          max={dataset.clim[1] + (dataset.clim[1] - dataset.clim[0]) * 0.5}
          step={(dataset.clim[1] - dataset.clim[0]) / 100}
          value={clim[1]}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setClim([clim[0], parseFloat(e.target.value)])
          }
        />
      </Box>

      <Box>
        Projection
        <Filter
          values={{ mercator: !globeProjection, globe: globeProjection }}
          setValues={(obj: Record<string, boolean>) => {
            if (obj.mercator) setGlobeProjection(false)
            if (obj.globe) setGlobeProjection(true)
          }}
        />
      </Box>

      <Box>
        Map provider
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
      </Box>

      <Box>
        Colormap
        <Box sx={{ display: 'block', mt: 1, width: '100%' }}>
          <Select
            value={colormap}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
              setColormap(e.target.value)
            }
          >
            {colormaps.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </Select>
        </Box>
      </Box>
    </Box>
  )
}

export default Controls
