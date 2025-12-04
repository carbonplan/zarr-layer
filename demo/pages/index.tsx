import React, { useState } from 'react'
import { Box, Container } from 'theme-ui'
// @ts-expect-error - carbonplan components types not available
import { Dimmer, Header } from '@carbonplan/components'
import Map from '../components/map'
import Sidebar from '../components/sidebar'
import { DATASETS } from '../lib/constants'
import { MapProvider } from '../components/map-shared'

export default function Home() {
  const [sidebarWidth, setSidebarWidth] = useState(0)

  const [datasetId, setDatasetId] = useState('salinity_v2')
  const [opacity, setOpacity] = useState(1)
  const [clim, setClim] = useState<[number, number]>([30, 37])
  const [colormap, setColormap] = useState('warm')
  const [time, setTime] = useState(0)
  const [band, setBand] = useState('tavg')
  const [month, setMonth] = useState(1)
  const [precipWeight, setPrecipWeight] = useState(1.0)
  const [globeProjection, setGlobeProjection] = useState(true)
  const [mapProvider, setMapProvider] = useState<MapProvider>('maplibre')

  const dataset = DATASETS[datasetId]

  return (
    <>
      <Container>
        <Box sx={{ position: 'relative', zIndex: 2000 }}>
          <Header
            menuItems={[
              <Dimmer key='dimmer' sx={{ mt: '-2px', color: 'primary' }} />,
            ]}
          />
        </Box>
      </Container>

      <Box
        sx={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          width: '100vw',
        }}
      >
        <Sidebar
          onSidebarWidthChange={setSidebarWidth}
          datasetId={datasetId}
          setDatasetId={setDatasetId}
          opacity={opacity}
          setOpacity={setOpacity}
          clim={clim}
          setClim={setClim}
          colormap={colormap}
          setColormap={setColormap}
          time={time}
          setTime={setTime}
          band={band}
          setBand={setBand}
          month={month}
          setMonth={setMonth}
          precipWeight={precipWeight}
          setPrecipWeight={setPrecipWeight}
          globeProjection={globeProjection}
          setGlobeProjection={setGlobeProjection}
          mapProvider={mapProvider}
          setMapProvider={setMapProvider}
          dataset={dataset}
        />
        <Map
          key={mapProvider}
          sidebarWidth={sidebarWidth}
          datasetId={datasetId}
          dataset={dataset}
          opacity={opacity}
          clim={clim}
          colormap={colormap}
          time={time}
          band={band}
          month={month}
          precipWeight={precipWeight}
          globeProjection={globeProjection}
          mapProvider={mapProvider}
        />
      </Box>
    </>
  )
}
