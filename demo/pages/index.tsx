import React from 'react'
import { Box, Container } from 'theme-ui'
// @ts-expect-error - carbonplan components types not available
import { Header, Meta } from '@carbonplan/components'
import { Map } from '../components/map-shared'
import Sidebar from '../components/sidebar'
import { useAppStore } from '../lib/store'

export default function Home() {
  const mapProvider = useAppStore((state) => state.mapProvider)

  return (
    <>
      <Meta
        description={'@carbonplan/zarr-layer demo'}
        title={'@carbonplan/zarr-layer demo'}
      />
      <Container>
        <Box sx={{ position: 'relative', zIndex: 2000 }}>
          <Header />
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
        <Sidebar />
        <Map key={mapProvider} />
      </Box>
    </>
  )
}
