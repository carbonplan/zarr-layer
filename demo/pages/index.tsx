import React, { useEffect } from 'react'
import { useRouter } from 'next/router'
import { Box, Container } from 'theme-ui'
// @ts-expect-error - carbonplan components types not available
import { Header, Meta } from '@carbonplan/components'
import { Map } from '../components/map-shared'
import Sidebar from '../components/sidebar'
import { useAppStore } from '../lib/store'
import { DATASET_MAP } from '../datasets'

export default function Home() {
  const router = useRouter()
  const mapProvider = useAppStore((state) => state.mapProvider)
  const sidebarWidth = useAppStore((state) => state.sidebarWidth)
  // Two-way sync: URL query param to store, store to URL
  useEffect(() => {
    if (!router.isReady) return

    // URL to store
    const urlDataset = router.query.dataset
    if (typeof urlDataset === 'string' && DATASET_MAP[urlDataset]) {
      useAppStore.getState().setDatasetId(urlDataset)
    }

    // Store to URL
    return useAppStore.subscribe((state, prev) => {
      if (state.datasetId !== prev.datasetId) {
        router.replace({ query: { dataset: state.datasetId } }, undefined, {
          shallow: true,
        })
      }
    })
  }, [router.isReady]) // eslint-disable-line react-hooks/exhaustive-deps

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
          left: 0,
          right: 0,
          overflow: 'hidden',
        }}
      >
        <Sidebar />
        {sidebarWidth !== null && <Map key={mapProvider} />}
      </Box>
    </>
  )
}
