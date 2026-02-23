import React from 'react'
import type { AppProps } from 'next/app'
import Script from 'next/script'
import { ThemeProvider } from 'theme-ui'
// @ts-expect-error - carbonplan theme types not available
import theme from '@carbonplan/theme'
import '@carbonplan/components/fonts.css'
import '@carbonplan/components/globals.css'
import 'maplibre-gl/dist/maplibre-gl.css'
import 'mapbox-gl/dist/mapbox-gl.css'

const App = ({ Component, pageProps }: AppProps) => {
  return (
    <ThemeProvider theme={theme}>
      {process.env.NEXT_PUBLIC_VERCEL_ENV === 'production' && (
        <Script
          data-domain='carbonplan.org'
          data-api='https://carbonplan.org/proxy/api/event'
          src='https://carbonplan.org/js/script.file-downloads.outbound-links.js'
        />
      )}
      <Component {...pageProps} />
    </ThemeProvider>
  )
}

export default App
