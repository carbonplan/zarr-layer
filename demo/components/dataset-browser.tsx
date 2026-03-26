import React, { useState } from 'react'
import { Box, Flex, IconButton } from 'theme-ui'
import { Info } from '@carbonplan/icons'
import { DATASET_MAP } from '../datasets'
import { SELECTOR_SECTIONS } from '../datasets/sections'
import { useAppStore } from '../lib/store'

const InfoButton = ({
  expanded,
  onClick,
}: {
  expanded: boolean
  onClick: () => void
}) => (
  <IconButton
    onClick={onClick}
    aria-label='More information'
    aria-expanded={expanded}
    sx={{
      cursor: 'pointer',
      width: '16px',
      height: '16px',
      p: 0,
      flexShrink: 0,
      '@media (hover: hover) and (pointer: fine)': {
        '&:hover > #info': { stroke: 'primary' },
      },
    }}
  >
    <Info
      id='info'
      height='16px'
      width='16px'
      sx={{
        stroke: expanded ? 'primary' : 'secondary',
        transition: '0.1s',
      }}
    />
  </IconButton>
)

const DatasetBrowser = () => {
  const datasetId = useAppStore((state) => state.datasetId)
  const setDatasetId = useAppStore((state) => state.setDatasetId)
  const datasetModule = useAppStore((state) => state.getDatasetModule())
  const datasetState = useAppStore((state) => state.getDatasetState())
  const setActiveDatasetState = useAppStore(
    (state) => state.setActiveDatasetState
  )
  const [expandedInfos, setExpandedInfos] = useState<Set<string>>(new Set())

  const toggleInfo = (label: string) => {
    setExpandedInfos((prev) => {
      const next = new Set(prev)
      if (next.has(label)) next.delete(label)
      else next.add(label)
      return next
    })
  }

  const ActiveDatasetControls = datasetModule.Controls

  return (
    <Box sx={{ mt: 3 }}>
      {SELECTOR_SECTIONS.map((section) => {
        const infoExpanded = expandedInfos.has(section.label)

        return (
          <Box key={section.label} sx={{ mb: 2 }}>
            <Flex sx={{ alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  fontFamily: 'mono',
                  letterSpacing: 'smallcaps',
                  textTransform: 'uppercase',
                  fontSize: [2, 2, 2, 2],
                }}
              >
                {section.label}
              </Box>
              <InfoButton
                expanded={infoExpanded}
                onClick={() => toggleInfo(section.label)}
              />
            </Flex>
            {infoExpanded && (
              <Box
                sx={{
                  fontSize: 0,
                  color: 'secondary',
                  mt: 1,
                  mb: 2,
                  fontFamily: 'body',
                }}
              >
                {section.description}
              </Box>
            )}

            <Box sx={{ mt: 1, mx: [-4, -5, -5, -6] }}>
              {section.datasetIds.map((id) => {
                const config = DATASET_MAP[id]
                if (!config) return null
                const isActive = id === datasetId

                return (
                  <Box
                    key={id}
                    sx={{
                      bg: isActive ? 'hinted' : 'transparent',
                      px: [4, 5, 5, 6],
                      py: isActive ? 2 : '2px',
                      transition: 'background-color 0.15s',
                      '@media (hover: hover) and (pointer: fine)': {
                        '&:hover': {
                          bg: 'hinted',
                        },
                      },
                    }}
                  >
                    <Box
                      onClick={() => setDatasetId(id)}
                      sx={{
                        color: isActive ? 'primary' : 'secondary',
                        fontSize: [1, 1, 1, 1],
                        cursor: 'pointer',
                        '@media (hover: hover) and (pointer: fine)': {
                          '&:hover': { color: 'primary' },
                        },
                      }}
                    >
                      <Box>
                        {config.info}
                        {isActive && config.sourceInfo && (
                          <Box
                            sx={{
                              color: 'secondary',
                              fontSize: 0,
                              mt: '2px',
                            }}
                          >
                            {config.sourceInfo}
                          </Box>
                        )}
                      </Box>
                    </Box>
                    {isActive && (
                      <Box sx={{ ml: 1 }}>
                        <ActiveDatasetControls
                          state={datasetState}
                          setState={setActiveDatasetState}
                        />
                      </Box>
                    )}
                  </Box>
                )
              })}
            </Box>
          </Box>
        )
      })}
    </Box>
  )
}

export default DatasetBrowser
