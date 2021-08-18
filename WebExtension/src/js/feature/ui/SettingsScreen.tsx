import React from 'react'
import styled from 'styled-components'

export const SettingsScreen = () => {
  return (
    <>
      <Line/>
      <div>
        <ContentContainer>Settings Screen</ContentContainer>
      </div>
    </>
  )
}

const Line = styled.div`
  flex: 1;
  background: #b2b2b2;
  height: 1px;
  color: #b2b2b2;
`

const ContentContainer = styled.div`
  flex-grow: 1;
  left: 1em;
  right: 1em;
`
