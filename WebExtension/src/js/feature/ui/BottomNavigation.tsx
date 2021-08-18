import React from 'react'
import { useHistory } from 'react-router-dom'
import styled from 'styled-components'
import HomeSVG from '../../../resources/home-24.svg'
import ListSVG from '../../../resources/list-24.svg'
import SettingsSVG from '../../../resources/settings-24.svg'

export const BottomNavigation = () => {
  const history = useHistory()

  const handleHomeClick = () => {
    console.log('Home clicked')
    history.push('/home')
  }

  const handleListClick = () => {
    console.log('List clicked')
    history.push('/list')
  }

  const handleSettingsClick = () => {
    console.log('Settings clicked')
    history.push('/settings')
  }

  return (
    <BottomBar>
      <Row>
        <Line />
      </Row>
      <Row>
        <Item src={HomeSVG} alt={'Missing'} onClick={handleHomeClick} />
        <Item src={ListSVG} alt={'Missing'} onClick={handleListClick} />
        <Item src={SettingsSVG} alt={'Missing'} onClick={handleSettingsClick} />
      </Row>
    </BottomBar>
  )
}

const BottomBar = styled.div`
  display: flex;
  flex-direction: column;
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background: white;
`

const Row = styled.div`
  display: flex;
  flex-direction: row;
  width: 100%;
`

const Line = styled.div`
  flex: 1;
  background: #b2b2b2;
  height: 1px;
  color: #b2b2b2;
`

const Item = styled.img`
  flex: 1;
  width: 40px;
  height: 40px;
  padding: 10px;

  &:hover {
    background: lightgrey;
  }
`
