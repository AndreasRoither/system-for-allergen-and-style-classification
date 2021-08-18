import * as React from 'react'
import styled from 'styled-components'

import PersonSVG from '../../../resources/person-24.svg'

export const Header = () => {

  return (
    <>
      <HeaderContent>
        <ProfileImage src={PersonSVG} />

        <LoggedInStatusContent>
          <LoggedInText>Logged in as</LoggedInText>
          <EmailStatus>currently not logged in</EmailStatus>

        </LoggedInStatusContent>
      </HeaderContent>
    </>
  )
}

const HeaderContent = styled.div`
  display: flex;
  margin-bottom: 10px;
  justify-content: space-between;
`

const ProfileImage = styled.img`
  border-radius: 50%;
  width: 60px;
  border: 1px solid grey;
  margin-bottom:  auto;
  margin-top:  auto;
  margin-left: 15px;
`

const LoggedInStatusContent = styled.div`
  display: flex;
  flex-direction: column;
  margin: 15px;
`

const LoggedInText = styled.div`
 text-align: right; 
 font-size: 20px;
`

const EmailStatus = styled.div `
  display: inline;
`