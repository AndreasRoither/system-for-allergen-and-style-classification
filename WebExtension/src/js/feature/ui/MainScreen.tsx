import React, { useState } from 'react'
import styled from 'styled-components'
import { PredictRequest, PredictTypeRequest } from '~/types/PredictRequest'
import { PredictResponseItem } from '~/types/PredictResponseItem'
import PredictionApi from '~/api/PredictionApi'
import PowerSVG from '../../../resources/power_settings_new-black-48.svg'
import { PredictProbaResponse } from '~/types/PredictProbaResponse'

export const MainScreen = () => {
  const [site, setSite] = useState<string>('')
  const [probaResponse, setProbaResponse] = useState<PredictProbaResponse>()
  const [predictionTitle, setPredictionTitle] = useState<string>('')
  let [currentSelection, setCurrentSelection] = useState<number>(0)
  let [textInput, setTextInput] = useState<string>('')

  browser.tabs.query({ currentWindow: true, active: true })
    .then((tabs) => {
      let tabURL = tabs[0].url
      if (tabURL) {
        let pathArray = tabURL.split('/')
        setSite(pathArray[2])
      }
    })

  const onButtonClick = () => {

    if (textInput.length == 0) {
      setPredictionTitle('No request text!')
      return
    }

    switch (currentSelection) {
      case 0:
        let styleReq: PredictRequest = {
          data: textInput,
          model: 'lr',
        }
        requestStylePrediction(styleReq)
        break
      case 1:
        let allergenReq: PredictRequest = {
          data: textInput,
          model: 'lr',
        }
        requestAllergenPrediction(allergenReq)
        break
      case 2:
        let styleProbaReq: PredictTypeRequest = {
          data: textInput,
          model: 'lr',
          type: 'style',
        }
        requestStyleProba(styleProbaReq)
        break
      case 3:
        let allergenProbaReq: PredictTypeRequest = {
          data: textInput,
          model: 'lr',
          type: 'allergens_single',
        }
        requestAllergenProba(allergenProbaReq)
        break
    }
  }

  const handleChange = (event: any) => {
    setCurrentSelection(+event.target.value)
  }

  const handleInput = (event: any) => {
    setTextInput(event.target.value)
  }

  const requestStylePrediction = (req: PredictRequest) => {
    PredictionApi.predict_style(req)
      .then((res) => {
        const data: PredictResponseItem = res.data

        let PredictResponseItem
        let response: PredictProbaResponse = {
          predictions: [
            PredictResponseItem = {
              name: data.name,
              probability: data.probability,
            },
          ],
          customFilter: [],
        }

        if (data.name.length > 0) {
          setPredictionTitle('Search input style predicted to be:')
          setProbaResponse(response)
        } else {
          setPredictionTitle('No prediction for your search input')
        }

      })
      .catch((ex) => {
        console.log('Error')
        console.log({ ex })
        setPredictionTitle('Error processing request')
      })
  }

  const requestAllergenPrediction = (req: PredictRequest) => {
    PredictionApi.predict_allergens(req)
      .then((res) => {
        const data: PredictProbaResponse = res.data

        if (data.predictions.length > 0) {
          setPredictionTitle('Search input allergen predicted to be:')
          setProbaResponse(data)
        } else {
          setPredictionTitle('No prediction for your search input')
        }
      })
      .catch((ex) => {
        console.log('Error')
        console.log({ ex })
        setPredictionTitle('Error processing request')
      })
  }

  const requestAllergenProba = (req: PredictTypeRequest) => {
    PredictionApi.predict_proba(req)
      .then((res) => {
        const data: PredictProbaResponse = res.data
        if (data.predictions.length > 0) {
          setPredictionTitle('Search input allergen predicted to be:')
          setProbaResponse(data)
        } else {
          setPredictionTitle('No prediction for your search input')
        }
      })
      .catch((ex) => {
        console.log('Error')
        console.log({ ex })
        setPredictionTitle('Error processing request')
      })
  }

  const requestStyleProba = (req: PredictTypeRequest) => {
    PredictionApi.predict_proba(req)
      .then((res) => {
        const data: PredictProbaResponse = res.data

        if (data.predictions.length > 0) {
          setPredictionTitle('Search input style predicted to be:')
          setProbaResponse(data)
        } else {
          setPredictionTitle('No prediction for your search input')
        }
      })
      .catch((ex) => {
        console.log('Error')
        console.log({ ex })
        setPredictionTitle('Error processing request')
      })
  }

  return (
    <>
      <div>
        <Row>
          <Line />
        </Row>
        <Row>
          <PowerContainer>
            <MainContent>
              <Power src={PowerSVG} />
              <SiteText>{site}</SiteText>
            </MainContent>
          </PowerContainer>
        </Row>
        <Row>
          <Line />
        </Row>
        <Row>
          <MainContent>
            <InputContainer>
              <Search type="text" id="searchInput" placeholder="Request text" title="Type in a request text" onChange={handleInput} />

              <Selection>
                <SelectionOptions id="requestSelection" onChange={handleChange}>
                  <option defaultValue="0">Style prediction</option>
                  <option value="1">Allergen prediction</option>
                  <option value="2">Style proba</option>
                  <option value="3">Allergen proba</option>
                </SelectionOptions>

                <Button onClick={onButtonClick}>
                  Send request
                </Button>
              </Selection>
            </InputContainer>

            <PredictionTitle>{predictionTitle}</PredictionTitle>
            <ScrollResult>
              {probaResponse?.predictions.map((value, _) => {
                return (
                  <>
                    <ResultItem>
                      {value.name}: {value.probability}
                    </ResultItem>
                  </>
                )
              })}
            </ScrollResult>
          </MainContent>
        </Row>
      </div>
    </>
  )
}

const MainContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`

const InputContainer = styled.div`
  display: flex;
  margin: 10px 15px;
  flex-direction: column;
`

const Selection = styled.div`
  display: flex;
  flex-direction: row;
`

const SelectionOptions = styled.select`
  border-radius: 4px;
  border: 1px solid #ddd;
  margin: 5px 10px 5px 0;
`

const ScrollResult = styled.div`
  overflow-y: auto;
  height: 100px;
  padding-bottom: 50px;
  scrollbar-width: thin;
`

const Row = styled.div`
  width: 100%;
`

const Line = styled.div`
  flex: 1;
  background: #b2b2b2;
  height: 1px;
  color: #b2b2b2;
`

const PowerContainer = styled.div`
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: center;
`

const Power = styled.img`
  width: 100px;
  height: 100px;
  padding: 5px;

  &:hover {
    background: lightgrey;
  }
`

const SiteText = styled.p`
  font-size: 17px;
  margin: 5px 0 25px 0;

  &:hover {
    background: lightgrey;
  }
`

const PredictionTitle = styled.div`
  font-weight: bold;
  font-size: 20px;

  &:hover {
    background: lightgrey;
  }
`

const Button = styled.button`
  background-color: #008CBA;
  color: white;
  border: none;
  padding: 10px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  transition-duration: 0.4s;
  cursor: pointer;
  border-radius: 4px;

  &:hover {
    background-color: #008CBA;
    color: white;
  }
`

const Search = styled.input`
  padding: 12px 20px 12px 20px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-position: 10px 12px;
  background-repeat: no-repeat;
  font-size: 16px;
  outline: none;

  &:focus {
    border: 1px solid #4e4e4e;
  }
`

const ResultItem = styled.p`
  display: compact;
  margin: 0;
`