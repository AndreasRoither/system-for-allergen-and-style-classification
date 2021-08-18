import React, { useEffect, useState } from 'react'
import styled from 'styled-components'

import DeleteSVG from '../../../resources/delete-24.svg'

export const ListScreen = () => {
  const [filterList, setFilterList] = useState<string[]>([])
  const [currentInput, setCurrentInput] = useState<string>('')

  const filter = () => {
    let filter, li, a, i, txtValue
    let input = document.getElementById('searchInput') as HTMLInputElement
    let ul = document.getElementById('filterList') as HTMLUListElement

    setCurrentInput(input.value)
    filter = input.value.toUpperCase()
    li = ul.getElementsByTagName('li')

    for (i = 0; i < li.length; i++) {
      a = li[i].getElementsByTagName('a')[0]
      txtValue = a.textContent || a.innerText

      if (txtValue.toUpperCase().indexOf(filter) > -1) {
        li[i].style.display = ''
      } else {
        li[i].style.display = 'none'
      }
    }
  }

  const remove = (text: string) => {
    setFilterList(filterList => filterList.filter(item => item !== text))
  }

  const addToFilterList = () => {
    setFilterList(filterList.concat(currentInput))
  }

  useEffect(() => {
    const obj = localStorage.getItem('filter-list')

    if (obj == null) {
      setFilterList([])
    } else {
      setFilterList(JSON.parse(obj))
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('filter-list', JSON.stringify(filterList))
  })

  return (
    <>
      <Line />
      <ContentContainer>

        <SearchContainer>
          <Search type="text" id="searchInput" onKeyUp={filter} placeholder="Filter" title="Type in a filter" />
          <Add onClick={addToFilterList}>Add</Add>
        </SearchContainer>

        {
          filterList.length == 0 &&
          <Empty>Nothing saved yet.</Empty>
        }

        <List id="filterList">
          {
            filterList.map((value, index) => {
              return (
                <ListItem key={index}>
                  <ListItemContainer>
                    <a href="#">{value}</a>
                    <Delete onClick={() => remove(value)}>
                      <img src={DeleteSVG} alt="Delete" />
                    </Delete>
                  </ListItemContainer>
                  <ListItemLine />
                </ListItem>
              )
            })
          }
        </List>
      </ContentContainer>
    </>
  )
}

const ContentContainer = styled.div`
  margin-top: 10px;
  display: flex;
  flex-direction: column;
`

const SearchContainer = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  margin-bottom: 15px;
`

const Search = styled.input`
  margin: 10px 10px 0 10px;
  padding: 12px 20px 12px 20px;
  width: 100%;
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

const Empty = styled.div`
  margin: 15px;
  text-align: center;
`

const List = styled.ul`
  list-style-type: none;
  padding: 0;
  margin: 0 10px;
  overflow-y: auto;
  height: 350px;
  scrollbar-width: thin;
`

const ListItem = styled.li`

  &:hover {
    background: none;
  }

  a {
    border: 0 solid #ddd;
    margin-top: -1px; /* Prevent double borders */
    padding: 10px 10px;
    text-decoration: none;
    font-size: 18px;
    color: black;
    display: block
  }
`

const ListItemContainer = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: space-between;
`

const Line = styled.div`
  flex: 1;
  background: #b2b2b2;
  height: 1px;
  color: #b2b2b2;
`

const ListItemLine = styled.div`
  margin: 10px 0;
  flex: 1;
  background: #b2b2b2;
  height: 1px;
  color: #b2b2b2;
`

const Add = styled.button`
  background-color: #008CBA;
  color: white;
  border: none;
  margin: 10px 10px 0 10px;
  padding: 10px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  transition-duration: 0.4s;
  cursor: pointer;
  border-radius: 4px;

  &:hover {
    background-color: #008CBA;
    color: white;
  }
`

const Delete = styled.button`
  border: none;
  background: none;
  border-radius: 4px;

  &:hover {
    background-color: lightgray;
  }
`