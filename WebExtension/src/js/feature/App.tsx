import { MainScreen } from './ui/MainScreen'
import * as React from 'react'
import { Header } from '~/feature/ui/Header'
import { BottomNavigation } from '~/feature/ui/BottomNavigation'

import {
  Redirect,
  BrowserRouter as Router,
  Route,
  Switch,
} from 'react-router-dom'
import { SettingsScreen } from '~/feature/ui/SettingsScreen'
import { ListScreen } from '~/feature/ui/ListScreen'

export const App = () => {
  /*
   * Important! Default route is not "/" since our entrypoint is /feature/extension/extension.html so we either need to redirect or
   * have two routes pointing to the same component, in this case I have placed a redirect too in case i have a wrong route somewhere
   */

  return (
    <>
      <Router>
        <Header />
        <Switch>
          <Route path="/feature/extension/" component={MainScreen} />
          <Route path="/home" component={MainScreen} />
          <Route path="/list" component={ListScreen} />
          <Route path="/settings" component={SettingsScreen} />
          <Redirect to="/home" />
        </Switch>
        <BottomNavigation />
      </Router>
    </>
  )
}
