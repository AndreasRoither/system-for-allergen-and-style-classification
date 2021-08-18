import { convertToMessage, CopiedMessage, copySelectionRequest, copySiteRequest, enqueuePopup, Message, requestTitles, sendMessageToTab } from '~/types/Message'
import { PredictTypeRequest } from '~/types/PredictRequest'
import PredictionApi from '~/api/PredictionApi'
import { PredictProbaResponse } from '~/types/PredictProbaResponse'


export class BackgroundAPI {
  constructor() {
    this.addCommandListener()
    this.addMessageListener()
  }

  private checkFilterList(text: string): string[] {
    // regardless of browser or request, check if any words that are save in localStorage are contained

    let content = text.toLowerCase()
    let containedFilter: string[] = []
    const obj = localStorage.getItem('filter-list')

    if (obj != null) {
      const list = JSON.parse(obj) as string[]
      list.forEach((value) => {
        if (content.includes(value)) {
          containedFilter.push(value)
        }
      })
    }

    return containedFilter
  }

  private getCurrentTabs(): Promise<number[]> {
    if (process.env.TARGET === 'chrome') {
      return new Promise((resolve) => {
        browser.tabs.query({
          currentWindow: true,
          active: true,
        }).then((tabs: browser.tabs.Tab[]) => {
          let list: number[] = []

          tabs.forEach(tab => {
            if (tab.id) {
              list.push(tab.id)
            }
          })
          resolve(list)
        })
      })
    } else {
      return new Promise((resolve) => {
        chrome.tabs.query({
          currentWindow: true,
          active: true,
        }, ((tabs: chrome.tabs.Tab[]) => {
          let list: number[] = []

          tabs.forEach(tab => {
            if (tab.id) {
              list.push(tab.id)
            }
          })
          resolve(list)
        }))
      })
    }
  }

  public executeCommand(command: string) {
    switch (command) {
      case 'site-scan':
        this.getCurrentTabs().then((tabs: number[]) => {
          tabs.forEach(tabId => {
            sendMessageToTab(copySiteRequest(), tabId)
          })
        })
        break
      case 'selected-content-scan':
        this.getCurrentTabs().then((tabs: number[]) => {
          tabs.forEach(tabId => {
            sendMessageToTab(copySelectionRequest(), tabId)
          })
        })
        break
    }
  }

  private onMessageReceived(message: Message): Promise<PredictProbaResponse> {

    let msg = message as CopiedMessage
    const list = this.checkFilterList(msg.text)

    const req: PredictTypeRequest = {
      data: msg.text,
      model: 'lr',
      type: 'allergens_single',
    }

    return new Promise<PredictProbaResponse>((resolve, reject) => {
      if (message.title == requestTitles.selectionCopied) {
        PredictionApi.predict_allergens(req)
          .then((res) => {
            const data: PredictProbaResponse = res.data
            data.customFilter = list
            console.log('We found this recipe contains:')
            console.log({ data })
            resolve(data)
          })
          .catch((err) => {
            console.log('Error')
            console.log(err.request)
            reject({error: err, customFilter: list})
          })

      } else if (message.title == requestTitles.siteCopied) {
        console.log('Site copied')
        console.log({ msg })

        PredictionApi.predict_allergens(req)
          .then((res) => {
            const data: PredictProbaResponse = res.data
            data.customFilter = list
            console.log('We found this recipe contains:')
            console.log(`${data}`)
            resolve(data)
          })
          .catch((err) => {
            console.log('Error')
            console.log(err.request)
            reject({error: err, customFilter: list})
          })
      }
    })
  }

  private handleMessageResponse(data: unknown, sendResponse: (response?: any) => void): Promise<PredictProbaResponse> | void {
    if (process.env.TARGET === 'chrome') {
      this.onMessageReceived(convertToMessage(data))
        .then((value: PredictProbaResponse) => {
          sendResponse(value)
        })
        .catch(reason => {
          sendResponse(reason)
        })
    } else {
      return this.onMessageReceived(convertToMessage(data))
    }
  }

  private addCommandListener() {
    console.log('Adding command listener')

    if (process.env.TARGET === 'chrome') {
      chrome.commands.onCommand.addListener((command) => {
        this.executeCommand(command)
      })
    } else {
      browser.commands.onCommand.addListener((command) => {
        this.executeCommand(command)
      })
    }
  }

  private addMessageListener() {
    console.log('Adding message listener')

    if (process.env.TARGET === 'chrome') {
      chrome.runtime.onMessage.addListener(async (data: unknown, _sender: chrome.runtime.MessageSender, sendResponse: (response?: any) => void) => {
        this.handleMessageResponse(data, sendResponse)
      })
    } else {
      browser.runtime.onMessage.addListener(async (data: unknown, _sender: browser.runtime.MessageSender, sendResponse: (response?: any) => void) => {
        return this.handleMessageResponse(data, sendResponse)
      })
    }
  }
}
