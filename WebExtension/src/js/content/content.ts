import { convertToMessage, copiedClipboard, CopiedMessage, copiedSelection, copiedSite, enqueuePopup, Message, requestTitles, sendMessageToBackground } from '~/types/Message'


export function onMessageReceived(message: Message): Promise<CopiedMessage> {

  if (message.title == requestTitles.selectionCopyRequested) {
    return new Promise((resolve, reject) => {
      let selObj = window.getSelection()

      if (selObj) {
        resolve(copiedSelection(selObj.toString()))
      } else {
        reject('Selection null')
      }
    })
  } else if (message.title === requestTitles.clipboardCopyRequested) {
    return new Promise<CopiedMessage>((resolve) => {
      navigator.clipboard.readText().then(response => {
        resolve(copiedClipboard(response))
      }).catch(err => {
        console.log({ err })
      })
    })
  } else if (message.title == requestTitles.siteCopyRequested) {
    return new Promise<CopiedMessage>(resolve => {
      resolve(copiedSite(document.documentElement.innerHTML))
    })
  } else {
    throw new Error(`unknown message ${JSON.stringify(message)}`)
  }
}

function setup(): void {
  console.log('Setting up content script')

  // setup listeners to send messages to the background script
  if (process.env.TARGET === 'chrome') {
    chrome.runtime.onMessage.addListener(async (data: unknown) => {
      onMessageReceived(convertToMessage(data))
        .then((msg: Message) => {
          sendMessageToBackground(msg)
        }).catch(err => {
        console.log(`Error copying selection: ${err}`)
      })
    })
  } else {
    browser.runtime.onMessage.addListener(async (data: unknown) => {
      onMessageReceived(convertToMessage(data))
        .then((msg: Message) => {
          sendMessageToBackground(msg);
        })
    })
  }
}

setup()